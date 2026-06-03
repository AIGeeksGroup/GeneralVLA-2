#!/usr/bin/env python3
"""Input-only anisotropic geometry refinement with mesh sync.

This experiment copies an existing MV-SAM3D output tree, optimizes a small
axis-wise affine transform of Gaussian centers using only the fixed input
views, then applies the same transform to result.glb. It is deliberately
low-dimensional: no view selection, no target-view/GT metric feedback, and no
Gaussian deletion.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image


@dataclass(frozen=True)
class AxisTransform:
    name: str
    matrix: np.ndarray


def read_objects(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.strip().startswith("#")]


def parse_views(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def parse_axis_transform(spec: str) -> AxisTransform:
    if spec == "identity":
        return AxisTransform(spec, np.eye(3, dtype=np.float32))
    perm_part, sign_part = spec.split("_sign", 1)
    perm = tuple(int(ch) for ch in perm_part[len("perm"):])
    signs = tuple(1.0 if ch == "p" else -1.0 for ch in sign_part)
    mat = np.zeros((3, 3), dtype=np.float32)
    for out_axis, in_axis in enumerate(perm):
        mat[out_axis, in_axis] = signs[out_axis]
    return AxisTransform(spec, mat)


def to4(mat3x4: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :] = mat3x4[:3, :].astype(np.float32)
    return out


def make_extrinsic(gso_w2c: np.ndarray, object_to_gso: np.ndarray, zflip: bool) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = object_to_gso
    extrinsic = to4(gso_w2c) @ transform
    if zflip:
        extrinsic = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32) @ extrinsic
    return extrinsic.astype(np.float32)


def load_gt(path: Path, resolution: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    rgb_img = Image.fromarray(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0)).convert("RGB")
    mask_img = Image.fromarray(np.uint8(np.clip(alpha[..., 0], 0.0, 1.0) * 255.0)).convert("L")
    if rgb_img.size != (resolution, resolution):
        rgb_img = rgb_img.resize((resolution, resolution), Image.BILINEAR)
        mask_img = mask_img.resize((resolution, resolution), Image.BILINEAR)
    rgb_t = torch.from_numpy(np.asarray(rgb_img, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)
    mask_t = torch.from_numpy(np.asarray(mask_img, dtype=np.float32) / 255.0)[None].to(device)
    return rgb_t, mask_t


def load_gaussian(repo: Path, ply_path: Path):
    sys.path.insert(0, str(repo))
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian

    gaussian = Gaussian(aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], sh_degree=0, device="cuda")
    gaussian.load_ply(str(ply_path))
    gaussian.max_sh_degree = gaussian.sh_degree
    return gaussian


def make_renderer(repo: Path, resolution: int):
    sys.path.insert(0, str(repo))
    from sam3d_objects.model.backbone.tdfy_dit.renderers import GaussianRenderer

    renderer = GaussianRenderer(
        {"resolution": resolution, "near": 0.01, "far": 100.0, "ssaa": 1, "bg_color": [1.0, 1.0, 1.0], "backend": "gsplat"}
    )
    renderer.pipe.convert_SHs_python = True
    return renderer


def render_torch(renderer, gaussian, extrinsic: np.ndarray, focal: float) -> torch.Tensor:
    extr = torch.tensor(extrinsic, dtype=torch.float32, device="cuda")
    intr = torch.tensor([[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda")
    return renderer.render(gaussian, extr, intr).color.clamp(0.0, 1.0)


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    c = image.shape[0]
    batch = image.unsqueeze(0)
    gx = F.conv2d(batch, kx.repeat(c, 1, 1, 1), padding=1, groups=c)
    gy = F.conv2d(batch, ky.repeat(c, 1, 1, 1), padding=1, groups=c)
    return torch.sqrt(gx.square() + gy.square() + 1.0e-8).squeeze(0)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pred_b = pred.unsqueeze(0)
    target_b = target.unsqueeze(0)
    mu_p = F.avg_pool2d(pred_b, 11, 1, 5, count_include_pad=False)
    mu_t = F.avg_pool2d(target_b, 11, 1, 5, count_include_pad=False)
    sig_p = F.avg_pool2d(pred_b * pred_b, 11, 1, 5, count_include_pad=False) - mu_p.square()
    sig_t = F.avg_pool2d(target_b * target_b, 11, 1, 5, count_include_pad=False) - mu_t.square()
    sig_pt = F.avg_pool2d(pred_b * target_b, 11, 1, 5, count_include_pad=False) - mu_p * mu_t
    c1, c2 = 0.01**2, 0.03**2
    val = ((2.0 * mu_p * mu_t + c1) * (2.0 * sig_pt + c2)) / ((mu_p.square() + mu_t.square() + c1) * (sig_p + sig_t + c2) + 1e-8)
    return ((1.0 - val.clamp(-1.0, 1.0)) * weight.unsqueeze(0)).mean()


def copy_base_output(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_dir / item.name)


def affine_vertices(vertices: np.ndarray, center: np.ndarray, axis_scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    return center.reshape(1, 3) + (vertices - center.reshape(1, 3)) * axis_scale.reshape(1, 3) + shift.reshape(1, 3)


def sync_mesh(mesh_path: Path, center: np.ndarray, axis_scale: np.ndarray, shift: np.ndarray) -> str:
    mesh_or_scene = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh_or_scene, trimesh.Scene):
        touched = 0
        for geom in mesh_or_scene.geometry.values():
            if hasattr(geom, "vertices") and len(geom.vertices):
                geom.vertices = affine_vertices(np.asarray(geom.vertices), center, axis_scale, shift)
                touched += 1
        mesh_or_scene.export(str(mesh_path))
        return f"scene_geoms={touched}"
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh_or_scene.vertices = affine_vertices(np.asarray(mesh_or_scene.vertices), center, axis_scale, shift)
        mesh_or_scene.export(str(mesh_path))
        return "mesh"
    raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)!r}")


def optimize_one(args: argparse.Namespace, obj: str, renderer, transform: AxisTransform) -> dict[str, str]:
    src_dir = args.pred_root / obj / f"{args.views}views"
    dst_dir = args.output_root / obj / f"{args.views}views"
    if not (src_dir / "result.ply").exists():
        return {"object": obj, "status": "missing_result_ply"}
    copy_base_output(src_dir, dst_dir)
    gaussian = load_gaussian(args.repo, dst_dir / "result.ply")

    base_xyz = gaussian._xyz.detach().clone()
    base_dc = gaussian._features_dc.detach().clone()
    base_opacity = gaussian._opacity.detach().clone()
    center = base_xyz.mean(dim=0, keepdim=True)
    axis_log = torch.nn.Parameter(torch.zeros(1, 3, device="cuda"))
    shift = torch.nn.Parameter(torch.zeros(1, 3, device="cuda"))
    color_scale = torch.nn.Parameter(torch.ones(1, 1, 3, device="cuda"))
    color_bias = torch.nn.Parameter(torch.zeros(1, 1, 3, device="cuda"))
    opacity_delta = torch.nn.Parameter(torch.zeros(1, 1, device="cuda"))

    train_data = []
    for view in args.train_views:
        stem = f"{view:03d}"
        gt_path = args.gso_root / obj / args.render_dir / "model" / f"{stem}.png"
        cam_path = args.gso_root / obj / args.render_dir / "model" / f"{stem}.npy"
        if gt_path.exists() and cam_path.exists():
            rgb, mask = load_gt(gt_path, args.resolution, torch.device("cuda"))
            extr = make_extrinsic(np.load(cam_path).astype(np.float32), transform.matrix, args.nvs_zflip)
            train_data.append((view, rgb, mask, sobel_edges(rgb), extr))
    if not train_data:
        return {"object": obj, "status": "missing_training_views"}

    params = [axis_log, shift, color_scale, color_bias, opacity_delta]
    opt = torch.optim.Adam(params, lr=args.lr)
    best_loss = float("inf")
    best_state = tuple(p.detach().clone() for p in params)
    last_loss = None
    for step_idx in range(args.steps):
        state_before_step = tuple(p.detach().clone() for p in params)
        with torch.no_grad():
            axis_log.clamp_(np.log(args.min_axis_scale), np.log(args.max_axis_scale))
            shift.clamp_(-args.max_shift, args.max_shift)
            color_scale.clamp_(args.min_color_scale, args.max_color_scale)
            color_bias.clamp_(-args.max_color_bias, args.max_color_bias)
            opacity_delta.clamp_(-args.max_opacity_delta, args.max_opacity_delta)
        axis_scale = torch.exp(axis_log)
        gaussian._xyz = center + (base_xyz - center) * axis_scale + shift
        gaussian._features_dc = base_dc * color_scale + color_bias
        gaussian._opacity = base_opacity + opacity_delta
        loss = torch.zeros((), device="cuda")
        for _, rgb, mask, rgb_edges, extr in train_data:
            pred = render_torch(renderer, gaussian, extr, args.nvs_focal)
            weight = args.bg_weight + (1.0 - args.bg_weight) * mask
            image_loss = (torch.abs(pred - rgb) * weight).mean()
            if args.edge_weight:
                image_loss = image_loss + args.edge_weight * (torch.abs(sobel_edges(pred) - rgb_edges) * weight).mean()
            if args.ssim_weight:
                image_loss = image_loss + args.ssim_weight * ssim_loss(pred, rgb, weight)
            loss = loss + image_loss
        loss = loss / len(train_data)
        loss = loss + args.axis_reg * (axis_scale - 1.0).square().mean()
        loss = loss + args.shift_reg * shift.square().mean()
        loss = loss + args.aniso_reg * (axis_log - axis_log.mean()).square().mean()
        loss = loss + args.color_reg * F.mse_loss(gaussian._features_dc, base_dc)
        loss = loss + args.opacity_reg * opacity_delta.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        last_loss = float(loss.detach().cpu())
        if last_loss < best_loss:
            best_loss = last_loss
            best_state = state_before_step
        opt.step()
        if step_idx == 0 or (step_idx + 1) % args.log_every == 0:
            print(f"{obj} step={step_idx + 1:03d}/{args.steps} loss={last_loss:.6f}", flush=True)

    with torch.no_grad():
        if args.restore_best and best_state is not None:
            for p, s in zip(params, best_state):
                p.copy_(s)
        axis_log.clamp_(np.log(args.min_axis_scale), np.log(args.max_axis_scale))
        shift.clamp_(-args.max_shift, args.max_shift)
        color_scale.clamp_(args.min_color_scale, args.max_color_scale)
        color_bias.clamp_(-args.max_color_bias, args.max_color_bias)
        opacity_delta.clamp_(-args.max_opacity_delta, args.max_opacity_delta)
        axis_scale = torch.exp(axis_log)
        gaussian._xyz = center + (base_xyz - center) * axis_scale + shift
        gaussian._features_dc = base_dc * color_scale + color_bias
        gaussian._opacity = base_opacity + opacity_delta
    gaussian.save_ply(str(dst_dir / "result.ply"))

    mesh_status = "missing_result_glb"
    mesh_path = dst_dir / "result.glb"
    if mesh_path.exists():
        mesh_status = sync_mesh(
            mesh_path,
            center.detach().cpu().numpy().reshape(3),
            axis_scale.detach().cpu().numpy().reshape(3),
            shift.detach().cpu().numpy().reshape(3),
        )
    return {
        "object": obj,
        "status": "ok",
        "mesh_status": mesh_status,
        "views": ",".join(str(v) for v, *_ in train_data),
        "steps": str(args.steps),
        "final_loss": f"{last_loss:.8f}",
        "best_loss": f"{best_loss:.8f}",
        "sx": f"{float(axis_scale[0, 0].detach().cpu()):.8f}",
        "sy": f"{float(axis_scale[0, 1].detach().cpu()):.8f}",
        "sz": f"{float(axis_scale[0, 2].detach().cpu()):.8f}",
        "shift_x": f"{float(shift[0, 0].detach().cpu()):.8f}",
        "shift_y": f"{float(shift[0, 1].detach().cpu()):.8f}",
        "shift_z": f"{float(shift[0, 2].detach().cpu()):.8f}",
        "opacity_delta": f"{float(opacity_delta.detach().cpu().mean()):.8f}",
        "output": str(dst_dir / "result.ply"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--gso_root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--objects_file", required=True, type=Path)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--train_views", type=parse_views, default=parse_views("0-4"))
    parser.add_argument("--render_dir", default="render_mvs_25")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--nvs_transform", default="perm012_signpnn")
    parser.add_argument("--nvs_zflip", action="store_true")
    parser.add_argument("--nvs_focal", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--bg_weight", type=float, default=0.15)
    parser.add_argument("--edge_weight", type=float, default=0.10)
    parser.add_argument("--ssim_weight", type=float, default=1.0)
    parser.add_argument("--axis_reg", type=float, default=80.0)
    parser.add_argument("--shift_reg", type=float, default=80.0)
    parser.add_argument("--aniso_reg", type=float, default=40.0)
    parser.add_argument("--color_reg", type=float, default=0.02)
    parser.add_argument("--opacity_reg", type=float, default=0.10)
    parser.add_argument("--min_axis_scale", type=float, default=0.98)
    parser.add_argument("--max_axis_scale", type=float, default=1.02)
    parser.add_argument("--max_shift", type=float, default=0.008)
    parser.add_argument("--min_color_scale", type=float, default=0.90)
    parser.add_argument("--max_color_scale", type=float, default=1.10)
    parser.add_argument("--max_color_bias", type=float, default=0.05)
    parser.add_argument("--max_opacity_delta", type=float, default=0.08)
    parser.add_argument("--restore_best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    transform = parse_axis_transform(args.nvs_transform)
    renderer = make_renderer(args.repo, args.resolution)
    rows = [optimize_one(args, obj, renderer, transform) for obj in read_objects(args.objects_file)]
    keys = sorted({k for row in rows for k in row})
    with (args.output_root / "axis_refine_meshsync_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output_root / "axis_refine_meshsync_summary.csv")


if __name__ == "__main__":
    main()
