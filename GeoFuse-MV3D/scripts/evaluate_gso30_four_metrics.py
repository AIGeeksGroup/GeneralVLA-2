#!/usr/bin/env python3
"""Unified GSO-30 four-metric evaluator for MV-SAM3D outputs.

This script is intentionally conservative about protocol claims:

* CD mirrors the public EscherNet 3D evaluator style: load GT
  ``meshes/model.obj``, load a predicted mesh-like output, normalize each mesh
  independently to a unit bounding box, then compute non-squared symmetric
  nearest-neighbor Chamfer distance on vertices.
* NVS uses the public EscherNet 2D metric formulas and GT convention:
  ``render_mvs_25/model/010..024.png``, white-background alpha compositing,
  256x256 images, OpenCV PSNR, skimage SSIM, and AlexNet LPIPS.
* The missing public piece is the MV-SAM3D paper's renderer bridge from
  ``result.ply``/``result.glb`` to EscherNet's expected prediction mosaic.  For
  NVS, this script uses MV-SAM3D's own ``Gaussian.load_ply`` and
  ``GaussianRenderer`` with one fixed, reportable bridge.  It should be treated
  as a reproducible diagnostic bridge unless an official MV-SAM3D benchmark
  renderer is obtained.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity as calculate_ssim


@dataclass(frozen=True)
class AxisTransform:
    name: str
    matrix: np.ndarray


def read_objects(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_view_spec(spec: str) -> list[int]:
    views: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = [int(x) for x in part.split("-", 1)]
            views.extend(range(first, last + 1))
        else:
            views.append(int(part))
    return views


def parse_axis_transform(spec: str) -> AxisTransform:
    if spec == "identity":
        return AxisTransform(spec, np.eye(3, dtype=np.float32))
    if not spec.startswith("perm") or "_sign" not in spec:
        raise ValueError(f"Unsupported axis transform: {spec}")
    perm_part, sign_part = spec.split("_sign", 1)
    perm = tuple(int(ch) for ch in perm_part[len("perm"):])
    signs = tuple(1.0 if ch == "p" else -1.0 for ch in sign_part)
    if sorted(perm) != [0, 1, 2] or len(signs) != 3:
        raise ValueError(f"Bad axis transform: {spec}")
    matrix = np.zeros((3, 3), dtype=np.float32)
    for out_axis, in_axis in enumerate(perm):
        matrix[out_axis, in_axis] = signs[out_axis]
    return AxisTransform(spec, matrix)


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    max_pt = np.max(vertices, axis=0)
    min_pt = np.min(vertices, axis=0)
    scale = 1.0 / np.max(max_pt - min_pt)
    vertices = vertices * scale
    max_pt = np.max(vertices, axis=0)
    min_pt = np.min(vertices, axis=0)
    center = (max_pt + min_pt) / 2.0
    return vertices - center[None, :]


def load_vertices(path: Path) -> np.ndarray:
    mesh_or_scene = trimesh.load(str(path), process=False)
    meshes = []
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = [
            geom
            for geom in mesh_or_scene.geometry.values()
            if hasattr(geom, "vertices") and len(geom.vertices)
        ]
    elif hasattr(mesh_or_scene, "vertices") and len(mesh_or_scene.vertices):
        meshes = [mesh_or_scene]
    if not meshes:
        raise ValueError(f"No mesh vertices found in {path}")
    return np.concatenate(
        [np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes],
        axis=0,
    )


def chamfer(gt_points: np.ndarray, rec_points: np.ndarray) -> float:
    rec_tree = cKDTree(rec_points)
    gt_to_rec = rec_tree.query(gt_points)[0].mean()
    gt_tree = cKDTree(gt_points)
    rec_to_gt = gt_tree.query(rec_points)[0].mean()
    return float((gt_to_rec + rec_to_gt) / 2.0)


def find_pred_mesh(pred_root: Path, obj: str, views: int) -> Path | None:
    candidates = [
        pred_root / obj / f"{views}views" / "mesh.ply",
        pred_root / obj / f"{views}views" / "result.glb",
        pred_root / obj / f"{views}views" / "result.ply",
    ]
    return next((path for path in candidates if path.exists()), None)


def find_pred_ply(pred_root: Path, obj: str, views: int) -> Path | None:
    path = pred_root / obj / f"{views}views" / "result.ply"
    return path if path.exists() else None


def evaluate_cd_one(
    pred_root: Path,
    gso_root: Path,
    obj: str,
    views: int,
    transform: AxisTransform,
) -> dict[str, str]:
    gt_path = gso_root / obj / "meshes" / "model.obj"
    pred_path = find_pred_mesh(pred_root, obj, views)
    row = {
        "object": obj,
        "cd": "",
        "cd_x1e3": "",
        "cd_transform": transform.name,
        "cd_pred_path": str(pred_path or ""),
        "cd_gt_path": str(gt_path),
        "cd_status": "",
    }
    if pred_path is None:
        row["cd_status"] = "missing_prediction"
        return row
    if not gt_path.exists():
        row["cd_status"] = "missing_gt"
        return row
    try:
        gt_vertices = normalize_vertices(load_vertices(gt_path))
        pred_vertices = load_vertices(pred_path) @ transform.matrix.T
        pred_vertices = normalize_vertices(pred_vertices)
        cd = chamfer(gt_vertices, pred_vertices)
        row.update(
            {
                "cd": f"{cd:.8f}",
                "cd_x1e3": f"{cd * 1000.0:.4f}",
                "cd_status": "ok",
            }
        )
    except Exception as exc:  # noqa: BLE001 - status CSV should capture failures.
        row["cd_status"] = f"error:{type(exc).__name__}:{exc}"
    return row


def load_gt_uint8(path: Path, resolution: int) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    img = Image.fromarray(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0)).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def to4(mat3x4: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :] = mat3x4[:3, :].astype(np.float32)
    return out


def make_extrinsic(gso_w2c: np.ndarray, object_to_gso: np.ndarray, zflip: bool) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = object_to_gso
    extrinsic = to4(gso_w2c) @ transform
    if zflip:
        flip = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
        extrinsic = flip @ extrinsic
    return extrinsic.astype(np.float32)


def calc_2d_metrics(pred_np: np.ndarray, gt_np: np.ndarray, lpips_model: lpips.LPIPS) -> dict[str, float]:
    pred_image = torch.from_numpy(pred_np.copy()).unsqueeze(0).permute(0, 3, 1, 2)
    gt_image = torch.from_numpy(gt_np.copy()).unsqueeze(0).permute(0, 3, 1, 2)
    pred_image = pred_image.float() / 127.5 - 1
    gt_image = gt_image.float() / 127.5 - 1
    loss = F.mse_loss(pred_image[0], gt_image[0].cpu()).item()
    with torch.no_grad():
        lp = lpips_model(pred_image[0].cuda(), gt_image[0].cuda()).item()
    return {
        "loss": float(loss),
        "lpips": float(lp),
        "ssim": float(calculate_ssim(pred_np, gt_np, channel_axis=2)),
        "psnr": float(cv2.PSNR(gt_np, pred_np)),
    }


def find_pred_mosaic(pred_root: Path, obj: str) -> Path | None:
    candidates = [
        pred_root / obj / "0.png",
        pred_root / obj / obj / "0.png",
    ]
    return next((path for path in candidates if path.exists()), None)


def load_mosaic_tile_uint8(path: Path, view: int, resolution: int) -> np.ndarray:
    mosaic = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    y0 = 0
    x0 = view * resolution
    y1 = resolution
    x1 = x0 + resolution
    if mosaic.shape[0] < y1 or mosaic.shape[1] < x1:
        raise ValueError(
            f"Prediction mosaic {path} is too small for view {view}: "
            f"shape={mosaic.shape}, required_width>={x1}"
        )
    return mosaic[y0:y1, x0:x1, :]


def build_official_mosaic(view_tiles: list[tuple[int, np.ndarray]], resolution: int) -> np.ndarray:
    if not view_tiles:
        raise ValueError("Cannot build a mosaic without rendered tiles.")
    total_views = max(view for view, _tile in view_tiles) + 1
    mosaic = np.full((resolution, total_views * resolution, 3), 255, dtype=np.uint8)
    for view, tile in view_tiles:
        if tile.shape[:2] != (resolution, resolution):
            tile = np.asarray(Image.fromarray(tile).resize((resolution, resolution), Image.BILINEAR))
        mosaic[:, view * resolution : (view + 1) * resolution, :] = tile[:, :, :3]
    return mosaic


def evaluate_nvs_official_mosaic_one(
    *,
    pred_root: Path,
    gso_root: Path,
    obj: str,
    target_views: list[int],
    render_dir: str,
    resolution: int,
    lpips_model: lpips.LPIPS,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    mosaic_path = find_pred_mosaic(pred_root, obj)
    detail_rows: list[dict[str, str]] = []
    summary = {
        "object": obj,
        "nvs_n": "0",
        "mean_loss": "",
        "mean_psnr": "",
        "mean_ssim": "",
        "mean_lpips": "",
        "nvs_status": "",
        "nvs_pred_path": str(mosaic_path or ""),
    }
    if mosaic_path is None:
        summary["nvs_status"] = "missing_mosaic"
        return detail_rows, summary

    for view in target_views:
        stem = f"{view:03d}"
        gt_path = gso_root / obj / render_dir / "model" / f"{stem}.png"
        row = {
            "object": obj,
            "view": str(view),
            "loss": "",
            "psnr": "",
            "ssim": "",
            "lpips": "",
            "nvs_transform": "none",
            "nvs_zflip": "False",
            "nvs_focal": "",
            "nvs_renderer_protocol": "eschernet_official_prediction_mosaic_0png",
            "status": "",
        }
        if not gt_path.exists():
            row["status"] = "missing_gt"
            detail_rows.append(row)
            continue
        try:
            gt = load_gt_uint8(gt_path, resolution)
            pred = load_mosaic_tile_uint8(mosaic_path, view, resolution)
            metrics = calc_2d_metrics(pred, gt, lpips_model)
            row.update(
                {
                    "loss": f"{metrics['loss']:.8f}",
                    "psnr": f"{metrics['psnr']:.6f}",
                    "ssim": f"{metrics['ssim']:.6f}",
                    "lpips": f"{metrics['lpips']:.6f}",
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"error:{type(exc).__name__}:{exc}"
        detail_rows.append(row)

    ok = [row for row in detail_rows if row["status"] == "ok"]
    if ok:
        summary.update(
            {
                "nvs_n": str(len(ok)),
                "mean_loss": f"{np.mean([float(row['loss']) for row in ok]):.8f}",
                "mean_psnr": f"{np.mean([float(row['psnr']) for row in ok]):.6f}",
                "mean_ssim": f"{np.mean([float(row['ssim']) for row in ok]):.6f}",
                "mean_lpips": f"{np.mean([float(row['lpips']) for row in ok]):.6f}",
                "nvs_status": "ok",
            }
        )
    else:
        summary["nvs_status"] = "no_valid_views"
    return detail_rows, summary


def load_gaussian(repo: Path, ply_path: Path):
    sys.path.insert(0, str(repo))
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian

    gaussian = Gaussian(aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], sh_degree=0, device="cuda")
    gaussian.load_ply(str(ply_path))
    gaussian.max_sh_degree = gaussian.sh_degree
    return gaussian


def render_one(renderer, gaussian, extrinsic: np.ndarray, focal: float) -> np.ndarray:
    extr = torch.tensor(extrinsic, dtype=torch.float32, device="cuda")
    intr = torch.tensor(
        [[float(focal), 0.0, 0.5], [0.0, float(focal), 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    with torch.no_grad():
        pred_t = renderer.render(gaussian, extr, intr).color.clamp(0.0, 1.0)
    return np.uint8(np.clip(pred_t.permute(1, 2, 0).detach().cpu().numpy() * 255.0, 0, 255))


def make_renderer(repo: Path, resolution: int):
    sys.path.insert(0, str(repo))
    from sam3d_objects.model.backbone.tdfy_dit.renderers import GaussianRenderer

    renderer = GaussianRenderer(
        {
            "resolution": resolution,
            "near": 0.01,
            "far": 100.0,
            "ssaa": 1,
            "bg_color": [1.0, 1.0, 1.0],
            "backend": "gsplat",
        }
    )
    renderer.pipe.convert_SHs_python = True
    return renderer


def evaluate_nvs_one(
    *,
    pred_root: Path,
    gso_root: Path,
    repo: Path,
    obj: str,
    views: int,
    target_views: list[int],
    transform: AxisTransform,
    zflip: bool,
    focal: float,
    render_dir: str,
    resolution: int,
    renderer,
    lpips_model: lpips.LPIPS,
    save_mosaic_dir: Path | None,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    ply_path = find_pred_ply(pred_root, obj, views)
    detail_rows: list[dict[str, str]] = []
    summary = {
        "object": obj,
        "nvs_n": "0",
        "mean_loss": "",
        "mean_psnr": "",
        "mean_ssim": "",
        "mean_lpips": "",
        "nvs_status": "",
        "nvs_pred_path": str(ply_path or ""),
    }
    if ply_path is None:
        summary["nvs_status"] = "missing_ply"
        return detail_rows, summary
    try:
        gaussian = load_gaussian(repo, ply_path)
    except Exception as exc:  # noqa: BLE001
        summary["nvs_status"] = f"error_load_gaussian:{type(exc).__name__}:{exc}"
        return detail_rows, summary

    rendered_tiles: list[tuple[int, np.ndarray]] = []
    for view in target_views:
        stem = f"{view:03d}"
        gt_path = gso_root / obj / render_dir / "model" / f"{stem}.png"
        cam_path = gso_root / obj / render_dir / "model" / f"{stem}.npy"
        base_row = {
            "object": obj,
            "view": str(view),
            "loss": "",
            "psnr": "",
            "ssim": "",
            "lpips": "",
            "nvs_transform": transform.name,
            "nvs_zflip": str(zflip),
            "nvs_focal": f"{focal:.6f}",
            "nvs_renderer_protocol": "self_built_fixed_bridge_mvsam3d_GaussianRenderer_not_official_benchmark_renderer",
            "status": "",
        }
        if not gt_path.exists() or not cam_path.exists():
            base_row["status"] = "missing_gt_or_camera"
            detail_rows.append(base_row)
            continue
        try:
            gt = load_gt_uint8(gt_path, resolution)
            cam = np.load(cam_path).astype(np.float32)
            pred = render_one(
                renderer,
                gaussian,
                make_extrinsic(cam, transform.matrix, zflip),
                focal,
            )
            rendered_tiles.append((view, pred))
            metrics = calc_2d_metrics(pred, gt, lpips_model)
            base_row.update(
                {
                    "loss": f"{metrics['loss']:.8f}",
                    "psnr": f"{metrics['psnr']:.6f}",
                    "ssim": f"{metrics['ssim']:.6f}",
                    "lpips": f"{metrics['lpips']:.6f}",
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            base_row["status"] = f"error:{type(exc).__name__}:{exc}"
        detail_rows.append(base_row)
    ok = [row for row in detail_rows if row["status"] == "ok"]
    if ok:
        summary.update(
            {
                "nvs_n": str(len(ok)),
                "mean_loss": f"{np.mean([float(row['loss']) for row in ok]):.8f}",
                "mean_psnr": f"{np.mean([float(row['psnr']) for row in ok]):.6f}",
                "mean_ssim": f"{np.mean([float(row['ssim']) for row in ok]):.6f}",
                "mean_lpips": f"{np.mean([float(row['lpips']) for row in ok]):.6f}",
                "nvs_status": "ok",
            }
        )
    else:
        summary["nvs_status"] = "no_valid_views"
    if save_mosaic_dir is not None and rendered_tiles:
        mosaic_obj_dir = save_mosaic_dir / obj
        mosaic_obj_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(build_official_mosaic(rendered_tiles, resolution)).save(mosaic_obj_dir / "0.png")
    del gaussian
    torch.cuda.empty_cache()
    return detail_rows, summary


def mean_field(rows: list[dict[str, str]], key: str, status_key: str) -> str:
    values = [
        float(row[key])
        for row in rows
        if row.get(status_key) == "ok" and row.get(key) not in ("", None)
    ]
    return f"{np.mean(values):.8f}" if values else ""


def percent_change(metric: str, baseline: str, candidate: str) -> str:
    if baseline == "" or candidate == "":
        return ""
    base = float(baseline)
    cand = float(candidate)
    if math.isclose(base, 0.0):
        return ""
    if metric in {"cd", "lpips"}:
        return f"{(base - cand) / base * 100.0:.4f}"
    return f"{(cand - base) / base * 100.0:.4f}"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", required=True, type=Path)
    parser.add_argument("--gso_root", required=True, type=Path)
    parser.add_argument("--objects_file", required=True, type=Path)
    parser.add_argument("--output_prefix", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path, help="MV-SAM3D repo copy to import renderer/load_ply from.")
    parser.add_argument("--views", default=5, type=int)
    parser.add_argument("--target_views", default="10-24")
    parser.add_argument("--render_dir", default="render_mvs_25")
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--cd_transform", default="perm021_signpnp")
    parser.add_argument("--nvs_transform", default="perm012_signpnn")
    parser.add_argument("--nvs_zflip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nvs_focal", default=1.0, type=float)
    parser.add_argument("--limit_objects", default=0, type=int)
    parser.add_argument("--skip_nvs", action="store_true")
    parser.add_argument("--skip_cd", action="store_true")
    parser.add_argument(
        "--nvs_source",
        choices=["gaussian_bridge", "official_mosaic"],
        default="gaussian_bridge",
        help="official_mosaic reads EscherNet-style pred_root/obj/0.png when available.",
    )
    parser.add_argument("--save_mosaics_dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.skip_nvs and not torch.cuda.is_available():
        raise RuntimeError("NVS evaluation requires CUDA for MV-SAM3D GaussianRenderer and LPIPS.")

    objects = read_objects(args.objects_file)
    if args.limit_objects > 0:
        objects = objects[: args.limit_objects]
    target_views = parse_view_spec(args.target_views)
    cd_transform = parse_axis_transform(args.cd_transform)
    nvs_transform = parse_axis_transform(args.nvs_transform)

    protocol = {
        "cd_protocol": "eschernet_official_style_mesh_vertices_unit_bbox_nonsquared_symmetric_cd",
        "cd_transform": cd_transform.name,
        "nvs_metric_protocol": "eschernet_eval_2D_NVS_formulas_gt_010_024_white_bg_256",
        "nvs_renderer_protocol": "self_built_fixed_bridge_mvsam3d_GaussianRenderer_not_official_benchmark_renderer",
        "nvs_transform": nvs_transform.name,
        "nvs_zflip": str(args.nvs_zflip),
        "nvs_focal": f"{args.nvs_focal:.6f}",
        "input_views": "0,1,2,3,4",
        "target_views": ",".join(str(x) for x in target_views),
        "pred_root": str(args.pred_root),
        "gso_root": str(args.gso_root),
        "repo": str(args.repo),
    }

    renderer = None
    lpips_model = None
    if not args.skip_nvs:
        sys.path.insert(0, str(args.repo))
        torch.set_grad_enabled(False)
        if args.nvs_source == "gaussian_bridge":
            renderer = make_renderer(args.repo, args.resolution)
        lpips_model = lpips.LPIPS(net="alex", version="0.1").cuda().eval()
        protocol["nvs_renderer_protocol"] = (
            "eschernet_official_prediction_mosaic_0png"
            if args.nvs_source == "official_mosaic"
            else "self_built_fixed_bridge_mvsam3d_GaussianRenderer_not_official_benchmark_renderer"
        )

    cd_rows: list[dict[str, str]] = []
    nvs_detail_rows: list[dict[str, str]] = []
    object_rows: list[dict[str, str]] = []

    for obj in objects:
        cd_row = {
            "object": obj,
            "cd": "",
            "cd_x1e3": "",
            "cd_transform": cd_transform.name,
            "cd_pred_path": "",
            "cd_gt_path": "",
            "cd_status": "skipped",
        }
        if not args.skip_cd:
            cd_row = evaluate_cd_one(args.pred_root, args.gso_root, obj, args.views, cd_transform)
            cd_rows.append(cd_row)

        nvs_summary = {
            "object": obj,
            "nvs_n": "0",
            "mean_loss": "",
            "mean_psnr": "",
            "mean_ssim": "",
            "mean_lpips": "",
            "nvs_status": "skipped",
            "nvs_pred_path": "",
        }
        if not args.skip_nvs:
            if args.nvs_source == "official_mosaic":
                detail, nvs_summary = evaluate_nvs_official_mosaic_one(
                    pred_root=args.pred_root,
                    gso_root=args.gso_root,
                    obj=obj,
                    target_views=target_views,
                    render_dir=args.render_dir,
                    resolution=args.resolution,
                    lpips_model=lpips_model,
                )
            else:
                detail, nvs_summary = evaluate_nvs_one(
                    pred_root=args.pred_root,
                    gso_root=args.gso_root,
                    repo=args.repo,
                    obj=obj,
                    views=args.views,
                    target_views=target_views,
                    transform=nvs_transform,
                    zflip=args.nvs_zflip,
                    focal=args.nvs_focal,
                    render_dir=args.render_dir,
                    resolution=args.resolution,
                    renderer=renderer,
                    lpips_model=lpips_model,
                    save_mosaic_dir=args.save_mosaics_dir,
                )
            nvs_detail_rows.extend(detail)

        object_row = {
            "object": obj,
            "cd": cd_row["cd"],
            "cd_x1e3": cd_row["cd_x1e3"],
            "mean_psnr": nvs_summary["mean_psnr"],
            "mean_ssim": nvs_summary["mean_ssim"],
            "mean_lpips": nvs_summary["mean_lpips"],
            "nvs_n": nvs_summary["nvs_n"],
            "cd_status": cd_row["cd_status"],
            "nvs_status": nvs_summary["nvs_status"],
            "cd_pred_path": cd_row["cd_pred_path"],
            "nvs_pred_path": nvs_summary["nvs_pred_path"],
        }
        object_rows.append(object_row)
        print(
            obj,
            "CDx1e3=" + (object_row["cd_x1e3"] or "NA"),
            "PSNR=" + (object_row["mean_psnr"] or "NA"),
            "SSIM=" + (object_row["mean_ssim"] or "NA"),
            "LPIPS=" + (object_row["mean_lpips"] or "NA"),
            flush=True,
        )

    summary = dict(protocol)
    summary.update(
        {
            "count_objects": str(len(object_rows)),
            "count_cd_ok": str(sum(1 for row in object_rows if row["cd_status"] == "ok")),
            "count_nvs_ok": str(sum(1 for row in object_rows if row["nvs_status"] == "ok")),
            "mean_cd": mean_field(object_rows, "cd", "cd_status"),
            "mean_cd_x1e3": (
                f"{float(mean_field(object_rows, 'cd', 'cd_status')) * 1000.0:.4f}"
                if mean_field(object_rows, "cd", "cd_status")
                else ""
            ),
            "mean_psnr": mean_field(object_rows, "mean_psnr", "nvs_status"),
            "mean_ssim": mean_field(object_rows, "mean_ssim", "nvs_status"),
            "mean_lpips": mean_field(object_rows, "mean_lpips", "nvs_status"),
        }
    )

    prefix = args.output_prefix
    write_csv(
        prefix.with_name(prefix.name + "_objects.csv"),
        object_rows,
        [
            "object",
            "cd",
            "cd_x1e3",
            "mean_psnr",
            "mean_ssim",
            "mean_lpips",
            "nvs_n",
            "cd_status",
            "nvs_status",
            "cd_pred_path",
            "nvs_pred_path",
        ],
    )
    if cd_rows:
        write_csv(
            prefix.with_name(prefix.name + "_cd_detail.csv"),
            cd_rows,
            ["object", "cd", "cd_x1e3", "cd_transform", "cd_pred_path", "cd_gt_path", "cd_status"],
        )
    if nvs_detail_rows:
        write_csv(
            prefix.with_name(prefix.name + "_nvs_views.csv"),
            nvs_detail_rows,
            [
                "object",
                "view",
                "loss",
                "psnr",
                "ssim",
                "lpips",
                "nvs_transform",
                "nvs_zflip",
                "nvs_focal",
                "nvs_renderer_protocol",
                "status",
            ],
        )
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), [summary], list(summary.keys()))
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
