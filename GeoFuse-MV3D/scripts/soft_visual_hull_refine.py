#!/usr/bin/env python3
"""Soft input-view visual-hull refinement for MV-SAM3D outputs.

This experiment copies an existing output tree and applies a conservative,
continuous correction using only the fixed input views.  Points or mesh vertices
that reproject poorly into all input masks are moved slightly toward the object
center and optionally get a small opacity reduction.  Nothing is hard deleted.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AxisTransform:
    name: str
    matrix: np.ndarray


def read_objects(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def parse_views(spec: str) -> list[int]:
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
    perm = tuple(int(ch) for ch in perm_part[len("perm") :])
    signs = tuple(1.0 if ch == "p" else -1.0 for ch in sign_part)
    if sorted(perm) != [0, 1, 2] or len(signs) != 3:
        raise ValueError(f"Bad axis transform: {spec}")
    matrix = np.zeros((3, 3), dtype=np.float32)
    for out_axis, in_axis in enumerate(perm):
        matrix[out_axis, in_axis] = signs[out_axis]
    return AxisTransform(spec, matrix)


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


def load_alpha_mask(path: Path, resolution: int) -> np.ndarray:
    rgba = Image.open(path).convert("RGBA")
    if rgba.size != (resolution, resolution):
        rgba = rgba.resize((resolution, resolution), Image.BILINEAR)
    return (np.asarray(rgba, dtype=np.float32)[..., 3] / 255.0).clip(0.0, 1.0)


def project_points(points: np.ndarray, extrinsic: np.ndarray, focal: float) -> tuple[np.ndarray, np.ndarray]:
    hom = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    cam = hom @ extrinsic.T
    z = cam[:, 2]
    valid = z > 1.0e-6
    uv = np.empty((len(points), 2), dtype=np.float32)
    uv[:, 0] = focal * (cam[:, 0] / np.where(valid, z, 1.0)) + 0.5
    uv[:, 1] = focal * (cam[:, 1] / np.where(valid, z, 1.0)) + 0.5
    valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
    return uv, valid


def sample_mask_bilinear(mask: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    x = uv[:, 0] * (w - 1)
    y = uv[:, 1] * (h - 1)
    inside = (x >= 0.0) & (x <= w - 1) & (y >= 0.0) & (y <= h - 1)
    out = np.zeros(len(uv), dtype=np.float32)
    if not np.any(inside):
        return out

    xi = x[inside]
    yi = y[inside]
    x0 = np.floor(xi).astype(np.int64)
    y0 = np.floor(yi).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xi - x0
    wy = yi - y0
    top = mask[y0, x0] * (1.0 - wx) + mask[y0, x1] * wx
    bot = mask[y1, x0] * (1.0 - wx) + mask[y1, x1] * wx
    out[inside] = top * (1.0 - wy) + bot * wy
    return out


def compute_support(
    points: np.ndarray,
    masks: list[np.ndarray],
    extrinsics: list[np.ndarray],
    focal: float,
    min_visible_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    support_sum = np.zeros(len(points), dtype=np.float32)
    visible_count = np.zeros(len(points), dtype=np.float32)
    for mask, extr in zip(masks, extrinsics):
        uv, valid = project_points(points, extr, focal)
        sampled = sample_mask_bilinear(mask, uv)
        support_sum += sampled * valid.astype(np.float32)
        visible_count += valid.astype(np.float32)
    enough = visible_count >= float(min_visible_views)
    denom = np.maximum(visible_count, 1.0)
    support = support_sum / denom
    support[~enough] = 1.0
    return support.clip(0.0, 1.0), visible_count


def support_to_strength(
    support: np.ndarray,
    threshold: float,
    softness: float,
    max_strength: float,
) -> np.ndarray:
    softness = max(float(softness), 1.0e-6)
    raw = 1.0 / (1.0 + np.exp((support - float(threshold)) / softness))
    raw = raw * raw
    return (float(max_strength) * raw).astype(np.float32)


def copy_output_tree(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_dir / item.name)


def refine_points(points: np.ndarray, support: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0, keepdims=True)
    strength = support_to_strength(support, args.support_threshold, args.support_softness, args.max_shrink)
    refined = center + (points - center) * (1.0 - strength[:, None])
    return refined.astype(np.float32), strength


def refine_ply(path: Path, masks: list[np.ndarray], extrinsics: list[np.ndarray], args: argparse.Namespace) -> dict[str, str]:
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    support, visible = compute_support(points, masks, extrinsics, args.nvs_focal, args.min_visible_views)
    refined, strength = refine_points(points, support, args)

    new_vertex = vertex.copy()
    new_vertex["x"] = refined[:, 0]
    new_vertex["y"] = refined[:, 1]
    new_vertex["z"] = refined[:, 2]
    if args.max_opacity_drop > 0 and "opacity" in new_vertex.dtype.names:
        opacity_drop = args.max_opacity_drop * (strength / max(args.max_shrink, 1.0e-6))
        new_vertex["opacity"] = new_vertex["opacity"] - opacity_drop.astype(new_vertex["opacity"].dtype)

    elements = [PlyElement.describe(new_vertex, "vertex")]
    elements.extend(elem for elem in ply.elements if elem.name != "vertex")
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(path))
    return {
        "mean_support": f"{float(np.mean(support)):.6f}",
        "p10_support": f"{float(np.quantile(support, 0.10)):.6f}",
        "mean_visible": f"{float(np.mean(visible)):.6f}",
        "mean_strength": f"{float(np.mean(strength)):.6f}",
        "p90_strength": f"{float(np.quantile(strength, 0.90)):.6f}",
    }


def refine_mesh_geometry(mesh, masks: list[np.ndarray], extrinsics: list[np.ndarray], args: argparse.Namespace) -> int:
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        return 0
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    support, _ = compute_support(vertices, masks, extrinsics, args.nvs_focal, args.min_visible_views)
    refined, _ = refine_points(vertices, support, args)
    mesh.vertices = refined
    return len(refined)


def refine_mesh(path: Path, masks: list[np.ndarray], extrinsics: list[np.ndarray], args: argparse.Namespace) -> str:
    import trimesh

    mesh_or_scene = trimesh.load(str(path), process=False)
    if isinstance(mesh_or_scene, trimesh.Scene):
        touched = 0
        for geom in mesh_or_scene.geometry.values():
            touched += refine_mesh_geometry(geom, masks, extrinsics, args)
        mesh_or_scene.export(str(path))
        return f"scene_vertices={touched}"
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        touched = refine_mesh_geometry(mesh_or_scene, masks, extrinsics, args)
        mesh_or_scene.export(str(path))
        return f"mesh_vertices={touched}"
    return f"unsupported:{type(mesh_or_scene).__name__}"


def load_object_inputs(args: argparse.Namespace, obj: str, transform: AxisTransform) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masks = []
    extrinsics = []
    for view in args.input_views:
        stem = f"{view:03d}"
        gt_path = args.gso_root / obj / args.render_dir / "model" / f"{stem}.png"
        cam_path = args.gso_root / obj / args.render_dir / "model" / f"{stem}.npy"
        if gt_path.exists() and cam_path.exists():
            masks.append(load_alpha_mask(gt_path, args.resolution))
            extrinsics.append(make_extrinsic(np.load(cam_path).astype(np.float32), transform.matrix, args.nvs_zflip))
    return masks, extrinsics


def refine_one(args: argparse.Namespace, obj: str, transform: AxisTransform) -> dict[str, str]:
    src_dir = args.pred_root / obj / f"{args.views}views"
    dst_dir = args.output_root / obj / f"{args.views}views"
    if not (src_dir / "result.ply").exists():
        return {"object": obj, "status": "missing_result_ply"}
    copy_output_tree(src_dir, dst_dir)
    masks, extrinsics = load_object_inputs(args, obj, transform)
    if not masks:
        return {"object": obj, "status": "missing_input_masks"}
    row = {"object": obj, "status": "ok", "n_input_views": str(len(masks))}
    row.update(refine_ply(dst_dir / "result.ply", masks, extrinsics, args))
    mesh_path = dst_dir / "result.glb"
    row["mesh_status"] = refine_mesh(mesh_path, masks, extrinsics, args) if mesh_path.exists() else "missing_result_glb"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--gso_root", required=True, type=Path)
    parser.add_argument("--objects_file", required=True, type=Path)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--input_views", type=parse_views, default=parse_views("0-4"))
    parser.add_argument("--render_dir", default="render_mvs_25")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--nvs_transform", default="perm012_signpnn")
    parser.add_argument("--nvs_zflip", action="store_true")
    parser.add_argument("--nvs_focal", type=float, default=1.0)
    parser.add_argument("--min_visible_views", type=int, default=2)
    parser.add_argument("--support_threshold", type=float, default=0.35)
    parser.add_argument("--support_softness", type=float, default=0.12)
    parser.add_argument("--max_shrink", type=float, default=0.015)
    parser.add_argument("--max_opacity_drop", type=float, default=0.04)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    transform = parse_axis_transform(args.nvs_transform)
    rows = [refine_one(args, obj, transform) for obj in read_objects(args.objects_file)]
    keys = sorted({key for row in rows for key in row})
    report = args.output_root / "soft_visual_hull_summary.csv"
    with report.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(report)


if __name__ == "__main__":
    main()
