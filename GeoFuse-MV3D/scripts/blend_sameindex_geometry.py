#!/usr/bin/env python3
"""Geometry-only same-index blend for MV-SAM3D outputs.

Given two output trees with the usual layout

    <root>/<object>/<views>views/result.ply
    <root>/<object>/<views>views/result.glb

this script copies source A to the destination and then blends geometry from
source B only where the topology/indexing is compatible.  It never changes
colors, opacity, scale, rotation, or SH features.  If either PLY vertex count or
mesh vertex count is incompatible, the corresponding file stays as source A.

The intended use is conservative: source A is the trusted main branch, source B
is a small geometry correction branch.  ``--alpha 0.5`` means halfway from A to B.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


def read_objects(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def copy_tree_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_dir / item.name)


def blend_ply_xyz(path_a: Path, path_b: Path, output_path: Path, alpha: float) -> str:
    from plyfile import PlyData, PlyElement

    if not path_a.exists() or not path_b.exists():
        return "copied_a_missing_ply"
    ply_a = PlyData.read(str(path_a))
    ply_b = PlyData.read(str(path_b))
    va = ply_a["vertex"].data
    vb = ply_b["vertex"].data
    required = {"x", "y", "z"}
    if not required.issubset(set(va.dtype.names or ())) or not required.issubset(set(vb.dtype.names or ())) :
        return "copied_a_missing_xyz_fields"
    if len(va) != len(vb):
        return "copied_a_incompatible_vertex_count"

    out_vertex = va.copy()
    for key in ("x", "y", "z"):
        out_vertex[key] = ((1.0 - alpha) * va[key].astype(np.float32) + alpha * vb[key].astype(np.float32)).astype(out_vertex[key].dtype)
    elements = [PlyElement.describe(out_vertex, "vertex")]
    elements.extend(elem for elem in ply_a.elements if elem.name != "vertex")
    PlyData(elements, text=ply_a.text, byte_order=ply_a.byte_order).write(str(output_path))
    return "ok"


def blend_mesh_vertices(path_a: Path, path_b: Path, output_path: Path, alpha: float) -> str:
    import trimesh

    if not path_a.exists() or not path_b.exists():
        return "copied_a_missing_mesh"
    mesh_a = trimesh.load(str(path_a), process=False)
    mesh_b = trimesh.load(str(path_b), process=False)

    if isinstance(mesh_a, trimesh.Scene) or isinstance(mesh_b, trimesh.Scene):
        return "copied_a_scene_not_supported"
    if not isinstance(mesh_a, trimesh.Trimesh) or not isinstance(mesh_b, trimesh.Trimesh):
        return "copied_a_unsupported_mesh_type"
    if len(mesh_a.vertices) != len(mesh_b.vertices):
        return "copied_a_incompatible_mesh_vertex_count"

    va = np.asarray(mesh_a.vertices, dtype=np.float32)
    vb = np.asarray(mesh_b.vertices, dtype=np.float32)
    mesh_a.vertices = (1.0 - alpha) * va + alpha * vb
    mesh_a.export(str(output_path))
    return "ok"


def blend_one(args: argparse.Namespace, obj: str) -> dict[str, str]:
    src_a = args.source_a / obj / f"{args.views}views"
    src_b = args.source_b / obj / f"{args.views}views"
    dst = args.output_root / obj / f"{args.views}views"
    if not src_a.exists():
        return {"object": obj, "status": "missing_source_a", "ply_status": "", "mesh_status": ""}

    copy_tree_files(src_a, dst)
    ply_status = blend_ply_xyz(src_a / "result.ply", src_b / "result.ply", dst / "result.ply", args.alpha)
    mesh_status = blend_mesh_vertices(src_a / "result.glb", src_b / "result.glb", dst / "result.glb", args.alpha)
    return {
        "object": obj,
        "status": "ok",
        "alpha": f"{args.alpha:.6f}",
        "source_a": str(src_a),
        "source_b": str(src_b),
        "output": str(dst),
        "ply_status": ply_status,
        "mesh_status": mesh_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_a", required=True, type=Path, help="Trusted main output tree.")
    parser.add_argument("--source_b", required=True, type=Path, help="Geometry correction output tree.")
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--objects_file", required=True, type=Path)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1].")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [blend_one(args, obj) for obj in read_objects(args.objects_file)]
    keys = ["object", "status", "alpha", "ply_status", "mesh_status", "source_a", "source_b", "output"]
    report = args.output_root / "geometry_only_blend_summary.csv"
    with report.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(report)


if __name__ == "__main__":
    main()
