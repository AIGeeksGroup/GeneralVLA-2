#!/usr/bin/env python3
"""One-command launcher for the documented MV-SAM3D improvement branch.

This wrapper does not bundle large checkpoints or GSO data. It expects the user
to provide those paths in a YAML config file and then runs:

1. source A soft visual hull refinement
2. source B no-VGGT axis refinement
3. same-index geometry-only blend
4. optional four-metric evaluation

The script is intentionally strict about missing inputs so that a colleague can
see exactly which external dependency is absent.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def optional_path(raw_value: object) -> Path | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_dir = Path(__file__).resolve().parents[1]
    paths = {k: optional_path(v) for k, v in cfg["paths"].items() if optional_path(v) is not None}
    runtime = cfg["runtime"]
    pipeline = cfg["pipeline"]

    require(paths["sam3d_objects_repo"], "sam3d_objects_repo")
    require(paths["gso_root"], "gso_root")
    rebuild_source_a = bool(pipeline.get("rebuild_source_a", False))
    rebuild_source_b = bool(pipeline.get("rebuild_source_b", False))

    source_a_root = paths.get("source_a_root")
    source_b_root = paths.get("source_b_root")
    source_a_base_root = paths.get("source_a_base_root")
    source_b_base_root = paths.get("source_b_base_root")

    if rebuild_source_a:
        require(source_a_base_root, "source_a_base_root")
    else:
        require(source_a_root, "source_a_root")

    if rebuild_source_b:
        require(source_b_base_root, "source_b_base_root")
    else:
        require(source_b_root, "source_b_root")

    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    objects_file = project_dir / "configs" / "gso30_objects.txt"
    soft_a = output_root.parent / (output_root.name + "_source_a_softvh")
    source_b = output_root.parent / (output_root.name + "_source_b_axisrefine")

    if rebuild_source_a:
        run(
            [
                sys.executable,
                str(project_dir / "scripts" / "soft_visual_hull_refine.py"),
                "--pred_root",
                str(source_a_base_root),
                "--output_root",
                str(soft_a),
                "--gso_root",
                str(paths["gso_root"]),
                "--objects_file",
                str(objects_file),
                "--views",
                str(runtime["views"]),
                "--input_views",
                runtime["input_views"],
                "--support_threshold",
                str(pipeline["source_a"]["support_threshold"]),
                "--support_softness",
                str(pipeline["source_a"]["support_softness"]),
                "--max_shrink",
                str(pipeline["source_a"]["max_shrink"]),
                "--max_opacity_drop",
                str(pipeline["source_a"]["max_opacity_drop"]),
                "--nvs_zflip",
            ],
            cwd=project_dir,
        )
        source_a_final = soft_a
    else:
        source_a_final = source_a_root

    if rebuild_source_b:
        run(
            [
                sys.executable,
                str(project_dir / "scripts" / "optimize_gaussian_axis_refine_meshsync.py"),
                "--pred_root",
                str(source_b_base_root),
                "--output_root",
                str(source_b),
                "--gso_root",
                str(paths["gso_root"]),
                "--repo",
                str(paths["sam3d_objects_repo"]),
                "--objects_file",
                str(objects_file),
                "--views",
                str(runtime["views"]),
                "--train_views",
                runtime["input_views"],
                "--steps",
                str(pipeline["source_b"]["steps"]),
                "--min_axis_scale",
                str(pipeline["source_b"]["min_axis_scale"]),
                "--max_axis_scale",
                str(pipeline["source_b"]["max_axis_scale"]),
                "--max_shift",
                str(pipeline["source_b"]["max_shift"]),
                "--nvs_zflip",
            ],
            cwd=project_dir,
        )
        source_b_final = source_b
    else:
        source_b_final = source_b_root

    run(
        [
            sys.executable,
            str(project_dir / "scripts" / "blend_sameindex_geometry.py"),
            "--source_a",
            str(source_a_final),
            "--source_b",
            str(source_b_final),
            "--output_root",
            str(output_root),
            "--objects_file",
            str(objects_file),
            "--views",
            str(runtime["views"]),
            "--alpha",
            str(pipeline["blend"]["alpha"]),
        ],
        cwd=project_dir,
    )

    if not args.skip_eval:
        run(
            [
                sys.executable,
                str(project_dir / "scripts" / "evaluate_gso30_four_metrics.py"),
                "--pred_root",
                str(output_root),
                "--gso_root",
                str(paths["gso_root"]),
                "--objects_file",
                str(objects_file),
                "--output_prefix",
                str(output_root / "results" / output_root.name),
                "--repo",
                str(paths["sam3d_objects_repo"]),
                "--views",
                str(runtime["views"]),
                "--target_views",
                runtime["target_views"],
                "--nvs_zflip",
            ],
            cwd=project_dir,
        )

    print(f"Done. Final outputs are under: {output_root}")


if __name__ == "__main__":
    main()
