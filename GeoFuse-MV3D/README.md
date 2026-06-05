# main_provider_appaff_softvh_alpha05_sameindex_geomblend

This repository packages the experiment branch named
`main_provider_appaff_softvh_alpha05_sameindex_geomblend`.

The branch is a conservative MV-SAM3D post-processing pipeline for GSO-30
single-object multi-view generation. It keeps the same input images and masks as
the baseline: views `0,1,2,3,4` for every object.

## What This Version Does

The final version has three parts:

1. **Main provider/appaff branch**
   Uses the stronger `main_provider` / `appaff` branch as the trusted base. This
   branch is VGGT-derived indirectly.

2. **softVH refinement**
   `softVH` means soft visual hull. For each Gaussian point or mesh vertex, the
   script projects it into the five input masks and computes how well that 3D
   point is supported by the masks. Low-support points are moved slightly toward
   the object center. Points are not deleted.

3. **same-index geometry blend**
   The final stage blends the trusted softVH result with a no-VGGT axis2 geometry
   compensation branch. It only blends `x,y,z` geometry when the two outputs have
   compatible vertex counts. If topology is incompatible, it copies the trusted
   source A result unchanged.

So this version is **VGGT-derived**, but the final `sameindex_geomblend` stage is
only a geometry fusion post-process.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── gso30_objects.txt
│   ├── method.yaml
│   └── paths.example.yaml
├── scripts/
│   ├── soft_visual_hull_refine.py
│   ├── optimize_gaussian_axis_refine_meshsync.py
│   ├── blend_sameindex_geometry.py
│   ├── evaluate_gso30_four_metrics.py
│   └── run_full_pipeline.py
├── tests/
│   └── test_soft_visual_hull.py
├── docs/
│   ├── external_assets.md
│   ├── method_note.md
│   └── reproduction.md
└── examples/
    └── ply/
```

## Quick Start

Install the lightweight dependencies in the same environment as the MV-SAM3D /
SAM3D renderer:

```bash
pip install -r requirements.txt
```

Copy the path template and edit all external paths:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

Large assets are not included. See `docs/external_assets.md` for download links
and expected layouts.

The shortest path is to prepare two existing intermediate output trees:

```text
source_a_root = main_provider_appaff_softvh_th045_so012_s008
source_b_root = novggt_axisgrid_s020_appaff_s80
```

Then run the packaged branch:

```bash
python scripts/run_full_pipeline.py \
  --config configs/paths.local.yaml
```

Skip evaluation when only PLY/GLB generation is needed:

```bash
python scripts/run_full_pipeline.py \
  --config configs/paths.local.yaml \
  --skip_eval
```

The final PLYs are written to:

```text
<output_root>/<object>/5views/result.ply
```

By default, `run_full_pipeline.py` directly blends the two existing source
trees. It does not rerun upstream MV-SAM3D, VGGT, DA3, or SAM3D model inference.
If source-A or source-B needs to be regenerated, set `rebuild_source_a: true` or
`rebuild_source_b: true` in `configs/paths.local.yaml` and fill the corresponding
`source_a_base_root` or `source_b_base_root`.

## Key Commands

Set these paths first:

```bash
export SAM3D_REPO=/path/to/sam-3d-objects
export GSO_ROOT=/path/to/GSO
export OBJECTS_FILE=configs/gso30_objects.txt
```

Run softVH on a source output tree:

```bash
python scripts/soft_visual_hull_refine.py \
  --pred_root /path/to/main_provider_appaff_source \
  --output_root /path/to/main_provider_appaff_softvh_th045_so012_s008 \
  --gso_root "$GSO_ROOT" \
  --objects_file "$OBJECTS_FILE" \
  --views 5 \
  --input_views 0-4 \
  --support_threshold 0.45 \
  --support_softness 0.12 \
  --max_shrink 0.008 \
  --max_opacity_drop 0.0 \
  --nvs_transform perm012_signpnn \
  --nvs_zflip \
  --nvs_focal 1.0
```

Run same-index geometry blend:

```bash
python scripts/blend_sameindex_geometry.py \
  --source_a /path/to/main_provider_appaff_softvh_th045_so012_s008 \
  --source_b /path/to/novggt_axisgrid_s020_appaff_s80 \
  --output_root /path/to/main_provider_appaff_softvh_alpha05_sameindex_geomblend \
  --objects_file "$OBJECTS_FILE" \
  --views 5 \
  --alpha 0.5
```

Evaluate four metrics:

```bash
python scripts/evaluate_gso30_four_metrics.py \
  --pred_root /path/to/main_provider_appaff_softvh_alpha05_sameindex_geomblend \
  --gso_root "$GSO_ROOT" \
  --repo "$SAM3D_REPO" \
  --objects_file "$OBJECTS_FILE" \
  --output_prefix results/main_provider_appaff_softvh_alpha05_sameindex_geomblend \
  --views 5 \
  --target_views 10-24 \
  --cd_transform perm021_signpnp \
  --nvs_transform perm012_signpnn \
  --nvs_zflip \
  --nvs_focal 1.0
```

## Current Artifact Status

The original full output directories for this exact version were not present on
this workstation when this repository was created. The available PLY examples
were copied into `examples/ply/` from the desktop request folder. To reproduce the
full branch, rerun the commands above using the original source output trees on
the server.
