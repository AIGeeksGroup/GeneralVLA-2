# Reproduction Notes

## Fastest Path

If you already have the two intermediate output trees, you only need to edit
`configs/paths.local.yaml` and run:

```bash
python scripts/run_full_pipeline.py --config configs/paths.local.yaml
```

This default path only runs the final same-index geometry blend and optional
evaluation. It does not download or store checkpoints, GSO data, or historical
experiment output trees inside this repository.

## Required Inputs

You need these external assets:

1. A working MV-SAM3D / SAM3D Objects repository.
2. GSO-30 data with `render_mvs_25/model/*.png`, camera `.npy` files, and
   `meshes/model.obj`.
3. Source A output tree:
   `main_provider_appaff_softvh_th045_so012_s008`.
4. Source B output tree:
   `novggt_axisgrid_s020_appaff_s80`.

See `docs/external_assets.md` for download links and external setup notes.

Large files should stay outside this repository. Put their filesystem locations
in `configs/paths.local.yaml`.

The expected output layout is:

```text
<output_root>/<object>/5views/result.ply
<output_root>/<object>/5views/result.glb
```

## Reproduce Final Branch

```bash
python scripts/blend_sameindex_geometry.py \
  --source_a /path/to/main_provider_appaff_softvh_th045_so012_s008 \
  --source_b /path/to/novggt_axisgrid_s020_appaff_s80 \
  --output_root /path/to/main_provider_appaff_softvh_alpha05_sameindex_geomblend \
  --objects_file configs/gso30_objects.txt \
  --views 5 \
  --alpha 0.5
```

The script writes:

```text
/path/to/main_provider_appaff_softvh_alpha05_sameindex_geomblend/geometry_only_blend_summary.csv
```

Inspect `ply_status` and `mesh_status`. `ok` means real same-index blending was
performed. `copied_a_*` means the object fell back to source A.

## Recompute softVH Source A

If the source-A branch needs to be regenerated from a provider/appaff root:

```bash
python scripts/soft_visual_hull_refine.py \
  --pred_root /path/to/main_provider_appaff_root \
  --output_root /path/to/main_provider_appaff_softvh_th045_so012_s008 \
  --gso_root /path/to/GSO \
  --objects_file configs/gso30_objects.txt \
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

## Recompute no-VGGT axis2 Source B

If the no-VGGT axis2 branch needs to be regenerated:

```bash
python scripts/optimize_gaussian_axis_refine_meshsync.py \
  --pred_root /path/to/baseline_or_novggt_source \
  --output_root /path/to/novggt_axisgrid_s020_appaff_s80 \
  --gso_root /path/to/GSO \
  --repo /path/to/sam-3d-objects \
  --objects_file configs/gso30_objects.txt \
  --views 5 \
  --train_views 0-4 \
  --steps 80 \
  --nvs_transform perm012_signpnn \
  --nvs_zflip \
  --nvs_focal 1.0
```

This axis-refine command is representative of the local script used for the
axis2 family. Exact upstream source roots should be matched to the server
experiment directory when rerunning historical numbers.

## Evaluate

```bash
python scripts/evaluate_gso30_four_metrics.py \
  --pred_root /path/to/main_provider_appaff_softvh_alpha05_sameindex_geomblend \
  --gso_root /path/to/GSO \
  --repo /path/to/sam-3d-objects \
  --objects_file configs/gso30_objects.txt \
  --output_prefix results/main_provider_appaff_softvh_alpha05_sameindex_geomblend \
  --views 5 \
  --target_views 10-24 \
  --cd_transform perm021_signpnp \
  --nvs_transform perm012_signpnn \
  --nvs_zflip \
  --nvs_focal 1.0
```

## Recorded Metrics

```text
baseline full30 approx:
  CD    = 45.8759e-3
  PSNR  = 13.242106
  SSIM  = 0.805061
  LPIPS = 0.279508

this method full30 approx:
  CD    = 44.7949e-3
  PSNR  = 13.552471
  SSIM  = 0.813360
  LPIPS = 0.273675
```
