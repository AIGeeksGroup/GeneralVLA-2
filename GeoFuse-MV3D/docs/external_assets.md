# External Assets

This repository intentionally does not include large checkpoints, GSO data, or
historical experiment output trees. Download or prepare them outside this repo
and point `configs/paths.example.yaml` to their locations.

## Required Code

### SAM 3D Objects

Official code:

```text
https://github.com/facebookresearch/sam-3d-objects
```

The official setup guide is in:

```text
https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md
```

The checkpoint download requires Hugging Face access:

```text
https://huggingface.co/facebook/sam-3d-objects
```

After access is approved, the official setup uses:

```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login
TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

### MV-SAM3D Experiment Code

This branch was developed on a patched MV-SAM3D/SAM3D Objects experiment repo
that contains `run_inference_weighted.py` and optional DA3 integration. The clean
Meta `sam-3d-objects` repo alone does not contain those MV-SAM3D wrappers.

If the colleague only wants to rerun this packaged branch, they can provide
already generated `source_a_root` and `source_b_root` output trees instead of
rerunning the upstream MV-SAM3D generation step.

### Depth Anything 3 / DA3

DA3 was used by some upstream provider branches, not by the final same-index
blend script itself. If rerunning the DA3-derived source branch, use the official
code:

```text
https://github.com/ByteDance-Seed/Depth-Anything-3
```

## Required Data

### GSO / GSO-30

Use the Google Scanned Objects data and the same 30-object subset listed in:

```text
configs/gso30_objects.txt
```

The expected local structure is:

```text
<GSO_ROOT>/<object>/render_mvs_25/model/000.png
<GSO_ROOT>/<object>/render_mvs_25/model/000.npy
...
<GSO_ROOT>/<object>/meshes/model.obj
```

Google's project page describes GSO as an open-source collection of over one
thousand 3D-scanned household items:

```text
https://research.google/pubs/google-scanned-objects-a-high-quality-dataset-of-3d-scanned-household-items/
```

## Required Intermediate Outputs

The direct reproduction path needs two external output trees:

```text
source_a_root = main_provider_appaff branch outputs
source_b_root = novggt_axisgrid_s020_appaff_s80 branch outputs
```

Each tree must use this layout:

```text
<source_root>/<object>/5views/result.ply
<source_root>/<object>/5views/result.glb
```

These are generated experiment outputs, so they may be too large for a desktop
handoff. If they are not available, regenerate them on the server using the
upstream MV-SAM3D experiment code, then point the YAML config to their paths.
