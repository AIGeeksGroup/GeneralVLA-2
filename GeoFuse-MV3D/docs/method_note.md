# Method Note

## Short Summary

`main_provider_appaff_softvh_alpha05_sameindex_geomblend` is a conservative
post-processing branch for MV-SAM3D. It improves geometry and view synthesis
metrics without changing the input image set.

It should be described as:

> VGGT-derived main-provider/appaff output + input-mask soft visual hull +
> no-VGGT axis2 geometry same-index blend.

It should **not** be described as a purely non-VGGT method.

## Step 1: Main Provider + Appaff Base

The source-A branch is `main_provider_appaff_softvh_th045_so012_s008`.
Before softVH, the branch already comes from the `main_provider/appaff` family.
Experiment notes indicate this family uses the VGGT-derived branch upstream.

`appaff` means appearance affine calibration. It is a low-dimensional color
adjustment on the Gaussian appearance, intended to improve input-view rendering
without changing object identity or selecting better target views.

## Step 2: softVH

`softVH` is short for soft visual hull.

For every Gaussian point or mesh vertex, the algorithm:

1. Projects the 3D point into each of the five input views.
2. Samples the alpha mask value at the projected 2D location.
3. Averages the valid mask values to obtain a support score.
4. Converts low support into a small shrink strength.
5. Moves the point slightly toward the object center.

The important design choice is that no point is hard-deleted. Low-support regions
are only adjusted continuously. This avoids the visible failure mode where a
method improves CD by removing uncertain geometry but produces broken visual
models.

The recorded best softVH parameters for this branch are:

```text
support_threshold = 0.45
support_softness  = 0.12
max_shrink        = 0.008
max_opacity_drop  = 0.0
```

`max_opacity_drop=0.0` is intentional. Earlier experiments showed that changing
opacity can improve metrics but may make the rendered object look brighter on a
white background.

## Step 3: no-VGGT axis2 Geometry Branch

The source-B branch is `novggt_axisgrid_s020_appaff_s80`.

This branch is used as a geometry compensation signal. It is lower-dimensional
than a full regeneration: it uses small axis-wise scale and shift corrections,
then synchronizes the same transform to both Gaussian centers and the GLB mesh.

The key idea is to get an orthogonal geometry correction that does not depend on
VGGT, then blend only where source A and source B are structurally compatible.

## Step 4: same-index Geometry Blend

The final branch is created by `scripts/blend_sameindex_geometry.py`.

For each object:

1. Copy source A output to the destination.
2. If source A and source B `result.ply` have the same vertex count, blend only
   `x,y,z` using:

```text
xyz_out = (1 - alpha) * xyz_A + alpha * xyz_B
```

3. If source A and source B `result.glb` have the same mesh vertex count, apply
   the same blend to mesh vertices.
4. If either file is incompatible, keep source A unchanged.

The recorded version uses:

```text
alpha = 0.5
```

This is why the method is conservative. It never blends color, opacity, scale,
rotation, or SH appearance fields. It also does not force incompatible topology
to match.

## Why This Helped

The source-A softVH branch already has stable four-metric gains over the
baseline. The no-VGGT axis2 branch can improve some geometry failures. The
same-index blend lets the method absorb axis2 geometry where the two outputs are
compatible, while falling back to the safer source-A output elsewhere.

The result improved full30 approximate CD and NVS metrics compared with the
baseline, but it did not reach the original goal of +5% on all four metrics.

## Limitations

- This version is VGGT-derived through the source-A family.
- The final blend is only valid when vertex indexing is compatible.
- Many objects may simply copy source A if topology differs.
- It is a post-processing branch, not a change inside MV-SAM3D's core generation
  network.
- The NVS numbers use a fixed self-built GaussianRenderer bridge, not an
  official unreleased MV-SAM3D benchmark renderer.
