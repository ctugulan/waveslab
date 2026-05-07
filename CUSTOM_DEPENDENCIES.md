# Dependencies

The refactor vendors the previously required steady `pywave.waves_helpers` code into `src/waveslab`. The main solver and cover examples no longer require the old custom `pywave` package.

## Required Python packages

```text
numpy
scipy
matplotlib
pandas
pillow
opencv-python
scikit-image
jax
jaxlib
```

## Optional segmentation packages

These are only needed for `examples/segment_sea_ice_fair_compare.py`:

```text
torch
segment-anything
cameratransform
```

You also need SAM model weights, for example `sam_vit_h_4b8939.pth`, if you run the SAM-based segmentation comparison. Those weights are not included.

## Still not included

Raw research outputs, large generated run folders, and unsteady solver code are intentionally excluded from this steady-only public seed.
