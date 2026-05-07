# WavesLab

Clean steady-case research software for hydroelastic wave modelling and image-derived ice-cover representations.

This public seed focuses on one workflow:

1. build or load an ice-cover field `F(x,y)`,
2. map it to a steady hydroelastic wave model,
3. run analytic, homogeneous, direct-mask, ellipse, or fragmented-channel cases,
4. save small reproducible outputs for comparison.

The unsteady solver and thesis-only figure scripts are intentionally left out for now.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For sea-ice segmentation with SAM:

```bash
pip install -e ".[segmentation]"
```

SAM weights are not included.

## Examples

Analytic ridge/channel sweep:

```bash
python examples/run_logistic_sweep.py --out outputs/logistic_sweep
```

Build solver-ready direct and ellipse covers from SAM outputs:

```bash
python examples/build_cover_models_from_sam.py \
  --pipeline-dir /path/to/sam_single_image_vscode \
  --out outputs/cover_models
```

Image-derived direct/ellipse covers compared with x-homogenized partners:

```bash
SAM_PIPELINE_DIR=/path/to/sam_single_image_vscode \
python examples/run_image_scattering_comparison.py --crop-scales 1,1.5,2
```

Smooth-to-fragmented channel bridge:

```bash
SAM_PIPELINE_DIR=/path/to/sam_single_image_vscode \
python examples/run_fragmented_channel_bridge.py --modes texture,product --sigmas 4
```

Estimate characteristic wavelengths from a deep-water baseline:

```bash
python scripts/estimate_characteristic_wavelengths.py --Fr auto --aleph 0.5
```

Fair comparison of sea-ice segmentation methods:

```bash
python examples/segment_sea_ice_fair_compare.py
```

## Layout

```text
src/waveslab/cover_core.py            cover builders and cover selection
src/waveslab/cover_rendering.py       publication-style cover plots
src/waveslab/steady/                  steady hydroelastic solver components
src/waveslab/imaging/                 sea-ice segmentation utilities
examples/                             small runnable workflows
scripts/                              analysis utilities
docs/original_file_map.md             map from old files to the cleaned layout
```

Generated outputs go under `outputs/`, which is ignored by Git.
