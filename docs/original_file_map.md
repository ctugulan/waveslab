# Original file map

This refactor treats the uploaded scripts as a thesis workbench and reduces them to a smaller steady-case public repo seed.

| Original file | Purpose | Refactor decision |
|---|---|---|
| `wrap_vary_cover_logistic.py` | Runs analytic ridge and channel cover sweeps using `F(x,y)=beta(y)`. | Replaced by `examples/run_logistic_sweep.py`. Shared plotting and metrics moved to `src/waveslab`. |
| `wrap_pancake_channel.py` | Composes logistic channel envelopes with direct and ellipse SAM-derived floe fields using `texture`, `product`, and `row_normalized_product` modes. | Replaced by `examples/run_fragmented_channel_bridge.py`. Composition rules moved to `src/waveslab/covers.py`. |
| `wrap_cover_bridge_stage1.py` | Builds a ladder from smooth logistic channels to rough data-derived covers to diagnose solver failure. | Merged into `examples/run_fragmented_channel_bridge.py`. The first public version keeps the direct and ellipse sources and the main composition ladder. |
| `wrap_scattering_1x1.py` | Runs the native 1x1 SAM crop and compares raw 2D cover with its x-homogenized partner. | Merged into `examples/run_image_scattering_comparison.py` with `--crop-scales 1`. |
| `wrap_scattering_largecrop_coarse.py` | Extends the same raw-minus-x-homogenized comparison to larger crops mapped onto the same solver grid. | Merged into `examples/run_image_scattering_comparison.py` with `--crop-scales 1,1.5,2`. |
| `wrap_reproduce_direct_ellipse_full_domain.py` | Patches the custom runner to reproduce direct and ellipse full-domain cases. | Covered by `examples/run_image_scattering_comparison.py`; the clean example runs direct and ellipse through one path. |
| `wrap_reproduce_direct_ellipse_lambda_scaled.py` | Rescales direct and ellipse covers so floe diameter matches a characteristic wavelength. | Deferred. Wavelength estimation remains in `scripts/estimate_characteristic_wavelengths.py`; geometry rescaling can be added later as a focused module. |
| `deep_characteristic_wavelengths.py` | Runs a deep-water baseline and estimates short and long wavelengths from the y=0 profile. | Replaced by `scripts/estimate_characteristic_wavelengths.py` and `src/waveslab/wavelengths.py`. |
| `pub_render_surface_direct_cover.py` | Post-processes one old run into a publication-style surface with cover projected underneath. | Replaced by the generic `save_surface_with_cover` function in `src/waveslab/plotting.py`. |

## Simplified workflow

1. Build or load a cover field `F(x,y)`.
2. Run the steady hydroelastic solver through `PyWaveAdapter`.
3. Save a small set of outputs: cover map, surface with cover, centerline comparison, metrics CSV, and summary JSON.
4. Keep all generated outputs under `outputs/`, which is ignored by Git.


## Additional files incorporated in this pass

| Original file | Purpose | Refactor decision |
|---|---|---|
| `sam_pancake_fair_compare.py` | Fair comparison between SAM automatic masks, threshold morphology, and k-means/watershed segmentation. | Moved to `src/waveslab/imaging/sea_ice_segmentation.py`; `examples/segment_sea_ice_fair_compare.py` is the public entry point. |
| `build_cover_npzs_from_sam_pipeline.py` | Press-play wrapper that builds direct and ellipse cover NPZs from SAM outputs. | Replaced by `examples/build_cover_models_from_sam.py`, which calls `waveslab.cover_core.build_cover_npzs_from_sam_pipeline`. |
| `cover_core.py` | Core cover construction, loading, scaling, homogeneous cases, and cover selection. | Moved to `src/waveslab/cover_core.py`. |
| `cover_rendering.py` | Publication-style surface and cross-section plots for cover runs. | Moved to `src/waveslab/cover_rendering.py`. |
| `run_cover_cases.py` | Main steady SAM-cover runner. | Moved to `src/waveslab/steady/cover_runner.py` and patched to use `waveslab` modules. |
| `driver_steady_cases.py` | Half-domain steady solver driver for deep, flatbed, and bathymetry cases. | Moved to `src/waveslab/steady/driver.py`. |
| `full_domain.py` | Full-y infinite-depth steady solver helpers. | Moved to `src/waveslab/steady/full_domain.py`. |
| `jax_biharmonic.py`, `error_all.py`, `build_all_blocks.py`, `block_schur_inv.py` | Biharmonic stencil, residuals, Jacobian blocks, and Schur preconditioners. | Moved into `src/waveslab/steady/` as `biharmonic.py`, `residuals.py`, `blocks.py`, and `preconditioners.py`. |
| `classify_regimes.py` | Infinite-depth, finite-depth, and viscoelastic regime classification. | Moved to `src/waveslab/steady/regimes.py`. |
| `names.py` | Filename and output-layout helpers. | Moved to `src/waveslab/steady/names.py`. |
| `plot_steady_all.py` | Steady result plotting helper. | Moved to `src/waveslab/steady/plotting.py` with a local plotting fallback so the old `viscice_demo_plots` helper is no longer required. |
