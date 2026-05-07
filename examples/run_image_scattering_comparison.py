from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PIPELINE_DIR = Path("outputs/sea_ice_segmentation")


def _nvidia_gpu_seems_present() -> bool:
    """Return True when the machine appears to have an NVIDIA driver/GPU."""
    return shutil.which("nvidia-smi") is not None or Path("/proc/driver/nvidia/version").exists()


def configure_jax_backend(backend: str) -> None:
    """Request the JAX backend before importing modules that may initialize JAX.

    The default for this script is strict CUDA. That is deliberate: these image
    scattering runs are expensive enough that silently falling back to CPU is
    usually worse than stopping immediately with a clear installation error.
    """
    if backend == "cuda":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
    elif backend == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif backend == "auto":
        return
    else:  # argparse prevents this in normal use.
        raise ValueError(f"Unknown backend {backend!r}")


def report_jax_backend(required_backend: str) -> None:
    """Print the active JAX backend and fail clearly if CUDA was requested."""
    try:
        import jax

        devices = jax.devices()
        default_backend = jax.default_backend()
    except Exception as exc:
        if required_backend == "cuda":
            raise RuntimeError(
                "Requested --backend cuda, but JAX could not initialize CUDA.\n"
                "This usually means the current virtual environment has CPU-only JAX/JAXLIB.\n\n"
                "Try reinstalling CUDA-enabled JAX inside this venv:\n"
                "  python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt\n"
                '  python -m pip install -U "jax[cuda13]"\n\n'
                "Then verify with:\n"
                "  python - <<'PY'\n"
                "  import jax\n"
                "  print(jax.default_backend())\n"
                "  print(jax.devices())\n"
                "  PY"
            ) from exc
        raise

    print(f"[jax] version: {jax.__version__}")
    print(f"[jax] default backend: {default_backend}")
    print("[jax] devices:")
    for dev in devices:
        print(f"  - {dev}")

    gpu_devices = [
        dev
        for dev in devices
        if getattr(dev, "platform", "").lower() in {"gpu", "cuda"}
    ]

    if required_backend == "cuda" and not gpu_devices:
        raise RuntimeError(
            "Requested --backend cuda, but JAX did not expose any CUDA/GPU devices.\n"
            "The machine may have a GPU, but this virtual environment is not using a CUDA-enabled JAX install.\n\n"
            "Try reinstalling CUDA-enabled JAX inside this venv:\n"
            "  python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt\n"
            '  python -m pip install -U "jax[cuda13]"'
        )

    if required_backend == "auto" and default_backend == "cpu" and _nvidia_gpu_seems_present():
        print(
            "[jax warning] NVIDIA GPU detected, but JAX is using CPU. "
            "Use --backend cuda to fail fast, or install CUDA-enabled JAX."
        )


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip())


def parse_str_list(text: str) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in text.replace(";", ",").split(",") if x.strip())


def run_cover_case(adapter: Any, F: np.ndarray, cfg: Any, out_dir: Path, label: str) -> dict:
    run = adapter.run_or_load(F, out_dir / "runs" / label, cfg, label=label)
    return {
        "label": label,
        "grid_x": np.asarray(run["grid_x"], dtype=float),
        "grid_y": np.asarray(run["grid_y"], dtype=float),
        "Z": np.asarray(run["Z"], dtype=float),
        "F": np.asarray(run["F"], dtype=float),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare image-derived covers with their x-homogenized partners.")
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help="Segmentation output folder. Default: outputs/sea_ice_segmentation.",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/image_scattering"))
    parser.add_argument("--sources", type=parse_str_list, default=("direct", "ellipse"))
    parser.add_argument(
        "--selected-method",
        default=None,
        help=(
            "Optional segmentation method to use instead of data/selected_floe_labels.npz, "
            "for example sam_auto, morphology, or kmeans_watershed."
        ),
    )
    parser.add_argument("--direct-source", default="accepted_label_map")
    parser.add_argument(
        "--overwrite-compat",
        action="store_true",
        help="Rewrite data/pipeline_arrays.npz and data/floe_catalog.csv before building covers.",
    )
    parser.add_argument("--crop-scales", type=parse_float_list, default=(1.0,))
    parser.add_argument("--N", type=int, default=240)
    parser.add_argument("--M", type=int, default=120)
    parser.add_argument("--dx", type=float, default=0.2)
    parser.add_argument("--dy", type=float, default=0.4)
    parser.add_argument("--x0", type=float, default=-24.0)
    parser.add_argument("--Fr", type=float, default=1.2)
    parser.add_argument("--aleph", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--backend",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help=(
            "JAX backend to request. Default: cuda, which prevents accidental CPU fallback. "
            "Use --backend auto to allow JAX to choose, or --backend cpu for a deliberate CPU run."
        ),
    )
    parser.add_argument("--no-mean-match", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_jax_backend(args.backend)
    report_jax_backend(args.backend)

    # Import JAX-dependent project modules only after backend selection.
    try:
        from build_cover_models_from_sam import ensure_cover_builder_inputs
    except Exception as exc:  # pragma: no cover - sibling script should normally be available
        raise RuntimeError(
            "Could not import ensure_cover_builder_inputs from build_cover_models_from_sam.py. "
            "Make sure build_cover_models_from_sam.py is beside this script."
        ) from exc

    from waveslab.covers import (
        centered_clamped_crop,
        coerce_shape,
        cover_roughness,
        crop_label,
        field_metrics,
        json_ready,
        load_sam_cover_sources,
        row_mean_cover,
        shift_to_mean,
    )
    from waveslab.env import safe_stem
    from waveslab.model_config import ImageScatteringSettings, SamCoverSettings, SolverSettings
    from waveslab.plotting import save_centerline_overlay, save_cover_map, save_delta_map, save_surface_with_cover
    from waveslab.pywave_adapter import PyWaveAdapter

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = PyWaveAdapter()
    solver = SolverSettings(
        N=args.N,
        M=args.M,
        dx=args.dx,
        dy=args.dy,
        x0=args.x0,
        Fr=args.Fr,
        aleph=args.aleph,
        mu=args.mu,
        epsilon=args.epsilon,
        full_domain=True,
        rigidity_min=0.08,
    )
    cfg = adapter.solver_config(solver)
    x, y = adapter.make_grid(cfg)
    sam = SamCoverSettings(
        pipeline_dir=args.pipeline_dir,
        direct_source=args.direct_source,
        sources=tuple(args.sources),
    )
    settings = ImageScatteringSettings(crop_scales=tuple(args.crop_scales), match_native_mean=not args.no_mean_match)

    pipeline_dir = adapter.resolve_sam_pipeline_dir(sam.pipeline_dir)
    compatibility = ensure_cover_builder_inputs(
        pipeline_dir,
        selected_method=args.selected_method,
        overwrite=args.overwrite_compat or args.selected_method is not None,
    )
    print("[cover inputs]")
    for action in compatibility["actions"]:
        print(f"  - {action}")

    full_shape, mask_key = adapter.full_binary_shape(pipeline_dir, direct_source=sam.direct_source)
    native_means: dict[str, float] = {}
    built: list[dict] = []

    for scale in settings.crop_scales:
        crop = centered_clamped_crop(
            scale=scale,
            full_shape_hw=full_shape,
            base_x0=sam.crop_x0,
            base_y0=sam.crop_y0,
            base_w=sam.crop_w,
            base_h=sam.crop_h,
        )
        label_crop = crop_label(crop)
        sources = load_sam_cover_sources(
            adapter=adapter,
            settings=sam,
            out_dir=out_dir / "cover_models" / label_crop,
            rows=int(cfg.M),
            cols=int(cfg.N),
            crop=crop,
        )
        for source_name, info in sources.items():
            F_raw = coerce_shape(info["F"], rows=int(cfg.M), cols=int(cfg.N), name=source_name)
            if abs(float(crop["requested_scale"]) - 1.0) < 1e-12:
                native_means[source_name] = float(np.mean(F_raw))
            built.append({"crop": crop, "crop_label": label_crop, "source": source_name, "F_raw": F_raw, "data": info.get("data", {})})

    records: list[dict] = []
    cases_for_overlay: list[dict] = []
    for item in built:
        source = item["source"]
        label_crop = item["crop_label"]
        F_raw = np.asarray(item["F_raw"], dtype=float)
        if settings.match_native_mean and source in native_means:
            F_raw = shift_to_mean(F_raw, native_means[source])
        F_hom = row_mean_cover(F_raw)

        for representation, F in (("raw_2d", F_raw), ("x_homogenized", F_hom)):
            label = safe_stem(source, representation, label_crop)
            case = run_cover_case(adapter, F, cfg, out_dir, label)
            cases_for_overlay.append(case)
            fig_dir = out_dir / "figures" / label_crop / source
            save_cover_map(case["F"], case["grid_x"], case["grid_y"], fig_dir / f"{label}__cover.png")
            save_surface_with_cover(x=case["grid_x"], y=case["grid_y"], Z=case["Z"], F=case["F"], out_png=fig_dir / f"{label}__surface_cover.png")
            record = {
                "label": label,
                "source": source,
                "representation": representation,
                "crop_label": label_crop,
                **cover_roughness(F, dx=float(cfg.dx), dy=float(cfg.dy)),
                **field_metrics(case["Z"]),
            }
            records.append(record)

        raw_case = cases_for_overlay[-2]
        hom_case = cases_for_overlay[-1]
        delta = raw_case["Z"] - hom_case["Z"]
        delta_label = safe_stem(source, "raw_minus_x_homogenized", label_crop)
        save_delta_map(delta, raw_case["grid_x"], raw_case["grid_y"], out_dir / "figures" / label_crop / source / f"{delta_label}__delta_zeta.png")
        records[-2].update(field_metrics(raw_case["Z"], hom_case["Z"], prefix="raw_minus_hom_"))
        save_centerline_overlay([hom_case, raw_case], out_dir / "figures" / label_crop / source / f"{delta_label}__centerline.png")

    csv_path = out_dir / "image_scattering_metrics.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in records for k in r.keys()}))
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "pipeline_dir": pipeline_dir,
        "selected_mask_key": mask_key,
        "compatibility_inputs": compatibility,
        "full_binary_shape_hw": full_shape,
        "solver": solver,
        "sam_cover": sam,
        "settings": settings,
        "native_means": native_means,
        "metrics_csv": csv_path,
        "records": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()