from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from waveslab.covers import (
    compose_fragmented_channel,
    cover_roughness,
    field_metrics,
    json_ready,
    load_sam_cover_sources,
    row_mean_cover,
    scale_to_mean,
)
from waveslab.env import safe_stem, slug_float
from waveslab.model_config import BridgeSettings, SamCoverSettings, SolverSettings
from waveslab.plotting import save_centerline_overlay, save_cover_map, save_surface_with_cover
from waveslab.pywave_adapter import PyWaveAdapter


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip())


def parse_str_list(text: str) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in text.replace(";", ",").split(",") if x.strip())


def run_case(adapter: PyWaveAdapter, F: np.ndarray, cfg, out_dir: Path, label: str) -> dict:
    run = adapter.run_or_load(F, out_dir / "runs" / label, cfg, label=label)
    return {
        "label": label,
        "grid_x": np.asarray(run["grid_x"], dtype=float),
        "grid_y": np.asarray(run["grid_y"], dtype=float),
        "Z": np.asarray(run["Z"], dtype=float),
        "F": np.asarray(run["F"], dtype=float),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge smooth logistic channels to fragmented image-derived covers.")
    parser.add_argument("--pipeline-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/fragmented_channel_bridge"))
    parser.add_argument("--sources", type=parse_str_list, default=("direct", "ellipse"))
    parser.add_argument("--sigmas", type=parse_float_list, default=(4.0,))
    parser.add_argument("--modes", type=parse_str_list, default=("texture",))
    parser.add_argument("--alphas", type=parse_float_list, default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--N", type=int, default=121)
    parser.add_argument("--M", type=int, default=61)
    parser.add_argument("--dx", type=float, default=10.0 / 31.0)
    parser.add_argument("--dy", type=float, default=10.0 / 31.0)
    parser.add_argument("--Fr", type=float, default=0.7)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = PyWaveAdapter()
    solver = SolverSettings(N=args.N, M=args.M, dx=args.dx, dy=args.dy, Fr=args.Fr, epsilon=args.epsilon)
    bridge = BridgeSettings(sigmas=tuple(args.sigmas), compose_modes=tuple(args.modes), alphas=tuple(args.alphas))
    sam = SamCoverSettings(pipeline_dir=args.pipeline_dir, sources=tuple(args.sources))
    cfg = adapter.solver_config(solver)
    x, y = adapter.make_grid(cfg)
    sources = load_sam_cover_sources(
        adapter=adapter,
        settings=sam,
        out_dir=out_dir / "cover_models" / "source_covers",
        rows=int(cfg.M),
        cols=int(cfg.N),
    )

    records: list[dict] = []
    cases_by_sigma: dict[str, list[dict]] = {}
    for sigma in bridge.sigmas:
        beta_1d = adapter.logistic_beta_y(y, gamma=bridge.gamma, sigma=sigma)
        beta = np.repeat(beta_1d.reshape(-1, 1), len(x), axis=1)
        baseline_label = safe_stem("continuous", "sigma", slug_float(sigma))
        baseline_case = run_case(adapter, beta, cfg, out_dir, baseline_label)
        cases_by_sigma.setdefault(str(sigma), []).append(baseline_case)

        fig_dir = out_dir / "figures" / f"sigma_{slug_float(sigma)}"
        save_cover_map(beta, x, y, fig_dir / f"{baseline_label}__cover.png")
        save_surface_with_cover(x=x, y=y, Z=baseline_case["Z"], F=baseline_case["F"], out_png=fig_dir / f"{baseline_label}__surface_cover.png")
        records.append({"label": baseline_label, "source": "analytic", "mode": "continuous", "sigma": float(sigma), **field_metrics(baseline_case["Z"])})

        for source_name, info in sources.items():
            S = np.clip(np.asarray(info["F"], dtype=float), 0.0, 1.0)
            for mode in bridge.compose_modes:
                if mode == "texture":
                    for alpha in bridge.alphas:
                        F = compose_fragmented_channel(S, beta, mode=mode, texture_contrast=bridge.texture_contrast * float(alpha))
                        label = safe_stem(source_name, mode, "alpha", slug_float(alpha), "sigma", slug_float(sigma))
                        case = run_case(adapter, F, cfg, out_dir, label)
                        save_cover_map(case["F"], case["grid_x"], case["grid_y"], fig_dir / f"{label}__cover.png")
                        save_surface_with_cover(x=case["grid_x"], y=case["grid_y"], Z=case["Z"], F=case["F"], out_png=fig_dir / f"{label}__surface_cover.png")
                        cases_by_sigma[str(sigma)].append(case)
                        records.append({"label": label, "source": source_name, "mode": mode, "alpha": float(alpha), "sigma": float(sigma), **cover_roughness(F, dx=float(cfg.dx), dy=float(cfg.dy)), **field_metrics(case["Z"], baseline_case["Z"])})
                else:
                    F = compose_fragmented_channel(S, beta, mode=mode, texture_contrast=bridge.texture_contrast)
                    label = safe_stem(source_name, mode, "sigma", slug_float(sigma))
                    case = run_case(adapter, F, cfg, out_dir, label)
                    save_cover_map(case["F"], case["grid_x"], case["grid_y"], fig_dir / f"{label}__cover.png")
                    save_surface_with_cover(x=case["grid_x"], y=case["grid_y"], Z=case["Z"], F=case["F"], out_png=fig_dir / f"{label}__surface_cover.png")
                    cases_by_sigma[str(sigma)].append(case)
                    records.append({"label": label, "source": source_name, "mode": mode, "alpha": None, "sigma": float(sigma), **cover_roughness(F, dx=float(cfg.dx), dy=float(cfg.dy)), **field_metrics(case["Z"], baseline_case["Z"])})

    for sigma_key, cases in cases_by_sigma.items():
        save_centerline_overlay(cases, out_dir / "figures" / f"sigma_{slug_float(float(sigma_key))}" / "centerline_overlay.png")

    csv_path = out_dir / "bridge_metrics.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in records for k in r.keys()}))
            writer.writeheader()
            writer.writerows(records)
    summary = {"solver": solver, "bridge": bridge, "sam_cover": sam, "metrics_csv": csv_path, "records": records}
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
