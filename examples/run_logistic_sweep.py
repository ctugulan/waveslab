from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from waveslab.covers import field_metrics, json_ready
from waveslab.env import safe_stem, slug_float
from waveslab.model_config import LogisticSweepSettings, SolverSettings
from waveslab.plotting import save_centerline_overlay, save_cover_map, save_surface_with_cover
from waveslab.pywave_adapter import PyWaveAdapter


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip())


def run_family(
    *,
    adapter: PyWaveAdapter,
    family: str,
    gamma: float,
    sigmas: tuple[float, ...],
    solver: SolverSettings,
    out_dir: Path,
) -> list[dict]:
    cfg = adapter.solver_config(solver)
    x, y = adapter.make_grid(cfg)
    cases: list[dict] = []
    for sigma in sigmas:
        beta = adapter.logistic_beta_y(y, gamma=gamma, sigma=sigma).reshape(-1, 1)
        F = np.repeat(np.clip(beta, 0.0, 1.0), len(x), axis=1)
        label = f"{family}_gamma{slug_float(gamma)}_sigma{slug_float(sigma)}"
        run_dir = out_dir / "runs" / label
        print(f"[run] {label}")
        run = adapter.run_or_load(F, run_dir, cfg, label=label)
        case = {
            "label": label,
            "family": family,
            "gamma": float(gamma),
            "sigma": float(sigma),
            "grid_x": np.asarray(run["grid_x"], dtype=float),
            "grid_y": np.asarray(run["grid_y"], dtype=float),
            "Z": np.asarray(run["Z"], dtype=float),
            "F": np.asarray(run["F"], dtype=float),
        }
        fig_dir = out_dir / "figures" / family
        save_cover_map(case["F"], case["grid_x"], case["grid_y"], fig_dir / f"{label}__cover.png")
        save_surface_with_cover(x=case["grid_x"], y=case["grid_y"], Z=case["Z"], F=case["F"], out_png=fig_dir / f"{label}__surface_cover.png")
        case.update(field_metrics(case["Z"]))
        cases.append(case)
    save_centerline_overlay(cases, out_dir / "figures" / f"{family}__centerline_overlay.png", label=f"{family} sweep")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small steady sweep of analytic logistic cover fields.")
    parser.add_argument("--out", type=Path, default=Path("outputs/logistic_sweep"))
    parser.add_argument("--N", type=int, default=121)
    parser.add_argument("--M", type=int, default=61)
    parser.add_argument("--dx", type=float, default=10.0 / 31.0)
    parser.add_argument("--dy", type=float, default=10.0 / 31.0)
    parser.add_argument("--Fr", type=float, default=0.7)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--ridge-sigmas", type=parse_float_list, default=(-4.0, 0.0, 4.0, 8.0))
    parser.add_argument("--channel-sigmas", type=parse_float_list, default=(-4.0, 0.0, 4.0, 8.0))
    parser.add_argument("--skip-ridge", action="store_true")
    parser.add_argument("--skip-channel", action="store_true")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = PyWaveAdapter()
    solver = SolverSettings(N=args.N, M=args.M, dx=args.dx, dy=args.dy, Fr=args.Fr, epsilon=args.epsilon)
    sweep = LogisticSweepSettings(
        ridge_sigmas=tuple(args.ridge_sigmas),
        channel_sigmas=tuple(args.channel_sigmas),
        run_ridge=not args.skip_ridge,
        run_channel=not args.skip_channel,
    )

    all_cases: list[dict] = []
    if sweep.run_ridge:
        all_cases.extend(run_family(adapter=adapter, family="ridge", gamma=sweep.ridge_gamma, sigmas=sweep.ridge_sigmas, solver=solver, out_dir=out_dir))
    if sweep.run_channel:
        all_cases.extend(run_family(adapter=adapter, family="channel", gamma=sweep.channel_gamma, sigmas=sweep.channel_sigmas, solver=solver, out_dir=out_dir))

    csv_path = out_dir / "logistic_sweep_metrics.csv"
    public_rows = [{k: v for k, v in c.items() if k not in {"Z", "F", "grid_x", "grid_y"}} for c in all_cases]
    if public_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(public_rows[0].keys()))
            writer.writeheader()
            writer.writerows(public_rows)
    (out_dir / "summary.json").write_text(json.dumps(json_ready({"solver": solver, "sweep": sweep, "cases": public_rows}), indent=2), encoding="utf-8")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
