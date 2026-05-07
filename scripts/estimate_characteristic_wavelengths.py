from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from waveslab.covers import json_ready
from waveslab.pywave_adapter import load_deep_runner
from waveslab.wavelengths import WavelengthSettings, auto_supercritical_F, summarize_wavelengths


def save_diagnostic_plot(x: np.ndarray, z: np.ndarray, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.5), constrained_layout=True)
    ax.plot(x, z, lw=2.0)
    ax.axvline(0.0, ls="--", lw=1.0, color="k", alpha=0.6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\zeta(x,0)$")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate short and long wavelengths from a steady deep-water baseline.")
    parser.add_argument("--out", type=Path, default=Path("outputs/deep_wavelength_baseline"))
    parser.add_argument("--N", type=int, default=240)
    parser.add_argument("--M", type=int, default=80)
    parser.add_argument("--dx", type=float, default=0.3)
    parser.add_argument("--dy", type=float, default=0.6)
    parser.add_argument("--Fr", type=str, default="auto")
    parser.add_argument("--aleph", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--near-skip", type=float, default=4.0)
    parser.add_argument("--edge-trim", type=float, default=3.0)
    args = parser.parse_args()

    settings = WavelengthSettings(
        N=args.N,
        M=args.M,
        dx=args.dx,
        dy=args.dy,
        Fr=None if args.Fr == "auto" else float(args.Fr),
        aleph=args.aleph,
        mu=args.mu,
        epsilon=args.epsilon,
        near_skip=args.near_skip,
        edge_trim=args.edge_trim,
    )
    Fr = settings.Fr if settings.Fr is not None else auto_supercritical_F(settings.aleph)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    run_cases = load_deep_runner()
    Z, csv_path, fig_path = run_cases.run_case_deep(
        out_base=out,
        N=settings.N,
        M=settings.M,
        dx=settings.dx,
        dy=settings.dy,
        x0=settings.x0,
        Fr=Fr,
        aleph=settings.aleph,
        mu=settings.mu,
        epsilon=settings.epsilon,
        Lx=settings.Lx,
        Ly=settings.Ly,
        tauf=settings.tauf,
        use_radiation=False,
    )
    grid = run_cases.make_grid(settings.N, settings.M, settings.dx, settings.dy, x0=settings.x0)
    x = np.asarray(grid.x, dtype=float).reshape(-1)
    z = np.asarray(Z[0, :], dtype=float).reshape(-1)
    summary = summarize_wavelengths(settings, x, z)
    summary["artifacts"] = {"solver_csv": csv_path, "solver_figure": fig_path, "diagnostic_plot": save_diagnostic_plot(x, z, out / "wavelength_diagnostic_y0.png")}
    summary_path = out / "characteristic_wavelengths.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
