#!/usr/bin/env python3
"""
Save the six red viscoelastic phase-line plots for the flatbed regimes.

This version has no command-line options. To use it:

    1. Put this file anywhere convenient.
    2. Edit the settings in the USER SETTINGS section below if needed.
    3. Run:

        python viscoelastic_phase_lines_simple.py

The six PNG files are saved in OUT_DIR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

# Folder where the six plots will be saved.
OUT_DIR = Path("phase_lines_viscoelastic")

# Viscoelastic parameters for the current flatbed U sweep.
BETA = 0.00748248
TAU = 0.12004529

# Number of phase-line families to draw: a = 1, 2, ..., A_MAX.
A_MAX = 15

# Plot window. Use the tighter window below if you want a cropped view.
XLIM = (-4.0, 20.0)
YLIM = (-20.0, 20.0)

# For a tighter cropped view, uncomment these two lines:
# XLIM = (-2.0, 10.0)
# YLIM = (-8.0, 8.0)

# Plot appearance.
PHASE_LINE_COLOR = "red"
LINE_WIDTH = 1.25
LINE_ALPHA = 0.90
DPI = 300
TRANSPARENT_BACKGROUND = False
SHOW_AXES = False
ADD_LABEL = False

# Numerical resolution for sampling the phase-line curves.
N_K = 3000
K_MIN = 1.0e-4
DENOM_TOL = 1.0e-7
SQRT_TOL = 0.0


# =============================================================================
# SIX FLATBED REGIMES
# =============================================================================

REGIMES: list[tuple[float, float, str, str]] = [
    # U, Fr, regime label, file tag
    (5.0, 0.6125, "A", "A_050"),
    (6.0, 0.7350, "B1", "B1_060"),
    (7.4, 0.9065, "B2", "B2_074"),
    (8.4, 1.0290, "C1", "C1_084"),
    (10.2, 1.2495, "C2", "C2_102"),
    (10.5, 1.2862, "D", "D_105"),
]


@dataclass(frozen=True)
class Params:
    c0: float
    beta: float
    tau: float
    denom_tol: float = DENOM_TOL
    sqrt_tol: float = SQRT_TOL


def c_hat_sq_zhestkaya_viscoelastic(
    kH: np.ndarray,
    *,
    beta: float,
    tau: float,
) -> np.ndarray:
    """Dimensionless viscoelastic phase-speed squared."""
    th = np.tanh(kH)
    return (
        -0.25 * beta**2 * tau**2 * kH**8 * th**2
        + ((1.0 / kH) + beta * kH**3) * th
    )


def contiguous_slices(mask: np.ndarray) -> list[slice]:
    """Return contiguous True regions of a Boolean mask as slices."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [slice(int(s), int(e) + 1) for s, e in zip(starts, ends)]


def find_k_cutoff(beta: float, tau: float) -> float:
    """Find the largest k value with positive phase-speed squared."""
    k_test = np.linspace(1.0e-4, 50.0, 25_000)
    sq = c_hat_sq_zhestkaya_viscoelastic(k_test, beta=beta, tau=tau)
    pos = np.where(sq > 0.0)[0]
    if pos.size == 0:
        return 0.0

    k_max_pos = float(k_test[pos[-1]])

    lo = max(1.0e-4, k_max_pos - 0.25)
    hi = k_max_pos + 0.25
    k_ref = np.linspace(lo, hi, 5_000)
    sq_ref = c_hat_sq_zhestkaya_viscoelastic(k_ref, beta=beta, tau=tau)
    pos_ref = np.where(sq_ref > 0.0)[0]
    if pos_ref.size == 0:
        return k_max_pos

    return float(k_ref[pos_ref[-1]])


def compute_phase_lines_for_U(
    *,
    params: Params,
    U: float,
    a_values: Iterable[float],
    k_min: float,
    k_max: float,
    n_k: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute the upper-half phase-line family for one value of U."""
    Fr = float(U) / float(params.c0)
    if Fr <= 0.0:
        raise ValueError("U and Fr must imply a positive Froude number.")

    k = np.linspace(float(k_min), float(k_max), int(n_k), dtype=float)

    beta = float(params.beta)
    tau = float(params.tau)
    Fr2 = Fr**2
    t = np.tanh(k)
    sech2 = 1.0 - t**2

    Q = (
        (-beta**2 * tau**2 / (4.0 * Fr2)) * k**10 * t**2
        + (1.0 / Fr2) * (1.0 + beta * k**4) * k * t
    )
    dQ = (
        (-beta**2 * tau**2 / (4.0 * Fr2))
        * (10.0 * k**9 * t**2 + k**10 * 2.0 * sech2 * t)
        + (1.0 / Fr2)
        * ((4.0 * beta * k**3) * k * t + (1.0 + beta * k**4) * (t + k * sech2))
    )

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        F = np.where(Q > 0.0, np.sqrt(Q), np.nan)
        Fp = np.where(np.isfinite(F) & (F > 0.0), 0.5 * dQ / F, np.nan)

    sqrt_arg = k**2 - Q
    valid = np.isfinite(F) & np.isfinite(Fp) & (sqrt_arg >= -float(params.sqrt_tol))

    base_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for sl in contiguous_slices(valid):
        kk = k[sl]
        FF = F[sl]
        if kk.size < 30:
            continue

        FFp = Fp[sl]
        denom = kk * (FF - kk * FFp)
        sa = np.sqrt(np.maximum(0.0, kk**2 - FF**2))

        ok = (
            np.isfinite(FFp)
            & np.isfinite(denom)
            & (np.abs(denom) > float(params.denom_tol))
            & np.isfinite(sa)
        )
        if np.count_nonzero(ok) < 30:
            continue

        kk2 = kk[ok]
        FF2 = FF[ok]
        FFp2 = FFp[ok]
        denom2 = kk2 * (FF2 - kk2 * FFp2)

        x1 = (kk2 - FF2 * FFp2) / denom2
        y1 = -FFp2 / denom2 * np.sqrt(np.maximum(0.0, kk2**2 - FF2**2))

        flip = (FF2 - kk2 * FFp2) < 0.0
        x1 = np.where(flip, -x1, x1)

        good = np.isfinite(x1) & np.isfinite(y1)
        if np.count_nonzero(good) >= 30:
            base_segments.append((x1[good], y1[good]))

    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for a in a_values:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        for i, (x1, y1) in enumerate(base_segments):
            if i > 0:
                xs.append(np.array([np.nan]))
                ys.append(np.array([np.nan]))
            xs.append(float(a) * x1)
            ys.append(float(a) * y1)

        if xs:
            curves.append((np.concatenate(xs), np.concatenate(ys)))
        else:
            curves.append((np.array([]), np.array([])))

    return curves


def clip_polyline_to_box(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Hide curve points outside the plotting window."""
    x2 = np.asarray(x, float).copy()
    y2 = np.asarray(y, float).copy()

    finite = np.isfinite(x2) & np.isfinite(y2)
    inside = finite & (x2 >= xlim[0]) & (x2 <= xlim[1]) & (y2 >= ylim[0]) & (y2 <= ylim[1])
    x2[~inside] = np.nan
    y2[~inside] = np.nan
    return x2, y2


def clean_axes(
    ax: plt.Axes,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Set the plot window and optionally show simple axes."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    if SHOW_AXES:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.axhline(0.0, color="0.65", linewidth=0.8, linestyle="--", zorder=0)
    else:
        ax.set_axis_off()


def draw_phase_lines(
    ax: plt.Axes,
    *,
    curves: list[tuple[np.ndarray, np.ndarray]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Draw upper and lower branches of the phase-line family."""
    for x, y in curves:
        if x.size == 0:
            continue

        xp, yp = clip_polyline_to_box(x, y, xlim=xlim, ylim=ylim)
        ax.plot(
            xp,
            yp,
            color=PHASE_LINE_COLOR,
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            solid_capstyle="round",
        )
        ax.plot(
            xp,
            -yp,
            color=PHASE_LINE_COLOR,
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            solid_capstyle="round",
        )


def save_one_regime(*, U: float, Fr: float, regime_tag: str) -> Path:
    """Save one PNG for one regime."""
    c0 = float(U) / float(Fr)
    params = Params(c0=c0, beta=float(BETA), tau=float(TAU))

    k_cut = find_k_cutoff(beta=float(BETA), tau=float(TAU))
    if k_cut <= 0.0:
        raise RuntimeError(f"No valid phase lines for BETA={BETA}, TAU={TAU}.")

    curves = compute_phase_lines_for_U(
        params=params,
        U=float(U),
        a_values=range(1, int(A_MAX) + 1),
        k_min=K_MIN,
        k_max=min(k_cut + 0.02, 100.0),
        n_k=int(N_K),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"phase_lines_visco_{regime_tag}.png"

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    draw_phase_lines(ax, curves=curves, xlim=XLIM, ylim=YLIM)
    clean_axes(ax, xlim=XLIM, ylim=YLIM)

    if ADD_LABEL:
        ax.text(
            0.02,
            0.96,
            rf"{regime_tag}: $U={U:g}$, $\mathrm{{Fr}}={Fr:.4f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=PHASE_LINE_COLOR,
        )

    fig.savefig(
        out_png,
        dpi=int(DPI),
        transparent=TRANSPARENT_BACKGROUND,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return out_png


def main() -> None:
    print(f"Saving plots to: {OUT_DIR.resolve()}")
    print(f"Parameters: BETA={BETA:g}, TAU={TAU:g}, a=1..{A_MAX}")
    print(f"Plot window: xlim={XLIM}, ylim={YLIM}")

    saved_files: list[Path] = []
    for U, Fr, _regime, regime_tag in REGIMES:
        saved_files.append(save_one_regime(U=U, Fr=Fr, regime_tag=regime_tag))

    print("\nSaved files:")
    for path in saved_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()