from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter


def _extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def save_cover_map(F: np.ndarray, x: np.ndarray, y: np.ndarray, out_png: Path, *, label: str | None = None) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.7), constrained_layout=True)
    im = ax.imshow(
        np.asarray(F, dtype=float),
        origin="lower",
        extent=_extent(x, y),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    if label:
        ax.set_title(label)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(r"cover $F$")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_png


def save_delta_map(delta: np.ndarray, x: np.ndarray, y: np.ndarray, out_png: Path, *, label: str | None = None) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    D = np.asarray(delta, dtype=float)
    vmax = max(abs(float(np.nanmin(D))), abs(float(np.nanmax(D))), 1e-14)
    fig, ax = plt.subplots(figsize=(6.2, 3.7), constrained_layout=True)
    im = ax.imshow(
        D,
        origin="lower",
        extent=_extent(x, y),
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    if label:
        ax.set_title(label)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid(False)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, format=formatter)
    cbar.set_label(r"$\Delta\zeta$")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_png


def save_centerline_overlay(cases: list[dict], out_png: Path, *, label: str | None = None) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.5), constrained_layout=True)
    for case in cases:
        x = np.asarray(case["grid_x"], dtype=float).reshape(-1)
        y = np.asarray(case["grid_y"], dtype=float).reshape(-1)
        Z = np.asarray(case["Z"], dtype=float)
        iy = int(np.argmin(np.abs(y)))
        ax.plot(x, Z[iy, :], lw=1.8, label=str(case.get("label", "case")))
    ax.axhline(0.0, lw=0.8, alpha=0.5)
    if label:
        ax.set_title(label)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\zeta(x,0)$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_png


def save_surface_with_cover(
    *,
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    F: np.ndarray,
    out_png: Path,
    cover_threshold: float = 0.5,
) -> Path:
    """Save a simple 3D surface with the cover projected beneath it."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float)
    F = np.asarray(F, dtype=float)
    X, Y = np.meshgrid(x, y)

    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))
    span = max(zmax - zmin, 1e-14)
    z0 = zmin - 0.08 * span
    vlim = max(abs(zmin), abs(zmax), 1e-14)

    fig = plt.figure(figsize=(8.0, 4.8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
    facecols = LightSource(azdeg=315, altdeg=45).shade(Z, cmap=plt.get_cmap("bone"), norm=norm)
    ax.plot_surface(X, Y, Z, facecolors=facecols, linewidth=0.0, shade=False, antialiased=True)

    cover = np.asarray(F >= float(cover_threshold), dtype=float)
    ax.plot_surface(X, Y, np.full_like(Z, z0), facecolors=plt.get_cmap("viridis")(cover), linewidth=0.0, shade=False, alpha=0.75)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$\zeta$")
    ax.view_init(elev=24, azim=60)
    try:
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1.0, 0.75, 0.35))
    except Exception:
        pass
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_png
