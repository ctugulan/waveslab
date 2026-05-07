from __future__ import annotations

"""
General steady-result plotting helpers.

This file keeps the manuscript-style 3D surface and slice formatting, but the
helpers are no longer bathy-specific. They can be used in two ways:

1) From ``run_cases.py`` after a solve, to save:
   - the flattened CSV,
   - a 3D surface PNG,
   - an x-slice PNG,
   - a y-slice PNG.

2) As a press-play post-processing script for an existing directory of CSVs.

The plotting style still relies on reusable utilities from
``viscice_demo_plots.py``.
"""

from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter


try:
    import pywave.waves_helpers.viscice_demo_plots as vdp  # type: ignore
except Exception:  # pragma: no cover - public-repo fallback
    import csv
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import ScalarFormatter

    class _VdpFallback:
        _PUB_RCPARAMS: dict = {}

        @staticmethod
        def _sci_formatter():
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 2))
            return fmt

        @staticmethod
        def _ice_cmap():
            colors = [
                (0.02, 0.03, 0.04),
                (0.13, 0.17, 0.24),
                (0.35, 0.43, 0.58),
                (0.70, 0.78, 0.90),
                (0.96, 0.97, 0.99),
            ]
            return LinearSegmentedColormap.from_list("fallback_ice", colors)

        @staticmethod
        def mirror_along_y0(X1d, Y1d, Z):
            X = np.asarray(X1d, dtype=float).reshape(-1)
            Y = np.asarray(Y1d, dtype=float).reshape(-1)
            A = np.asarray(Z, dtype=float)
            if Y.size and np.isclose(Y[0], 0.0):
                return X, np.concatenate([-Y[:0:-1], Y]), np.vstack([A[:0:-1], A])
            return X, np.concatenate([-Y[::-1], Y]), np.vstack([A[::-1], A])

        @staticmethod
        def _centerlines(X1d, Y1d, Z):
            X = np.asarray(X1d, dtype=float).reshape(-1)
            Y = np.asarray(Y1d, dtype=float).reshape(-1)
            A = np.asarray(Z, dtype=float)
            iy = int(np.argmin(np.abs(Y)))
            ix = int(np.argmin(np.abs(X)))
            return X, Y, A[iy, :], A[:, ix]

        @staticmethod
        def _centerlines_absmax(X1d, Y1d, Z):
            X = np.asarray(X1d, dtype=float).reshape(-1)
            Y = np.asarray(Y1d, dtype=float).reshape(-1)
            A = np.asarray(Z, dtype=float)
            iy, ix = np.unravel_index(np.nanargmax(np.abs(A)), A.shape)
            return X, Y, A[iy, :], A[:, ix], float(X[ix]), float(Y[iy])

        @staticmethod
        def load_csv_to_grid(csv_path):
            arr = np.genfromtxt(csv_path, delimiter=",", names=True)
            x = np.unique(arr["x"])
            y = np.unique(arr["y"])
            Z = arr["zeta"].reshape(len(y), len(x))
            return x, y, Z

    vdp = _VdpFallback()



# -----------------------------------------------------------------------------
# User-editable defaults for press-play post-processing
# -----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
OUT_BASE = _THIS_DIR / "out"
DEFAULT_RESULTS_DIR = OUT_BASE / "bathy_manuscript_results"
DEFAULT_FIGS_DIR = OUT_BASE / "bathy_manuscript_figs"

RESULTS_DIR: str = ""   # "" -> DEFAULT_RESULTS_DIR
FIGS_DIR: str = ""      # "" -> DEFAULT_FIGS_DIR
RESULTS_GLOB = "zeta_*.csv"

SLICE_KIND = "absmax"   # "absmax" or "origin"
MIRROR_Y0 = True         # show y in [-Ymax, Ymax] on the surface plot
DPI = 300
ONLY_STEM: str | None = None


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _autocrop_png_inplace(png_path: Path, *, bg_threshold: int = 252, pad_px: int = 2) -> None:
    """Trim top/bottom whitespace from a saved PNG while keeping full width."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return

    try:
        img = Image.open(png_path)
    except Exception:
        return

    arr = np.asarray(img)
    if arr.ndim < 2:
        return

    if arr.ndim == 2:
        mask = arr < bg_threshold
    else:
        rgb = arr[..., :3]
        mask = np.any(rgb < bg_threshold, axis=-1)
        if arr.shape[-1] == 4:
            mask &= (arr[..., 3] > 0)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return

    y0 = int(coords[:, 0].min())
    y1 = int(coords[:, 0].max())
    y0 = max(y0 - pad_px, 0)
    y1 = min(y1 + pad_px + 1, arr.shape[0])

    if (y1 - y0) <= 1:
        return

    img.crop((0, y0, arr.shape[1], y1)).save(png_path)


def _zlim_with_pad(Z: np.ndarray, pad_frac: float = 0.05) -> tuple[float, float]:
    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))
    span = (zmax - zmin) if zmax > zmin else 1.0
    pad = pad_frac * span
    return (zmin - pad, zmax + pad)


def _resolve_dir(path_text: str, default: Path) -> Path:
    return Path(path_text).expanduser().resolve() if str(path_text).strip() else default


def infer_figs_dir(results_dir: Path) -> Path:
    name = results_dir.name
    if name.endswith("_results"):
        return results_dir.with_name(name[:-8] + "_figs")
    return results_dir.parent / f"{name}_figs"


def case_plot_paths(*, stem: str, figs_dir: Path) -> dict[str, Path]:
    figs_dir = Path(figs_dir)
    return {
        "surface": figs_dir / f"{stem}__surface.png",
        "slice_x": figs_dir / f"{stem}__slice_x.png",
        "slice_y": figs_dir / f"{stem}__slice_y.png",
    }


# -----------------------------------------------------------------------------
# Core manuscript-style plotting primitives
# -----------------------------------------------------------------------------
def save_surface_png(
    *,
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
    out_png: Path,
    zlim: tuple[float, float],
    mirror_y0: bool = True,
) -> None:
    if mirror_y0:
        X1d, Y1d, Z = vdp.mirror_along_y0(X1d, Y1d, Z)

    X2d, Y2d = np.meshgrid(X1d, Y1d)

    fig = plt.figure(figsize=(12.5, 5.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_position([0.02, -0.02, 0.96, 1.04])

    ax.set_xlabel(r"$x$", fontsize=16, labelpad=10)
    ax.set_ylabel(r"$y$", fontsize=16, labelpad=10)
    ax.tick_params(axis="z", pad=0, labelsize=11)

    ax.set_zlabel("")
    zlab = ax.text2D(
        -0.060,
        0.54,
        r"$\zeta(x,y)$",
        transform=ax.transAxes,
        rotation=90,
        rotation_mode="anchor",
        va="center",
        ha="center",
        fontsize=16,
    )
    zlab.set_clip_on(False)

    ax.tick_params(which="major", width=1.2, labelsize=11, pad=2)
    ax.set_zlim(float(zlim[0]), float(zlim[1]))

    try:
        ax.set_box_aspect((1.0, 1.0, 0.45))
        ax.set_proj_type("ortho")
    except Exception:
        pass

    ax.view_init(elev=24, azim=60)
    ax.invert_xaxis()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass

    ice = vdp._ice_cmap()
    ls = LightSource(azdeg=315, altdeg=45)
    qlo, qhi = np.nanpercentile(Z, [1, 99])
    vlim = max(abs(float(qlo)), abs(float(qhi)), 1e-14)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
    facecols = ls.shade(Z, cmap=ice, norm=norm, vert_exag=0.8, blend_mode="soft")
    facecols[..., 3] = 0.95

    ax.plot_surface(
        X2d,
        Y2d,
        Z,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        edgecolor="none",
        facecolors=facecols,
        shade=False,
        antialiased=True,
    )

    M, N = Z.shape
    wire_density = 24
    ax.plot_wireframe(
        X2d,
        Y2d,
        Z,
        rstride=max(M // wire_density, 1),
        cstride=max(N // wire_density, 1),
        linewidth=0.35,
        color=(0.20, 0.20, 0.20, 0.12),
    )

    zmaxabs = max(abs(float(zlim[0])), abs(float(zlim[1])), 1e-300)
    exp = int(np.floor(np.log10(zmaxabs)))
    scale = 10.0 ** exp
    ax.zaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v/scale:g}"))

    sci = None
    if exp != 0:
        sci = ax.text2D(
            0.060,
            0.70,
            rf"$\times 10^{{{exp}}}$",
            transform=ax.transAxes,
            rotation=0,
            va="bottom",
            ha="left",
            fontsize=14,
        )
    if sci is not None:
        sci.set_clip_on(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    _autocrop_png_inplace(out_png)
    plt.close(fig)
    print(f"[save] {out_png}")


def _slice_lines(
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
    slice_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    if slice_kind == "origin":
        X_line, Y_line, zx, zy = vdp._centerlines(X1d, Y1d, Z)
        ylab_x = r"$\zeta(x,0)$"
        ylab_y = r"$\zeta(0,y)$"
        return X_line, Y_line, zx, zy, ylab_x, ylab_y

    if slice_kind == "absmax":
        X_line, Y_line, zx, zy, _xstar, _ystar = vdp._centerlines_absmax(X1d, Y1d, Z)
        ylab_x = r"$\zeta(x, y_{\max})$"
        ylab_y = r"$\zeta(x_{\max}, y)$"
        return X_line, Y_line, zx, zy, ylab_x, ylab_y

    raise ValueError("slice_kind must be 'origin' or 'absmax'")


def save_slice_png(
    *,
    s: np.ndarray,
    z: np.ndarray,
    xlabel: str,
    ylabel: str,
    out_png: Path,
    zlim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.plot(s, z, lw=2.2, color="black")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(float(zlim[0]), float(zlim[1]))
    ax.grid(False)
    ax.yaxis.set_major_formatter(vdp._sci_formatter())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.10)
    _autocrop_png_inplace(out_png)
    plt.close(fig)
    print(f"[save] {out_png}")


# -----------------------------------------------------------------------------
# Public helper API used by run_cases.py
# -----------------------------------------------------------------------------
def save_case_plots(
    *,
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
    stem: str,
    figs_dir: Path,
    slice_kind: str = "absmax",
    mirror_y0: bool = True,
    zlim: tuple[float, float] | None = None,
) -> dict[str, Path]:
    """Save manuscript-style surface + x/y slice plots for one steady case."""
    X1d = np.asarray(X1d).reshape(-1)
    Y1d = np.asarray(Y1d).reshape(-1)
    Z = np.asarray(Z)

    if zlim is None:
        zlim = _zlim_with_pad(Z, pad_frac=0.05)

    X_line, Y_line, zx, zy, ylab_x, ylab_y = _slice_lines(X1d, Y1d, Z, slice_kind)
    paths = case_plot_paths(stem=stem, figs_dir=Path(figs_dir))

    rc = getattr(vdp, "_PUB_RCPARAMS", None)
    ctx = plt.rc_context(rc) if isinstance(rc, dict) else nullcontext()
    with ctx:
        save_surface_png(
            X1d=X1d,
            Y1d=Y1d,
            Z=Z,
            out_png=paths["surface"],
            zlim=zlim,
            mirror_y0=mirror_y0,
        )
        save_slice_png(
            s=X_line,
            z=zx,
            xlabel=r"$x$",
            ylabel=ylab_x,
            out_png=paths["slice_x"],
            zlim=zlim,
        )
        save_slice_png(
            s=Y_line,
            z=zy,
            xlabel=r"$y$",
            ylabel=ylab_y,
            out_png=paths["slice_y"],
            zlim=zlim,
        )

    return paths


def save_case_outputs(
    *,
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
    csv_path: Path | None = None,
    figs_dir: Path | None = None,
    stem: str | None = None,
    slice_kind: str = "absmax",
    mirror_y0: bool = True,
    save_csv: bool = True,
) -> dict[str, Path]:
    """Optionally save the steady CSV, then save the surface + slice plots.

    Parameters
    ----------
    csv_path
        Path for the flattened steady CSV. If provided and ``save_csv`` is True,
        ``Z`` is saved as a single-column file matching the current steady
        runner's convention.
    figs_dir
        Directory for PNG outputs. If omitted and ``csv_path`` is provided, the
        PNGs are placed beside the CSV.
    stem
        Basename used for the PNGs. Defaults to ``csv_path.stem``.
    """
    if stem is None:
        if csv_path is None:
            raise ValueError("Provide either stem or csv_path.")
        stem = Path(csv_path).stem

    if figs_dir is None:
        if csv_path is None:
            raise ValueError("Provide either figs_dir or csv_path.")
        figs_dir = Path(csv_path).parent

    out: dict[str, Path] = {}
    if csv_path is not None:
        csv_path = Path(csv_path)
        out["csv"] = csv_path
        if save_csv:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(csv_path, np.asarray(Z).reshape(-1, 1), delimiter=",")
            print(f"[save] {csv_path}")

    out.update(
        save_case_plots(
            X1d=X1d,
            Y1d=Y1d,
            Z=Z,
            stem=stem,
            figs_dir=Path(figs_dir),
            slice_kind=slice_kind,
            mirror_y0=mirror_y0,
        )
    )
    return out


def render_existing_csv_results(
    *,
    results_dir: Path,
    figs_dir: Path,
    only_stem: str | None = None,
    pattern: str = "zeta_*.csv",
    slice_kind: str = "absmax",
    mirror_y0: bool = True,
) -> list[dict[str, Path]]:
    """Render plots for every matching CSV in an existing results directory."""
    results_dir = Path(results_dir)
    figs_dir = Path(figs_dir)

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    csvs = sorted(results_dir.glob(pattern))
    if only_stem is not None:
        csvs = [p for p in csvs if p.stem == only_stem]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in: {results_dir}")

    saved: list[dict[str, Path]] = []
    for csv_path in csvs:
        X1d, Y1d, Z = vdp.load_csv_to_grid(csv_path)
        paths = save_case_outputs(
            X1d=X1d,
            Y1d=Y1d,
            Z=Z,
            csv_path=csv_path,
            figs_dir=figs_dir,
            stem=csv_path.stem,
            slice_kind=slice_kind,
            mirror_y0=mirror_y0,
            save_csv=False,
        )
        saved.append(paths)
    return saved


# -----------------------------------------------------------------------------
# Press-play entry point
# -----------------------------------------------------------------------------
def main() -> None:
    results_dir = _resolve_dir(RESULTS_DIR, DEFAULT_RESULTS_DIR)
    figs_dir = _resolve_dir(FIGS_DIR, infer_figs_dir(results_dir) if not str(FIGS_DIR).strip() else DEFAULT_FIGS_DIR)
    render_existing_csv_results(
        results_dir=results_dir,
        figs_dir=figs_dir,
        only_stem=ONLY_STEM,
        pattern=RESULTS_GLOB,
        slice_kind=SLICE_KIND,
        mirror_y0=MIRROR_Y0,
    )


if __name__ == "__main__":
    main()