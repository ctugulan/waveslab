from __future__ import annotations

"""
Publication-style rendering utilities for flexural cover runs.

This module is the package version of ``pub_render_surface_direct_cover_full.py``.
It intentionally focuses on final publication figures rather than maintaining
separate quicklook/publication renderers.  The runner passes the solved arrays
in directly, while this helper handles full-domain versus y=0 mirrored plotting,
field orientation checks, the manuscript-style 3D view, sunset-style cover
projection, and multi-case cross-section comparisons.
"""

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

try:  # keep this helper usable from tests/docs even if pywave is not fully importable
    import waveslab._missing_old_plot_style as vdp
except Exception:  # pragma: no cover - fallback only
    vdp = None  # type: ignore[assignment]


DomainMode = Literal["full", "mirror_y0"]

DEFAULT_DOMAIN_MODE: DomainMode = "full"
DEFAULT_COVER_THRESHOLD = 0.5
DEFAULT_COVER_ALPHA = 0.98
DEFAULT_PAD_FRAC = 0.05
DEFAULT_WIRE_DENSITY = 24
DEFAULT_DPI = 300

# Binary cover plane colours.  The cover mask is thresholded into B where
# B=0 is open water/gaps and B=1 is ice/cover.  This palette follows the
# reference sunset photographs: warm reflected water and dark teal ice.
DEFAULT_COVER_CMAP_COLORS: tuple[str, str] = ( "#006582","#e8c7d8")

# Sunset-inspired, publication-safe curve colours for multi-case slices.
# These are deliberately not the same colours as the binary cover plane so
# that line plots remain readable in print and in small thesis panels.
DEFAULT_CROSS_SECTION_COLORS: tuple[str, ...] = (
    "#4C698A",  # homogeneous: muted slate blue
    "#D9895B",  # direct mask: warm apricot
    "#8C5A7A",  # ellipse approximation: muted dusk plum
    "#3A6B5F",  # spare: muted teal
    "#6F5B8C",  # spare: muted violet
)
DEFAULT_CROSS_SECTION_LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")


def _fallback_ice_cmap() -> LinearSegmentedColormap:
    """Return a local fallback matching the dark-to-light ice palette."""
    try:
        return plt.get_cmap("ice")  # type: ignore[return-value]
    except Exception:
        cols = np.array(
            [
                [0, 0, 0],
                [11, 13, 17],
                [20, 23, 32],
                [28, 33, 45],
                [38, 43, 59],
                [48, 55, 74],
                [61, 70, 92],
                [74, 84, 111],
                [88, 100, 131],
                [102, 116, 150],
                [118, 133, 169],
                [135, 150, 190],
                [153, 169, 208],
                [174, 188, 224],
                [197, 209, 237],
                [221, 229, 247],
                [246, 248, 252],
            ],
            dtype=float,
        ) / 255.0
        return LinearSegmentedColormap.from_list("fallback_ice", cols)


def _ice_cmap():
    if vdp is not None and hasattr(vdp, "_ice_cmap"):
        return vdp._ice_cmap()
    return _fallback_ice_cmap()


def _pub_rcparams() -> dict:
    if vdp is not None:
        return getattr(vdp, "_PUB_RCPARAMS", {})
    return {}


def _autocrop_png_inplace(png_path: Path, *, bg_threshold: int = 252, pad_px: int = 2) -> None:
    """Tightly crop vertical whitespace without changing the horizontal extent."""
    try:
        from PIL import Image
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
            mask &= arr[..., 3] > 0

    coords = np.argwhere(mask)
    if coords.size == 0:
        return

    y0 = max(int(coords[:, 0].min()) - pad_px, 0)
    y1 = min(int(coords[:, 0].max()) + pad_px + 1, arr.shape[0])
    if (y1 - y0) > 1:
        img.crop((0, y0, arr.shape[1], y1)).save(png_path)


def _zlim_with_pad(Z: np.ndarray, pad_frac: float = DEFAULT_PAD_FRAC) -> tuple[float, float]:
    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))
    span = (zmax - zmin) if zmax > zmin else 1.0
    pad = float(pad_frac) * span
    return zmin - pad, zmax + pad


def _line_zlim_with_pad(lines: Sequence[np.ndarray], pad_frac: float = DEFAULT_PAD_FRAC) -> tuple[float, float]:
    """Return a shared y-limit for one or more 1D cross-section curves."""
    finite_chunks = []
    for line in lines:
        arr = np.asarray(line, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_chunks.append(arr)
    if not finite_chunks:
        return -1.0, 1.0
    vals = np.concatenate(finite_chunks)
    zmin = float(np.nanmin(vals))
    zmax = float(np.nanmax(vals))
    span = (zmax - zmin) if zmax > zmin else max(abs(zmin), abs(zmax), 1.0)
    pad = float(pad_frac) * span
    return zmin - pad, zmax + pad


def _sci_formatter():
    """Use the same scientific tick formatter as plot_steady_all.py when available."""
    if vdp is not None and hasattr(vdp, "_sci_formatter"):
        return vdp._sci_formatter()
    return FuncFormatter(lambda value, _pos: f"{value:g}")


def _nearest_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values, dtype=float).reshape(-1)
    return int(np.nanargmin(np.abs(values - float(target))))


def _fallback_centerlines(
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return zeta(x,0) and zeta(0,y) without requiring viscice_demo_plots."""
    ix0 = _nearest_index(X1d, 0.0)
    iy0 = _nearest_index(Y1d, 0.0)
    return X1d, Y1d, Z[iy0, :], Z[:, ix0]


def _fallback_centerlines_absmax(
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return zeta(x,y*) and zeta(x*,y), where |zeta| is largest at (x*, y*)."""
    iy, ix = np.unravel_index(int(np.nanargmax(np.abs(Z))), Z.shape)
    return X1d, Y1d, Z[iy, :], Z[:, ix], float(X1d[ix]), float(Y1d[iy])


def normalize_domain_mode(domain_mode: str | None) -> DomainMode:
    """Normalize user-facing domain aliases to the two supported modes."""
    mode = str(domain_mode or DEFAULT_DOMAIN_MODE).strip().lower().replace("-", "_")
    aliases = {
        "native": "full",
        "saved": "full",
        "saved_full": "full",
        "full_y": "full",
        "full_domain": "full",
        "full": "full",
        "mirror": "mirror_y0",
        "mirrored": "mirror_y0",
        "symmetric": "mirror_y0",
        "sym": "mirror_y0",
        "mirror_y": "mirror_y0",
        "mirror_y0": "mirror_y0",
        "half_to_full": "mirror_y0",
    }
    if mode not in aliases:
        raise ValueError("Unknown domain_mode={!r}. Expected 'full' or 'mirror_y0'.".format(domain_mode))
    return aliases[mode]  # type: ignore[return-value]


def _orient_field_to_grid(
    A: np.ndarray,
    *,
    X1d: np.ndarray,
    Y1d: np.ndarray,
    name: str,
) -> np.ndarray:
    """Return ``A`` with shape ``(len(Y1d), len(X1d))``, transposing if needed."""
    A = np.asarray(A)
    expected = (int(Y1d.size), int(X1d.size))
    if A.shape == expected:
        return A
    if A.ndim == 2 and A.T.shape == expected:
        print(f"[info] transposing {name} from {A.shape} to {expected} to match grid")
        return A.T
    raise ValueError(
        f"{name} has shape {A.shape}, but grid expects {expected} "
        f"=(len(Y1d), len(X1d)). Check the saved field orientation."
    )


def _mirror_z_along_y0(X1d: np.ndarray, Y1d: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror a half-y field across y=0, avoiding a duplicated y=0 row."""
    if vdp is not None and hasattr(vdp, "mirror_along_y0"):
        return vdp.mirror_along_y0(X1d, Y1d, Z)

    X1d = np.asarray(X1d, dtype=float).reshape(-1)
    Y1d = np.asarray(Y1d, dtype=float).reshape(-1)
    Z = np.asarray(Z)
    if Y1d.size and np.isclose(Y1d[0], 0.0):
        Yneg = -Y1d[1:][::-1]
        Zneg = Z[1:, :][::-1, :]
    else:
        Yneg = -Y1d[::-1]
        Zneg = Z[::-1, :]
    return X1d, np.concatenate([Yneg, Y1d]), np.vstack([Zneg, Z])


def _mirror_cover_along_y0(F: np.ndarray, Y1d: np.ndarray) -> np.ndarray:
    """Mirror a half-y cover field across y=0, avoiding a duplicated y=0 row."""
    F = np.asarray(F, dtype=float)
    Y1d = np.asarray(Y1d, dtype=float).reshape(-1)
    if Y1d.size and np.isclose(Y1d[0], 0.0):
        Fneg = F[1:, :][::-1, :]
    else:
        Fneg = F[::-1, :]
    return np.vstack([Fneg, F])


def prepare_surface_cover_domain(
    *,
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
    cover: np.ndarray,
    domain_mode: str = DEFAULT_DOMAIN_MODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare x, y, zeta and cover arrays for native or mirrored plotting."""
    mode = normalize_domain_mode(domain_mode)
    X1d = np.asarray(X1d, dtype=float).reshape(-1)
    Y1d = np.asarray(Y1d, dtype=float).reshape(-1)

    Z = _orient_field_to_grid(Z, X1d=X1d, Y1d=Y1d, name="zeta")
    cover = _orient_field_to_grid(cover, X1d=X1d, Y1d=Y1d, name="cover")

    if mode == "full":
        return X1d, Y1d, Z, cover

    if np.nanmin(Y1d) < 0.0 and np.nanmax(Y1d) > 0.0:
        raise ValueError(
            "domain_mode='mirror_y0' expects a half-y grid, but this run already "
            "contains both negative and positive y values. Use domain_mode='full'."
        )

    Xplot, Yplot, Zplot = _mirror_z_along_y0(X1d, Y1d, Z)
    Fplot = _mirror_cover_along_y0(cover, Y1d)
    Fplot = _orient_field_to_grid(Fplot, X1d=Xplot, Y1d=Yplot, name="mirrored cover")
    return Xplot, Yplot, Zplot, Fplot


def prepare_surface_domain(
    *,
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
    domain_mode: str = DEFAULT_DOMAIN_MODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare x, y and zeta arrays for native or mirrored plotting."""
    mode = normalize_domain_mode(domain_mode)
    X1d = np.asarray(X1d, dtype=float).reshape(-1)
    Y1d = np.asarray(Y1d, dtype=float).reshape(-1)
    Z = _orient_field_to_grid(Z, X1d=X1d, Y1d=Y1d, name="zeta")

    if mode == "full":
        return X1d, Y1d, Z

    if np.nanmin(Y1d) < 0.0 and np.nanmax(Y1d) > 0.0:
        raise ValueError(
            "domain_mode='mirror_y0' expects a half-y grid, but this run already "
            "contains both negative and positive y values. Use domain_mode='full'."
        )

    return _mirror_z_along_y0(X1d, Y1d, Z)


def _slice_lines(
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
    slice_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    """
    Return the x- and y-direction cross-sections from an already oriented field.

    This intentionally does not delegate to ``viscice_demo_plots._centerlines``.
    The older helper can silently assume a different symmetry/orientation
    convention, which is what produced misleading nearly straight x=0 slices in
    some full-domain cover comparisons.
    """
    X1d = np.asarray(X1d, dtype=float).reshape(-1)
    Y1d = np.asarray(Y1d, dtype=float).reshape(-1)
    Z = _orient_field_to_grid(np.asarray(Z, dtype=float), X1d=X1d, Y1d=Y1d, name="slice zeta")

    kind = str(slice_kind).strip().lower().replace("-", "_")
    if kind in {"origin", "center", "centre", "y0", "x0"}:
        ix0 = _nearest_index(X1d, 0.0)
        iy0 = _nearest_index(Y1d, 0.0)
        zx = np.asarray(Z[iy0, :], dtype=float)
        zy = np.asarray(Z[:, ix0], dtype=float)
        return X1d, Y1d, zx, zy, r"$\zeta(x,0)$", r"$\zeta(0,y)$"

    if kind in {"absmax", "max", "maximum", "max_abs"}:
        iy, ix = np.unravel_index(int(np.nanargmax(np.abs(Z))), Z.shape)
        zx = np.asarray(Z[iy, :], dtype=float)
        zy = np.asarray(Z[:, ix], dtype=float)
        return X1d, Y1d, zx, zy, r"$\zeta(x, y_{\max})$", r"$\zeta(x_{\max}, y)$"

    raise ValueError("slice_kind must be 'origin' or 'absmax'.")

def _display_label(raw: Any) -> str:
    label = str(raw).strip()
    aliases = {
        "homogeneous": "homogeneous",
        "homogeneous_direct_mean": "homogeneous",
        "homogeneous_ellipse_mean": "homogeneous",
        "direct": "direct mask",
        "direct_native_mask": "direct mask",
        "direct_lambda_flex_matched": r"direct mask, $\lambda_f$ matched",
        "ellipse": "ellipse approximation",
        "ellipse_native_mask": "ellipse approximation",
        "ellipse_lambda_flex_matched": r"ellipse, $\lambda_f$ matched",
    }
    return aliases.get(label, label.replace("_", " "))


def _curve_color(label: str, index: int, curve_colors: Sequence[str] | None = None) -> str:
    """
    Return a stable colour for a cross-section curve.

    Label-based colours make the plots robust to case ordering.  The optional
    ``curve_colors`` sequence is still available for quick experiments.
    """
    label_norm = str(label).strip().lower()
    by_label = {
        "homogeneous": DEFAULT_CROSS_SECTION_COLORS[0],
        "direct mask": DEFAULT_CROSS_SECTION_COLORS[1],
        "direct": DEFAULT_CROSS_SECTION_COLORS[1],
        "ellipse approximation": DEFAULT_CROSS_SECTION_COLORS[2],
        "ellipse": DEFAULT_CROSS_SECTION_COLORS[2],
    }
    if label_norm in by_label:
        return by_label[label_norm]

    colors = tuple(curve_colors or DEFAULT_CROSS_SECTION_COLORS)
    if not colors:
        colors = DEFAULT_CROSS_SECTION_COLORS
    return colors[index % len(colors)]


def _save_comparison_slice_png(
    *,
    curves: Sequence[tuple[np.ndarray, np.ndarray, str]],
    xlabel: str,
    ylabel: str,
    out_png: Path,
    zlim: tuple[float, float],
    dpi: int = DEFAULT_DPI,
    autocrop: bool = True,
    show_legend: bool = True,
    curve_colors: Sequence[str] | None = None,
    curve_linestyles: Sequence[str] | None = None,
) -> Path:
    """Save one multi-case 1D comparison plot with plot_steady_all styling."""
    out_png = Path(out_png)
    with plt.rc_context(_pub_rcparams()):
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        linestyles = tuple(curve_linestyles or DEFAULT_CROSS_SECTION_LINESTYLES)
        if not linestyles:
            linestyles = DEFAULT_CROSS_SECTION_LINESTYLES

        for i, (s, z, label) in enumerate(curves):
            ax.plot(
                np.asarray(s, dtype=float),
                np.asarray(z, dtype=float),
                lw=2.2,
                color=_curve_color(label, i, curve_colors),
                linestyle=linestyles[i % len(linestyles)],
                label=label if show_legend else None,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(float(zlim[0]), float(zlim[1]))
        ax.grid(False)
        ax.yaxis.set_major_formatter(_sci_formatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_legend:
            ax.legend(frameon=False, fontsize=9, loc="best")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.10)
        plt.close(fig)

    if autocrop:
        _autocrop_png_inplace(out_png)
    print(f"[save:cross-section] {out_png}")
    return out_png


def save_publication_cross_section_comparison(
    *,
    cases: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    stem: str = "flexural_cover_cases",
    domain_mode: str = DEFAULT_DOMAIN_MODE,
    slice_kind: str = "origin",
    pad_frac: float = DEFAULT_PAD_FRAC,
    dpi: int = DEFAULT_DPI,
    autocrop: bool = True,
    show_legend: bool = True,
    curve_colors: Sequence[str] | None = None,
    curve_linestyles: Sequence[str] | None = None,
) -> dict[str, Path]:
    """
    Save publication-style x/y cross-section comparisons for multiple cover runs.

    Each case mapping must provide ``X1d``, ``Y1d`` and ``Z``.  A case label can
    be supplied as ``label`` or ``selected_cover_case``.  The function uses the
    same centerline/absmax conventions and formatting as ``plot_steady_all.py``.
    """
    if len(cases) < 2:
        raise ValueError("At least two cases are required for a comparison plot.")

    mode = normalize_domain_mode(domain_mode)
    x_curves: list[tuple[np.ndarray, np.ndarray, str]] = []
    y_curves: list[tuple[np.ndarray, np.ndarray, str]] = []
    x_lines_for_limits: list[np.ndarray] = []
    y_lines_for_limits: list[np.ndarray] = []
    ylab_x = r"$\zeta(x,0)$"
    ylab_y = r"$\zeta(0,y)$"

    for case in cases:
        label = _display_label(case.get("label", case.get("selected_cover_case", case.get("run_tag", "case"))))
        Xplot, Yplot, Zplot = prepare_surface_domain(
            X1d=np.asarray(case["X1d"], dtype=float),
            Y1d=np.asarray(case["Y1d"], dtype=float),
            Z=np.asarray(case["Z"], dtype=float),
            domain_mode=mode,
        )
        X_line, Y_line, zx, zy, ylab_x, ylab_y = _slice_lines(Xplot, Yplot, Zplot, slice_kind)
        x_curves.append((np.asarray(X_line), np.asarray(zx), label))
        y_curves.append((np.asarray(Y_line), np.asarray(zy), label))
        x_lines_for_limits.append(np.asarray(zx))
        y_lines_for_limits.append(np.asarray(zy))

    out_dir = Path(out_dir)
    x_path = out_dir / f"{stem}__slice_x_compare.png"
    y_path = out_dir / f"{stem}__slice_y_compare.png"
    return {
        "slice_x_compare": _save_comparison_slice_png(
            curves=x_curves,
            xlabel=r"$x$",
            ylabel=ylab_x,
            out_png=x_path,
            zlim=_line_zlim_with_pad(x_lines_for_limits, pad_frac=pad_frac),
            dpi=dpi,
            autocrop=autocrop,
            show_legend=show_legend,
            curve_colors=curve_colors,
            curve_linestyles=curve_linestyles,
        ),
        "slice_y_compare": _save_comparison_slice_png(
            curves=y_curves,
            xlabel=r"$y$",
            ylabel=ylab_y,
            out_png=y_path,
            zlim=_line_zlim_with_pad(y_lines_for_limits, pad_frac=pad_frac),
            dpi=dpi,
            autocrop=autocrop,
            show_legend=show_legend,
            curve_colors=curve_colors,
            curve_linestyles=curve_linestyles,
        ),
    }


def save_publication_surface_with_cover(
    *,
    X1d: np.ndarray,
    Y1d: np.ndarray,
    Z: np.ndarray,
    cover: np.ndarray,
    out_png: str | Path,
    domain_mode: str = DEFAULT_DOMAIN_MODE,
    cover_threshold: float = DEFAULT_COVER_THRESHOLD,
    cover_alpha: float = DEFAULT_COVER_ALPHA,
    pad_frac: float = DEFAULT_PAD_FRAC,
    wire_density: int = DEFAULT_WIRE_DENSITY,
    dpi: int = DEFAULT_DPI,
    autocrop: bool = True,
    cover_colors: Sequence[str] = DEFAULT_COVER_CMAP_COLORS,
) -> Path:
    """
    Save the single publication-style 3D free-surface plot with cover projection.

    Parameters
    ----------
    domain_mode:
        ``"full"`` plots the saved/native y-domain. ``"mirror_y0"`` mirrors a
        half-y solution across y=0 for the older symmetric-view figure.
    cover_threshold:
        Threshold used to convert the cover fraction into the binary sunset-style
        water/ice cover plane shown beneath the surface.
    """
    mode = normalize_domain_mode(domain_mode)
    out_png = Path(out_png)

    Xplot, Yplot, Zplot, Fplot = prepare_surface_cover_domain(
        X1d=X1d,
        Y1d=Y1d,
        Z=Z,
        cover=cover,
        domain_mode=mode,
    )
    X2d, Y2d = np.meshgrid(Xplot, Yplot)
    zlim = _zlim_with_pad(Zplot, pad_frac=float(pad_frac))

    with plt.rc_context(_pub_rcparams()):
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

        ls = LightSource(azdeg=315, altdeg=45)
        qlo, qhi = np.nanpercentile(Zplot, [1, 99])
        vlim = max(abs(float(qlo)), abs(float(qhi)), 1e-14)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
        facecols = ls.shade(Zplot, cmap=_ice_cmap(), norm=norm, vert_exag=0.8, blend_mode="soft")
        facecols[..., 3] = 0.95

        ax.plot_surface(
            X2d,
            Y2d,
            Zplot,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            edgecolor="none",
            facecolors=facecols,
            shade=False,
            antialiased=True,
        )

        mm, nn = Zplot.shape
        ax.plot_wireframe(
            X2d,
            Y2d,
            Zplot,
            rstride=max(mm // int(wire_density), 1),
            cstride=max(nn // int(wire_density), 1),
            linewidth=0.35,
            color=(0.20, 0.20, 0.20, 0.12),
        )

        B = (np.asarray(Fplot, dtype=float) >= float(cover_threshold)).astype(int)
        zmin = float(np.nanmin(Zplot))
        zmax = float(np.nanmax(Zplot))
        dz = max(zmax - zmin, np.finfo(float).eps)
        z0 = zmin - 0.05 * dz

        cover_cmap = ListedColormap(list(cover_colors))
        facecolors_cover = cover_cmap(B)
        facecolors_cover[..., 3] = float(cover_alpha)
        ax.plot_surface(
            X2d,
            Y2d,
            np.full_like(Zplot, z0),
            facecolors=facecolors_cover,
            shade=False,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            alpha=float(cover_alpha),
        )

        zmaxabs = max(abs(float(zlim[0])), abs(float(zlim[1])), 1e-300)
        exp = int(np.floor(np.log10(zmaxabs)))
        scale = 10.0**exp
        ax.zaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v / scale:g}"))

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
            sci.set_clip_on(False)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    if autocrop:
        _autocrop_png_inplace(out_png)
    print(f"[save:{mode}] {out_png}")
    return out_png


# Backward-compatible alias for old callers that only need the final renderer.
def save_surface_with_cover(fig_path: str | Path, **kwargs) -> Path:
    if "F" in kwargs and "cover" not in kwargs:
        kwargs["cover"] = kwargs.pop("F")
    kwargs.pop("publication", None)
    kwargs.pop("title", None)
    return save_publication_surface_with_cover(out_png=fig_path, **kwargs)


__all__ = [
    "normalize_domain_mode",
    "prepare_surface_cover_domain",
    "save_publication_surface_with_cover",
    "save_surface_with_cover",
]