from __future__ import annotations

"""
Unified cover-function builders for flexural-cover wave experiments.

This module is intended to live at:
    src/waveslab/cover_core.py

It centralizes the cover-model construction that used to be split across the
SAM cover builder, ellipse/synthetic cover helpers, and the wavelength-scaling
press-play script.

Main public entry points
------------------------
- build_cover_npzs_from_sam_pipeline(...)
    Build the native direct-mask and ellipse-approximation solver covers.
- select_cover_for_case(...)
    Return the selected solver cover for a runner case, including optional
    lambda_flex matching and homogeneous baselines.
- calibrate_cover_npz_to_flexural_wavelength(...)
    Scale floe geometry in a saved cover NPZ so a chosen equivalent-diameter
    statistic matches a target flexural wavelength.
- RigidityField.from_cover_array(...)
    Map F in [0,1] to a spatially varying rigidity/aleph field.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np

ReferenceStat = Literal[
    "number_mean",
    "number_median",
    "area_weighted_mean",
    "area_weighted_median",
]


# =============================================================================
# JSON / NPZ utilities
# =============================================================================
def json_ready(obj: Any) -> Any:
    """Convert Paths and numpy scalars/arrays into JSON-serializable objects."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):
        return json_ready(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def load_cover_npz(npz_path: str | Path) -> dict[str, Any]:
    """Load a saved cover NPZ into an eager dictionary."""
    npz_path = Path(npz_path).expanduser()
    if not npz_path.exists():
        raise FileNotFoundError(f"Cover NPZ not found: {npz_path}")

    out: dict[str, Any] = {"npz_path": npz_path}
    with np.load(str(npz_path), allow_pickle=True) as z:
        for key in z.files:
            arr = z[key]
            if arr.shape == ():
                out[key] = arr.item()
            else:
                out[key] = np.asarray(arr)
    if "F_low" not in out:
        raise KeyError(f"{npz_path} does not contain F_low.")
    if "F_hi" not in out:
        out["F_hi"] = np.asarray(out["F_low"])
    if "binary_crop" in out:
        out["binary_crop"] = np.asarray(out["binary_crop"]).astype(bool)
    return out


def infer_pixel_scale_m(
    run_summary_json: str | Path | None,
    *,
    override: float | None = None,
    fallback: float = 1.0 / 29.0,
) -> float:
    """Read the physical pixel scale from a SAM run summary if available."""
    if override is not None:
        return float(override)
    if run_summary_json is not None:
        path = Path(run_summary_json).expanduser()
        if path.exists():
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                for key in (
                    "pixel_scale_m",
                    "meters_per_pixel",
                    "metres_per_pixel",
                    "m_per_px",
                    "mpp",
                ):
                    value = summary.get(key)
                    if value is not None:
                        return float(value)
            except Exception:
                pass
    return float(fallback)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# =============================================================================
# Cover building primitives
# =============================================================================
def _smooth_heaviside(phi: np.ndarray, width: float) -> np.ndarray:
    w = max(float(width), 1e-12)
    return 0.5 * (1.0 + np.tanh(np.pi * phi / w))


def _cover_from_binary_crop(binary_bool: np.ndarray, smooth_width_px: float) -> np.ndarray:
    """Convert a binary floe mask into a smooth cover fraction F in [0,1]."""
    import cv2

    binary_bool = np.asarray(binary_bool).astype(bool)
    floes_u8 = binary_bool.astype(np.uint8)
    water_u8 = 1 - floes_u8
    d_in = cv2.distanceTransform(floes_u8, distanceType=cv2.DIST_L2, maskSize=3)
    d_out = cv2.distanceTransform(water_u8, distanceType=cv2.DIST_L2, maskSize=3)
    phi = d_in - d_out
    return np.clip(_smooth_heaviside(phi, width=smooth_width_px), 0.0, 1.0)


def resize_cover_to_solver_grid(F_hi: np.ndarray, *, N: int, M: int, preserve_mean: bool = True) -> np.ndarray:
    """Downsample a high-resolution cover field to solver shape (M,N)."""
    import cv2

    F_hi = np.asarray(F_hi, dtype=np.float32)
    F_low = cv2.resize(F_hi, (int(N), int(M)), interpolation=cv2.INTER_AREA)
    if preserve_mean:
        mean_hi = float(np.mean(F_hi))
        mean_low = float(np.mean(F_low))
        if mean_low > 0.0:
            F_low *= mean_hi / mean_low
    return np.clip(F_low, 0.0, 1.0)


def _validate_crop(shape_hw: tuple[int, int], *, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int) -> tuple[int, int, int, int]:
    H, W = int(shape_hw[0]), int(shape_hw[1])
    x0, y0 = int(crop_x0), int(crop_y0)
    cw, ch = int(crop_w), int(crop_h)
    if cw <= 0 or ch <= 0:
        raise ValueError(f"Crop dimensions must be positive; got crop_w={cw}, crop_h={ch}.")
    if x0 < 0 or y0 < 0 or x0 + cw > W or y0 + ch > H:
        raise ValueError(f"Crop out of bounds for array with W={W}, H={H}: x0={x0}, y0={y0}, w={cw}, h={ch}.")
    return x0, y0, cw, ch


def _save_cover_npz(
    npz_path: str | Path,
    *,
    F_hi: np.ndarray,
    F_low: np.ndarray,
    binary_crop: np.ndarray,
    origin_xy: tuple[int, int],
    crop_w: int,
    crop_h: int,
    N: int,
    M: int,
    smooth_width_px: float,
    cover_kind: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a standardized cover NPZ."""
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(
        F_hi=np.asarray(F_hi, dtype=np.float32),
        F_low=np.asarray(F_low, dtype=np.float32),
        binary_crop=np.asarray(binary_crop, dtype=np.uint8),
        origin_xy=np.array([int(origin_xy[0]), int(origin_xy[1])], dtype=np.int32),
        crop_size=np.int32(int(crop_w)),  # legacy compatibility for square crops
        crop_w=np.int32(int(crop_w)),
        crop_h=np.int32(int(crop_h)),
        crop_shape=np.array([int(crop_h), int(crop_w)], dtype=np.int32),
        N=np.int32(int(N)),
        M=np.int32(int(M)),
        smooth_width_px=np.float32(float(smooth_width_px)),
        cover_kind=np.asarray(str(cover_kind)),
    )
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, Path):
                payload[key] = np.asarray(str(value))
            elif isinstance(value, str):
                payload[key] = np.asarray(value)
            else:
                payload[key] = value
    np.savez_compressed(npz_path, **payload)
    return npz_path


def _write_binary_png(path: Path, mask: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask).astype(bool) * 255).astype(np.uint8)).save(path)


def _write_cover_preview(
    out_png: Path,
    *,
    binary_crop: np.ndarray,
    F_hi: np.ndarray,
    F_low: np.ndarray,
    title: str,
) -> None:
    try:
        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ch, cw = np.asarray(binary_crop).shape
        F_low_up = cv2.resize(np.asarray(F_low, dtype=np.float32), (cw, ch), interpolation=cv2.INTER_NEAREST)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
        axes[0].imshow(binary_crop, cmap="gray")
        axes[0].set_title("Binary crop")
        axes[0].axis("off")
        axes[1].imshow(F_hi, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Smooth high-res")
        axes[1].axis("off")
        im = axes[2].imshow(F_low_up, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Solver grid")
        axes[2].axis("off")
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax).set_label("F")
        fig.suptitle(title)
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] cover preview failed for {out_png}: {exc}")


def _build_cover_from_binary_array(
    *,
    binary_array: np.ndarray,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    smooth_width_px: float,
    N: int,
    M: int,
    out_path: str | Path,
    cover_kind: str,
    source_label: str,
    write_preview: bool = True,
    extra: dict[str, Any] | None = None,
) -> Path:
    binary_array = np.asarray(binary_array).astype(bool)
    x0, y0, cw, ch = _validate_crop(binary_array.shape, crop_x0=crop_x0, crop_y0=crop_y0, crop_w=crop_w, crop_h=crop_h)
    binary_crop = binary_array[y0 : y0 + ch, x0 : x0 + cw]
    F_hi = _cover_from_binary_crop(binary_crop, smooth_width_px=smooth_width_px)
    F_low = resize_cover_to_solver_grid(F_hi, N=N, M=M, preserve_mean=True)
    npz_path = _save_cover_npz(
        out_path,
        F_hi=F_hi,
        F_low=F_low,
        binary_crop=binary_crop,
        origin_xy=(x0, y0),
        crop_w=cw,
        crop_h=ch,
        N=N,
        M=M,
        smooth_width_px=smooth_width_px,
        cover_kind=cover_kind,
        extra={"source_label": source_label, **(extra or {})},
    )
    if write_preview:
        _write_cover_preview(npz_path.with_suffix(".png"), binary_crop=binary_crop, F_hi=F_hi, F_low=F_low, title=cover_kind)
    return npz_path


def build_cover_from_binary_crop(
    *,
    binary_path: str | Path,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    smooth_width_px: float,
    N: int,
    M: int,
    out_dir: str | Path,
    write_preview: bool = True,
) -> Path:
    """Legacy helper: build cover_direct_from_crop.npz from a binary image file."""
    from PIL import Image

    binary_path = Path(binary_path).expanduser()
    full = np.asarray(Image.open(binary_path).convert("L")) > 127
    return _build_cover_from_binary_array(
        binary_array=full,
        crop_x0=crop_x0,
        crop_y0=crop_y0,
        crop_w=crop_w,
        crop_h=crop_h,
        smooth_width_px=smooth_width_px,
        N=N,
        M=M,
        out_path=Path(out_dir) / "cover_direct_from_crop.npz",
        cover_kind="direct_binary_crop",
        source_label=str(binary_path),
        write_preview=write_preview,
        extra={"binary_path": binary_path},
    )


# =============================================================================
# SAM pipeline input discovery and builders
# =============================================================================
def _as_binary_mask(arr: np.ndarray) -> np.ndarray:
    """Convert a label/mask-like array into a boolean mask."""
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim == 3:
        # Either a stack of masks or an RGB label/overlay-like image. For a stack,
        # any nonzero value means covered. For RGB, any nonzero channel is accepted.
        if a.shape[-1] in (3, 4):
            a = np.any(a[..., :3] > 0, axis=-1)
        else:
            a = np.any(a > 0, axis=0)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2D mask/label array after squeezing, got shape {a.shape}.")
    if a.dtype == bool:
        return a
    finite = np.asarray(a, dtype=float)
    vals = finite[np.isfinite(finite)]
    if vals.size == 0:
        return np.zeros_like(a, dtype=bool)
    vmax = float(np.max(vals))
    vmin = float(np.min(vals))
    if vmax <= 1.0 and vmin >= 0.0:
        return finite > 0.5
    # Label maps and binary uint8 masks both become >0. This is intentional for
    # accepted_label_map because label values encode floe IDs.
    return finite > 0.0


def _candidate_mask_keys(keys: list[str], requested: str) -> list[str]:
    requested = str(requested).strip()
    lower_to_key = {k.lower(): k for k in keys}
    out: list[str] = []

    def add(key: str) -> None:
        if key in keys and key not in out:
            out.append(key)
        elif key.lower() in lower_to_key and lower_to_key[key.lower()] not in out:
            out.append(lower_to_key[key.lower()])

    if requested:
        add(requested)

    for key in [
        "accepted_label_map",
        "accepted_labels",
        "accepted_instance_labels",
        "accepted_instances",
        "final_accepted_label_map",
        "final_label_map",
        "final_accepted_mask",
        "accepted_mask",
        "mask_accepted",
        "final_mask",
        "floe_mask",
        "combined_mask",
        "binary_mask",
        "candidate_binary_roi",
        "final_candidate_binary_roi",
    ]:
        add(key)

    scored: list[tuple[int, str]] = []
    for key in keys:
        lk = key.lower()
        score = 0
        if "accepted" in lk:
            score += 10
        if "final" in lk:
            score += 6
        if "label" in lk:
            score += 5
        if "mask" in lk:
            score += 5
        if "binary" in lk:
            score += 3
        if "overlay" in lk or "rgb" in lk or "image" in lk or "orthorect" in lk:
            score -= 12
        if score > 0:
            scored.append((score, key))
    for _, key in sorted(scored, key=lambda item: (-item[0], item[1])):
        if key not in out:
            out.append(key)
    return out


def _load_direct_binary_from_pipeline(pipeline_dir: str | Path, *, direct_source: str) -> tuple[np.ndarray, str, Path | None]:
    pipeline_dir = Path(pipeline_dir).expanduser()
    arrays_path = pipeline_dir / "data" / "pipeline_arrays.npz"
    if arrays_path.exists():
        with np.load(str(arrays_path), allow_pickle=True) as z:
            keys = list(z.files)
            tried: list[str] = []
            for key in _candidate_mask_keys(keys, direct_source):
                tried.append(key)
                arr = np.asarray(z[key])
                try:
                    mask = _as_binary_mask(arr)
                except Exception:
                    continue
                # Avoid accidentally selecting a blank overlay/placeholder.
                if mask.size and 0.0001 < float(np.mean(mask)) < 0.9999:
                    return mask, key, arrays_path
            raise KeyError(
                f"Could not find a usable accepted mask in {arrays_path}. "
                f"Requested direct_source={direct_source!r}; tried={tried}; available={keys}."
            )

    # Fallback for older pipelines that saved mask images instead of arrays.
    image_candidates = [
        pipeline_dir / "data" / f"{direct_source}.png",
        pipeline_dir / "data" / "accepted_label_map.png",
        pipeline_dir / "data" / "final_accepted_mask.png",
        pipeline_dir / "data" / "accepted_mask.png",
        pipeline_dir / "20_area_coloured_accepted_instances.png",
        pipeline_dir / "19_final_candidate_binary_roi.png",
    ]
    from PIL import Image

    for path in image_candidates:
        if path.exists():
            arr = np.asarray(Image.open(path).convert("L"))
            mask = arr > 0
            if mask.size and 0.0001 < float(np.mean(mask)) < 0.9999:
                return mask, path.stem, path

    raise FileNotFoundError(
        f"Could not find data/pipeline_arrays.npz or a fallback mask image in {pipeline_dir}."
    )


def _rasterize_ellipses_to_roi(df, *, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int) -> np.ndarray:
    """Rasterize a floe catalogue's fitted ellipses into a crop-sized mask."""
    import cv2

    required = {"centroid_col", "centroid_row", "major_axis", "minor_axis", "angle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Floe catalogue is missing columns needed for ellipse cover: {sorted(missing)}")

    roi = np.zeros((int(crop_h), int(crop_w)), dtype=np.uint8)
    for _, r in df.iterrows():
        cx = float(r["centroid_col"]) - float(crop_x0)
        cy = float(r["centroid_row"]) - float(crop_y0)
        if cx < -2 or cy < -2 or cx > (crop_w + 2) or cy > (crop_h + 2):
            continue
        a = 0.5 * float(r["major_axis"])
        b = 0.5 * float(r["minor_axis"])
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
            continue
        angle_deg = float(r["angle"])
        # The current catalogue normally stores radians. If values look like degrees,
        # keep them unchanged.
        if abs(angle_deg) <= 2.0 * np.pi + 1e-6:
            angle_deg = angle_deg * 180.0 / np.pi
        cv2.ellipse(
            roi,
            center=(int(round(cx)), int(round(cy))),
            axes=(max(1, int(round(a))), max(1, int(round(b)))),
            angle=angle_deg,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1,
            lineType=cv2.LINE_8,
        )
    return roi.astype(bool)


def build_cover_from_ellipse_crop(
    *,
    df_path: str | Path,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    smooth_width_px: float,
    N: int,
    M: int,
    out_dir: str | Path,
    write_preview: bool = True,
) -> Path:
    """Build cover_from_ellipse_crop.npz from a floe ellipse catalogue CSV."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(df_path).expanduser())

    x0, y0 = int(crop_x0), int(crop_y0)
    cw, ch = int(crop_w), int(crop_h)
    pad = 2 * max(cw, ch)
    df_roi = df[
        (df["centroid_col"] >= x0 - pad)
        & (df["centroid_col"] <= x0 + cw + pad)
        & (df["centroid_row"] >= y0 - pad)
        & (df["centroid_row"] <= y0 + ch + pad)
    ].copy()

    roi_bool = _rasterize_ellipses_to_roi(df_roi, crop_x0=x0, crop_y0=y0, crop_w=cw, crop_h=ch)
    F_hi = _cover_from_binary_crop(roi_bool, smooth_width_px=smooth_width_px)
    F_low = resize_cover_to_solver_grid(F_hi, N=N, M=M, preserve_mean=True)
    npz_path = _save_cover_npz(
        out_dir / "cover_from_ellipse_crop.npz",
        F_hi=F_hi,
        F_low=F_low,
        binary_crop=roi_bool,
        origin_xy=(x0, y0),
        crop_w=cw,
        crop_h=ch,
        N=N,
        M=M,
        smooth_width_px=smooth_width_px,
        cover_kind="ellipse_crop",
        extra={"df_path": Path(df_path).expanduser(), "n_ellipses_roi": np.int32(len(df_roi))},
    )
    if write_preview:
        _write_cover_preview(npz_path.with_suffix(".png"), binary_crop=roi_bool, F_hi=F_hi, F_low=F_low, title="ellipse_crop")
    return npz_path


def build_cover_npzs_from_sam_pipeline(
    *,
    pipeline_dir: str | Path,
    out_dir: str | Path,
    direct_source: str = "accepted_label_map",
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    smooth_width_px: float,
    N: int,
    M: int,
    write_full_binary_exports: bool = True,
    write_preview: bool = True,
) -> dict[str, Any]:
    """Build native direct-mask and ellipse-approximation cover NPZs from SAM outputs.

    Expected current layout:
        pipeline_dir/data/pipeline_arrays.npz
        pipeline_dir/data/floe_catalog.csv
        pipeline_dir/run_summary.json                 (optional)

    The direct mask is selected flexibly from pipeline_arrays.npz so the builder
    remains usable across minor naming changes in the segmentation pipeline.
    """
    pipeline_dir = Path(pipeline_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    direct_binary, direct_key, direct_source_path = _load_direct_binary_from_pipeline(
        pipeline_dir,
        direct_source=direct_source,
    )
    direct_npz = _build_cover_from_binary_array(
        binary_array=direct_binary,
        crop_x0=crop_x0,
        crop_y0=crop_y0,
        crop_w=crop_w,
        crop_h=crop_h,
        smooth_width_px=smooth_width_px,
        N=N,
        M=M,
        out_path=out_dir / "cover_direct_from_crop.npz",
        cover_kind="direct_mask_crop",
        source_label=direct_key,
        write_preview=write_preview,
        extra={
            "pipeline_dir": pipeline_dir,
            "direct_source_key": direct_key,
            "direct_source_path": direct_source_path,
        },
    )

    floe_catalog = pipeline_dir / "data" / "floe_catalog.csv"
    if not floe_catalog.exists():
        raise FileNotFoundError(f"Missing floe catalogue: {floe_catalog}")
    ellipse_npz = build_cover_from_ellipse_crop(
        df_path=floe_catalog,
        crop_x0=crop_x0,
        crop_y0=crop_y0,
        crop_w=crop_w,
        crop_h=crop_h,
        smooth_width_px=smooth_width_px,
        N=N,
        M=M,
        out_dir=out_dir,
        write_preview=write_preview,
    )

    if write_full_binary_exports:
        _write_binary_png(out_dir / "direct_mask_full.png", direct_binary)
        x0, y0, cw, ch = _validate_crop(direct_binary.shape, crop_x0=crop_x0, crop_y0=crop_y0, crop_w=crop_w, crop_h=crop_h)
        _write_binary_png(out_dir / "direct_mask_crop.png", direct_binary[y0 : y0 + ch, x0 : x0 + cw])
        ellipse = load_cover_npz(ellipse_npz)
        _write_binary_png(out_dir / "ellipse_mask_crop.png", np.asarray(ellipse["binary_crop"]).astype(bool))

    direct_cover = load_cover_npz(direct_npz)
    ellipse_cover = load_cover_npz(ellipse_npz)
    run_summary = _read_json_if_exists(pipeline_dir / "run_summary.json")
    meta: dict[str, Any] = {
        "pipeline_dir": pipeline_dir,
        "out_dir": out_dir,
        "direct_npz": direct_npz,
        "ellipse_npz": ellipse_npz,
        "direct_source_requested": direct_source,
        "direct_source_selected": direct_key,
        "direct_source_path": direct_source_path,
        "floe_catalog": floe_catalog,
        "crop": {"x0": int(crop_x0), "y0": int(crop_y0), "w": int(crop_w), "h": int(crop_h)},
        "solver_grid": {"N": int(N), "M": int(M)},
        "smooth_width_px": float(smooth_width_px),
        "run_summary_json": pipeline_dir / "run_summary.json" if (pipeline_dir / "run_summary.json").exists() else None,
        "pixel_scale_m": run_summary.get("pixel_scale_m"),
        "cover_stats": {
            "direct": _cover_stats(direct_cover["F_low"]),
            "ellipse": _cover_stats(ellipse_cover["F_low"]),
        },
    }
    summary_json = out_dir / "cover_build_summary.json"
    summary_json.write_text(json.dumps(json_ready(meta), indent=2), encoding="utf-8")
    meta["summary_json"] = summary_json
    return meta


def _cover_stats(F: np.ndarray) -> dict[str, float]:
    F = np.asarray(F, dtype=float)
    return {
        "mean": float(np.mean(F)),
        "min": float(np.min(F)),
        "max": float(np.max(F)),
        "std": float(np.std(F)),
        "nonzero_fraction": float(np.mean(F > 0)),
    }


# =============================================================================
# Floe catalogue stats and synthetic cover generation
# =============================================================================
def load_floe_catalog(csv_path: str | Path):
    """Load a floe ellipse catalogue CSV."""
    import pandas as pd

    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Floe catalogue not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"area", "major_axis", "minor_axis", "angle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    return df


def fit_floe_pdfs(df, area_weight: bool = True):
    """Fit simple parametric PDFs for area, aspect ratio, and orientation."""
    from scipy import stats

    A = np.asarray(df["area"], dtype=float)
    A = A[np.isfinite(A) & (A > 0)]
    if A.size == 0:
        raise ValueError("No positive areas found for fitting.")
    shape_a, loc_a, scale_a = stats.lognorm.fit(A, floc=0)
    pdf_area = stats.lognorm(shape_a, loc=loc_a, scale=scale_a)

    r_raw = np.asarray(df["major_axis"], float) / np.asarray(df["minor_axis"], float)
    r_raw = r_raw[np.isfinite(r_raw) & (r_raw > 0)]
    r_raw = np.maximum(r_raw, 1.0)
    if r_raw.size == 0:
        raise ValueError("No valid aspect ratios found for fitting.")
    r_lo = 1.0
    r_hi = float(np.percentile(r_raw, 99.0))
    r = np.clip(r_raw, r_lo, r_hi)

    r_n = (r - r_lo) / max((r_hi - r_lo), np.finfo(float).tiny)
    eps = float(np.finfo(float).eps ** 0.5)
    r_n = np.clip(r_n, eps, 1.0 - eps)
    a_r, b_r, _, _ = stats.beta.fit(r_n, floc=0, fscale=1)
    pdf_ar = stats.beta(a_r, b_r, loc=0, scale=1)

    theta = np.asarray(df["angle"], float)
    theta = theta[np.isfinite(theta)]
    if theta.size == 0:
        raise ValueError("No valid angles found for fitting.")
    if np.nanmax(np.abs(theta)) > 2.0 * np.pi + 1e-6:
        theta = np.deg2rad(theta)
    kappa, loc_vm, _ = stats.vonmises.fit(theta, fscale=1)
    pdf_theta = stats.vonmises(kappa, loc=loc_vm, scale=1)

    weights = None
    if area_weight:
        s = float(np.asarray(df["area"], float).sum())
        if s > 0:
            weights = df["area"] / s
    return pdf_area, pdf_ar, r_hi, pdf_theta, weights


def sample_floes(pdf_area, pdf_ar, r_hi: float, n_floes: int, *, px2m: float = 0.05, pdf_theta=None, rng=None):
    """Sample n_floes ellipses (a,b,theta). Semi-axes are returned in metres."""
    if rng is None:
        rng = np.random.default_rng()

    A = pdf_area.rvs(size=int(n_floes), random_state=rng)
    r_n = pdf_ar.rvs(size=int(n_floes), random_state=rng)
    r = 1.0 + r_n * (float(r_hi) - 1.0)

    b = np.sqrt(A / (np.pi * r))
    a = r * b
    th = (
        pdf_theta.rvs(size=int(n_floes), random_state=rng)
        if pdf_theta is not None
        else rng.uniform(-np.pi, np.pi, size=int(n_floes))
    )
    return a * float(px2m), b * float(px2m), th


def synthetic_cover(
    *,
    nx: int,
    ny: int,
    x_phys: Tuple[float, float],
    y_phys: Tuple[float, float],
    df_path: str | Path,
    cover_fraction: float | None,
    smoothing_width: float,
    seed: int,
    px2m: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic ellipse cover from fitted floe statistics."""
    df = load_floe_catalog(Path(df_path).expanduser())
    rng = np.random.default_rng(int(seed))
    pdf_A, pdf_ar, r_hi, pdf_th, _ = fit_floe_pdfs(df)

    x = np.linspace(*x_phys, int(nx))
    y = np.linspace(*y_phys, int(ny))
    X, Y = np.meshgrid(x, y, indexing="xy")
    Lx = float(x_phys[1] - x_phys[0])
    Ly = float(y_phys[1] - y_phys[0])

    if cover_fraction is None:
        cover_fraction = float(df["area"].sum() / ((2220 * px2m) * (1300 * px2m)))
    area_mean_m2 = float(pdf_A.mean() * (float(px2m) ** 2))
    n_floes = int((float(cover_fraction) * Lx * Ly) / max(area_mean_m2, 1e-12))
    n_floes = max(4, min(n_floes, 250))

    a, b, th = sample_floes(pdf_A, pdf_ar, r_hi, n_floes, px2m=px2m, pdf_theta=pdf_th, rng=rng)
    xc = rng.uniform(float(x_phys[0]), float(x_phys[1]), size=n_floes)
    yc = rng.uniform(float(y_phys[0]), float(y_phys[1]), size=n_floes)

    F = np.zeros((int(ny), int(nx)), dtype=float)
    H = lambda s, w: 0.5 * (1.0 + np.tanh(np.pi * s / max(float(w), 1e-12)))
    for aj, bj, thetaj, xj, yj in zip(a, b, th, xc, yc):
        cT, sT = np.cos(thetaj), np.sin(thetaj)
        xR = (X - xj) * cT + (Y - yj) * sT
        yR = -(X - xj) * sT + (Y - yj) * cT
        rho = (xR / aj) ** 2 + (yR / bj) ** 2
        F += H(1.0 - rho, w=smoothing_width)
    return X, Y, np.clip(F, 0.0, 1.0)


# =============================================================================
# Homogeneous and logistic channel/ridge covers
# =============================================================================
def build_homogeneous_cover_npz(
    *,
    out_path: str | Path,
    value: float,
    N: int,
    M: int,
    F_hi_shape: tuple[int, int] | None = None,
    source_label: str = "constant",
) -> Path:
    """Write a spatially homogeneous cover NPZ."""
    value = float(np.clip(value, 0.0, 1.0))
    if F_hi_shape is None:
        F_hi_shape = (int(M), int(N))
    F_low = np.full((int(M), int(N)), value, dtype=np.float32)
    F_hi = np.full(tuple(int(v) for v in F_hi_shape), value, dtype=np.float32)
    return _save_cover_npz(
        out_path,
        F_hi=F_hi,
        F_low=F_low,
        binary_crop=F_hi >= 0.5,
        origin_xy=(0, 0),
        crop_w=int(F_hi.shape[1]),
        crop_h=int(F_hi.shape[0]),
        N=N,
        M=M,
        smooth_width_px=0.0,
        cover_kind="homogeneous",
        extra={
            "homogeneous_value": np.asarray(value, dtype=np.float32),
            "homogeneous_value_source": source_label,
        },
    )


def logistic_beta_y(y: np.ndarray, *, gamma: float, sigma: float) -> np.ndarray:
    """Logistic transverse cover field beta(y) used for channel/ridge tests."""
    y = np.asarray(y, dtype=float)
    y_M = float(np.max(np.abs(y))) if np.any(y < 0.0) else float(np.max(y))
    beta_pos = 1.0 / (1.0 + np.exp(float(gamma) * (np.abs(y) - y_M / 2.0 + float(sigma))))
    return np.clip(beta_pos, 0.0, 1.0)


def logistic_y_cover(
    *,
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    sigma: float,
) -> np.ndarray:
    """Return F(x,y)=beta(y) repeated across x for channel/ridge covers."""
    beta = logistic_beta_y(np.asarray(y, dtype=float).reshape(-1), gamma=gamma, sigma=sigma)
    return np.repeat(beta[:, None], int(np.asarray(x).size), axis=1)


def build_logistic_y_cover_npz(
    *,
    out_path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    sigma: float,
) -> Path:
    """Write a logistic channel/ridge cover NPZ on an existing solver grid."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    F = logistic_y_cover(x=x, y=y, gamma=gamma, sigma=sigma)
    kind = "ridge" if float(gamma) > 0 else "channel" if float(gamma) < 0 else "logistic"
    return _save_cover_npz(
        out_path,
        F_hi=F,
        F_low=F,
        binary_crop=F >= 0.5,
        origin_xy=(0, 0),
        crop_w=int(F.shape[1]),
        crop_h=int(F.shape[0]),
        N=int(F.shape[1]),
        M=int(F.shape[0]),
        smooth_width_px=0.0,
        cover_kind=f"logistic_y_{kind}",
        extra={"gamma": np.asarray(float(gamma)), "sigma": np.asarray(float(sigma))},
    )


# =============================================================================
# Cover → rigidity mapping
# =============================================================================
def pair_average_x(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[1] < 2:
        raise ValueError("pair_average_x expects 2D array with >=2 columns.")
    return 0.5 * (A[:, :-1] + A[:, 1:])


def rigidity_from_cover(F: np.ndarray, bounds: Tuple[float, float] = (0.05, 1.0)) -> np.ndarray:
    dmin, dmax = bounds
    if not (0.0 <= dmin <= dmax):
        raise ValueError("bounds must satisfy 0 <= dmin <= dmax")
    F = np.clip(np.asarray(F, dtype=float), 0.0, 1.0)
    return dmin + (dmax - dmin) * F


@dataclass
class RigidityField:
    aleph: np.ndarray
    cover: np.ndarray | None = None

    @staticmethod
    def constant(M: int, N: int, aleph: float) -> "RigidityField":
        return RigidityField(float(aleph) * np.ones((int(M), int(N)), dtype=float), None)

    @staticmethod
    def from_cover_array(F: np.ndarray, *, bounds: tuple[float, float]) -> "RigidityField":
        F = np.asarray(F, dtype=float)
        return RigidityField(rigidity_from_cover(F, bounds=bounds), F)


def cover_from_npz(npz_path: str | Path, *, use_low: bool, bounds: tuple[float, float]) -> RigidityField:
    cover = load_cover_npz(npz_path)
    F = cover["F_low"] if use_low else cover["F_hi"]
    return RigidityField.from_cover_array(F, bounds=bounds)


def homogenized_cover_1d(F: np.ndarray, bounds: tuple[float, float]) -> RigidityField:
    if F.ndim != 2:
        raise ValueError("homogenized_cover_1d expects 2D array.")
    Fx = pair_average_x(F)
    Fy = Fx.mean(axis=1, keepdims=True)
    Fbar = np.repeat(Fy, F.shape[1], axis=1)
    return RigidityField(rigidity_from_cover(Fbar, bounds=bounds), Fbar)


def _rescale_cover_to_target(F: np.ndarray, target: float, *, threshold: float = 0.5, it: int = 30) -> np.ndarray:
    F = np.asarray(F, dtype=float)

    def frac(scale: float) -> float:
        return float((np.clip(scale * F, 0.0, 1.0) >= threshold).mean())

    lo, hi = 0.0, 10.0
    for _ in range(int(it)):
        mid = 0.5 * (lo + hi)
        if frac(mid) < target:
            lo = mid
        else:
            hi = mid
    return np.clip(0.5 * (lo + hi) * F, 0.0, 1.0)


# =============================================================================
# Wavelength scaling / lambda_flex matching
# =============================================================================
def _center_zoom_binary(binary_bool: np.ndarray, scale: float) -> np.ndarray:
    """Scale floe geometry about the crop center, preserving crop dimensions."""
    import cv2

    binary_bool = np.asarray(binary_bool).astype(bool)
    H, W = binary_bool.shape
    scale = max(float(scale), 1e-6)

    new_W = max(1, int(round(scale * W)))
    new_H = max(1, int(round(scale * H)))
    z = cv2.resize(binary_bool.astype(np.uint8), (new_W, new_H), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((H, W), dtype=np.uint8)

    if new_H >= H:
        y0_src = (new_H - H) // 2
        y1_src = y0_src + H
        y0_dst = 0
        y1_dst = H
    else:
        y0_src = 0
        y1_src = new_H
        y0_dst = (H - new_H) // 2
        y1_dst = y0_dst + new_H

    if new_W >= W:
        x0_src = (new_W - W) // 2
        x1_src = x0_src + W
        x0_dst = 0
        x1_dst = W
    else:
        x0_src = 0
        x1_src = new_W
        x0_dst = (W - new_W) // 2
        x1_dst = x0_dst + new_W

    out[y0_dst:y1_dst, x0_dst:x1_dst] = z[y0_src:y1_src, x0_src:x1_src]
    return out.astype(bool)


def _touches_border(component_mask: np.ndarray) -> bool:
    component_mask = np.asarray(component_mask).astype(bool)
    return bool(
        component_mask[0, :].any()
        or component_mask[-1, :].any()
        or component_mask[:, 0].any()
        or component_mask[:, -1].any()
    )


def floe_diameters_from_binary(
    binary_bool: np.ndarray,
    *,
    meters_per_pixel: float | None,
    min_area_px: int = 4,
    exclude_border: bool = True,
) -> dict[str, Any]:
    """Measure equivalent-circle floe diameter statistics from a binary crop."""
    import cv2

    binary_bool = np.asarray(binary_bool).astype(bool)
    lab_n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_bool.astype(np.uint8), connectivity=8)
    areas_px: list[float] = []
    for lab in range(1, lab_n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < int(min_area_px):
            continue
        mask = labels == lab
        if exclude_border and _touches_border(mask):
            continue
        areas_px.append(float(area))

    if not areas_px:
        raise RuntimeError("No interior floes found after filtering.")

    A_px = np.asarray(areas_px, dtype=float)
    D_eq_px = np.sqrt(4.0 * A_px / np.pi)

    out: dict[str, Any] = {
        "count": int(len(D_eq_px)),
        "areas_px": A_px,
        "diameters_px": D_eq_px,
        "number_mean_px": float(np.mean(D_eq_px)),
        "number_median_px": float(np.median(D_eq_px)),
        "area_weighted_mean_px": float(np.sum(A_px * D_eq_px) / np.sum(A_px)),
    }

    order = np.argsort(D_eq_px)
    d_sorted = D_eq_px[order]
    w_sorted = A_px[order]
    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    out["area_weighted_median_px"] = float(d_sorted[int(np.searchsorted(cdf, 0.5))])

    if meters_per_pixel is not None:
        mpp = float(meters_per_pixel)
        D_eq_m = D_eq_px * mpp
        out.update(
            {
                "diameters_m": D_eq_m,
                "number_mean_m": float(np.mean(D_eq_m)),
                "number_median_m": float(np.median(D_eq_m)),
                "area_weighted_mean_m": float(np.sum(A_px * D_eq_m) / np.sum(A_px)),
                "area_weighted_median_m": float(out["area_weighted_median_px"] * mpp),
            }
        )
    return out


def _reference_diameter(stats: dict[str, Any], which: ReferenceStat) -> float:
    key = {
        "number_mean": "number_mean_m",
        "number_median": "number_median_m",
        "area_weighted_mean": "area_weighted_mean_m",
        "area_weighted_median": "area_weighted_median_m",
    }[which]
    val = stats.get(key, None)
    if val is None:
        raise ValueError(f"Reference diameter statistic {which!r} is unavailable; did you pass meters_per_pixel?")
    return float(val)


def _match_mean_fraction(F: np.ndarray, target: float, *, tol: float = 1e-10, maxiter: int = 80) -> np.ndarray:
    """Shift/clamp F so its mean matches target while preserving spatial pattern."""
    F0 = np.asarray(F, dtype=float)
    target = float(np.clip(target, 0.0, 1.0))

    def g(delta: float) -> float:
        return float(np.mean(np.clip(F0 + delta, 0.0, 1.0)))

    lo, hi = -1.0, 1.0
    if target <= g(lo):
        return np.clip(F0 + lo, 0.0, 1.0)
    if target >= g(hi):
        return np.clip(F0 + hi, 0.0, 1.0)

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if abs(gm - target) < tol:
            return np.clip(F0 + mid, 0.0, 1.0)
        if gm < target:
            lo = mid
        else:
            hi = mid
    return np.clip(F0 + 0.5 * (lo + hi), 0.0, 1.0)


def _write_scaling_preview(out_png: Path, *, binary0: np.ndarray, binary1: np.ndarray, F_low0: np.ndarray, F_low1: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axs[0, 0].imshow(binary0, cmap="gray")
    axs[0, 0].set_title("Original binary crop")
    axs[0, 0].axis("off")
    axs[0, 1].imshow(binary1, cmap="gray")
    axs[0, 1].set_title("Scaled binary crop")
    axs[0, 1].axis("off")
    axs[1, 0].imshow(F_low0, cmap="gray", vmin=0, vmax=1)
    axs[1, 0].set_title("Original F_low")
    axs[1, 0].axis("off")
    im = axs[1, 1].imshow(F_low1, cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title("Scaled F_low")
    axs[1, 1].axis("off")
    fig.colorbar(im, ax=axs[1, :], shrink=0.8, location="right", label="F")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def calibrate_cover_npz_to_flexural_wavelength(
    *,
    input_npz: str | Path,
    output_npz: str | Path,
    target_lambda_flex_m: float,
    meters_per_pixel: float,
    reference_stat: ReferenceStat = "area_weighted_mean",
    min_area_px: int = 4,
    match_tol_rel: float = 0.05,
    max_scale_iters: int = 10,
    preserve_mean_cover: bool = True,
    current_reference_diameter_m: float | None = None,
    write_summary: bool = True,
    write_preview: bool = True,
) -> dict[str, Any]:
    """Scale a cover NPZ so a floe diameter statistic matches lambda_flex.

    The safe pattern is used: the native NPZ is left untouched and a separate
    *_lambda_flex_matched.npz is written for the optional scale-sensitivity case.
    """
    input_npz = Path(input_npz).expanduser()
    output_npz = Path(output_npz).expanduser()
    cover0 = load_cover_npz(input_npz)

    binary0 = np.asarray(cover0.get("binary_crop"), dtype=bool)
    F_low0 = np.asarray(cover0["F_low"], dtype=float)
    F_hi0 = np.asarray(cover0["F_hi"], dtype=float)
    N = int(cover0.get("N", F_low0.shape[1]))
    M = int(cover0.get("M", F_low0.shape[0]))
    smooth_width_px = float(cover0.get("smooth_width_px", 2.0))
    origin_xy = np.asarray(cover0.get("origin_xy", np.array([0, 0])), dtype=int)
    crop_h, crop_w = binary0.shape
    target_mean = float(np.mean(F_low0))

    stats0 = floe_diameters_from_binary(
        binary0,
        meters_per_pixel=float(meters_per_pixel),
        min_area_px=int(min_area_px),
        exclude_border=True,
    )
    if current_reference_diameter_m is None:
        D_ref0 = _reference_diameter(stats0, reference_stat)
        reference_source = f"measured_from_binary_crop:{reference_stat}"
    else:
        D_ref0 = float(current_reference_diameter_m)
        reference_source = "user_supplied_current_reference_diameter_m"

    target = float(target_lambda_flex_m)
    initial_scale = target / max(D_ref0, 1e-12)

    # Direct zoom works well most of the time, but crop truncation can perturb
    # the achieved diameter. Search a small log-scale neighborhood and keep the
    # best achieved diameter.
    evals: list[dict[str, Any]] = []

    def evaluate(scale: float) -> dict[str, Any] | None:
        try:
            b = _center_zoom_binary(binary0, scale=scale)
            stats = floe_diameters_from_binary(
                b,
                meters_per_pixel=float(meters_per_pixel),
                min_area_px=int(min_area_px),
                exclude_border=True,
            )
            D = _reference_diameter(stats, reference_stat)
            return {
                "scale": float(scale),
                "binary": b,
                "stats": stats,
                "D": float(D),
                "rel_error": float(abs(D - target) / max(abs(target), 1e-12)),
            }
        except Exception as exc:
            evals.append({"scale": float(scale), "error": f"{type(exc).__name__}: {exc}"})
            return None

    base = max(initial_scale, 1e-6)
    factors = np.geomspace(0.35, 2.75, max(9, int(max_scale_iters) + 9))
    candidates = [base * float(f) for f in factors]
    candidates.append(base)
    for scale in sorted(set(round(c, 10) for c in candidates)):
        res = evaluate(scale)
        if res is not None:
            evals.append({k: v for k, v in res.items() if k not in {"binary", "stats"}})

    valid = [e for e in evals if "D" in e]
    if not valid:
        raise RuntimeError(f"Could not evaluate any valid scaled cover for {input_npz}.")
    best_scale = float(min(valid, key=lambda e: e["rel_error"])["scale"])

    # Refine around the best log-scale value.
    for _ in range(max(0, int(max_scale_iters))):
        local = np.geomspace(best_scale / 1.18, best_scale * 1.18, 7)
        improved = False
        current_best = min([e for e in evals if "D" in e], key=lambda e: e["rel_error"])
        for scale in local:
            res = evaluate(float(scale))
            if res is not None:
                compact = {k: v for k, v in res.items() if k not in {"binary", "stats"}}
                evals.append(compact)
                if compact["rel_error"] < current_best["rel_error"]:
                    best_scale = compact["scale"]
                    current_best = compact
                    improved = True
        if current_best["rel_error"] <= float(match_tol_rel) or not improved:
            break

    best = evaluate(best_scale)
    if best is None:
        # This should be unreachable because best_scale came from a valid eval.
        raise RuntimeError("Best scale could not be re-evaluated.")
    binary1 = np.asarray(best["binary"], dtype=bool)
    stats1 = best["stats"]
    F_hi1 = _cover_from_binary_crop(binary1, smooth_width_px=smooth_width_px)
    F_low1 = resize_cover_to_solver_grid(F_hi1, N=N, M=M, preserve_mean=True)
    if preserve_mean_cover:
        F_low1 = _match_mean_fraction(F_low1, target_mean)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    _save_cover_npz(
        output_npz,
        F_hi=F_hi1,
        F_low=F_low1,
        binary_crop=binary1,
        origin_xy=(int(origin_xy[0]), int(origin_xy[1])),
        crop_w=int(crop_w),
        crop_h=int(crop_h),
        N=N,
        M=M,
        smooth_width_px=smooth_width_px,
        cover_kind="lambda_flex_matched",
        extra={
            "input_npz": input_npz,
            "target_lambda_flex_m": np.asarray(target, dtype=np.float64),
            "meters_per_pixel": np.asarray(float(meters_per_pixel), dtype=np.float64),
            "reference_stat": reference_stat,
            "best_scale": np.asarray(best_scale, dtype=np.float64),
            "preserve_mean_cover": np.asarray(bool(preserve_mean_cover)),
            "source_cover_kind": np.asarray(str(cover0.get("cover_kind", "unknown"))),
        },
    )

    preview_path = output_npz.with_suffix(".png")
    if write_preview:
        _write_scaling_preview(preview_path, binary0=binary0, binary1=binary1, F_low0=F_low0, F_low1=F_low1)

    summary = {
        "enabled": True,
        "input_npz": input_npz,
        "output_npz": output_npz,
        "meters_per_pixel": float(meters_per_pixel),
        "target_lambda_flex_m": target,
        "reference_source": reference_source,
        "reference_stat": reference_stat,
        "current_reference_diameter_m": float(D_ref0),
        "initial_scale": float(initial_scale),
        "best_scale": float(best_scale),
        "match_tol_rel": float(match_tol_rel),
        "achieved_reference_diameter_m": float(_reference_diameter(stats1, reference_stat)),
        "relative_error": float(abs(_reference_diameter(stats1, reference_stat) - target) / max(abs(target), 1e-12)),
        "preserve_mean_cover": bool(preserve_mean_cover),
        "before": _diameter_summary_for_json(stats0, F_low0),
        "after_binary_geometry": _diameter_summary_for_json(stats1, F_low1),
        "cover_mean_before": float(np.mean(F_low0)),
        "cover_mean_after": float(np.mean(F_low1)),
        "preview_png": preview_path if write_preview else None,
        "search_evaluations": evals,
    }
    if write_summary:
        summary_path = output_npz.with_suffix(".json")
        summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
        summary["summary_json"] = summary_path
    return summary


def _diameter_summary_for_json(stats: dict[str, Any], F_low: np.ndarray) -> dict[str, Any]:
    return {
        "count": stats.get("count"),
        "number_mean_m": stats.get("number_mean_m"),
        "number_median_m": stats.get("number_median_m"),
        "area_weighted_mean_m": stats.get("area_weighted_mean_m"),
        "area_weighted_median_m": stats.get("area_weighted_median_m"),
        "mean_cover_fraction": float(np.mean(F_low)),
    }


# =============================================================================
# Case selection helpers used by the flexural cover runner
# =============================================================================
@dataclass(frozen=True)
class CalibrationOptions:
    enabled: bool = False
    target_lambda_flex_m: float = 0.6041194293172856
    reference_stat: ReferenceStat = "area_weighted_mean"
    min_area_px: int = 4
    match_tol_rel: float = 0.05
    max_scale_iters: int = 10
    preserve_mean_cover: bool = True
    meters_per_pixel: float | None = None
    meters_per_pixel_fallback: float = 1.0 / 29.0


@dataclass
class SelectedCover:
    cover: dict[str, Any]
    npz_path: Path
    label: str
    source: str
    scale_summary: dict[str, Any]
    scale_summary_json: Path | None = None


def normalize_cover_case(case: str, *, calibration_enabled: bool = False) -> str:
    """Normalize user-facing cover case aliases."""
    raw = str(case).strip().lower().replace("-", "_")
    aliases = {
        "direct": "direct_native_mask",
        "direct_native": "direct_native_mask",
        "direct_native_mask": "direct_native_mask",
        "native": "direct_native_mask",
        "mask": "direct_native_mask",
        "ellipse": "ellipse_native_mask",
        "ellipse_native": "ellipse_native_mask",
        "ellipse_native_mask": "ellipse_native_mask",
        "ellipse_approx": "ellipse_native_mask",
        "ellipse_approximation": "ellipse_native_mask",
        "homogeneous": "homogeneous",
        "homogeneous_mean": "homogeneous",
        "uniform": "homogeneous",
        "constant": "homogeneous",
        "direct_lambda_flex": "direct_lambda_flex_matched",
        "direct_lambda_flex_matched": "direct_lambda_flex_matched",
        "direct_calibrated": "direct_lambda_flex_matched",
        "ellipse_lambda_flex": "ellipse_lambda_flex_matched",
        "ellipse_lambda_flex_matched": "ellipse_lambda_flex_matched",
        "ellipse_calibrated": "ellipse_lambda_flex_matched",
    }
    if raw not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown cover case {case!r}. Expected one of: {valid}.")
    out = aliases[raw]
    if calibration_enabled and out == "direct_native_mask":
        return "direct_lambda_flex_matched"
    if calibration_enabled and out == "ellipse_native_mask":
        return "ellipse_lambda_flex_matched"
    return out


def cover_cases_to_run(
    *,
    run_homogeneous: bool,
    run_direct: bool,
    run_ellipse: bool,
    use_single_case: bool,
    single_case: str,
    calibration_enabled: bool,
) -> list[str]:
    """Return requested cover cases in execution order."""
    if bool(use_single_case):
        return [single_case]
    requested: list[str] = []
    if run_homogeneous:
        requested.append("homogeneous")
    if run_direct:
        requested.append("direct")
    if run_ellipse:
        requested.append("ellipse")
    if not requested:
        raise ValueError("No cover cases are enabled.")

    seen: set[str] = set()
    out: list[str] = []
    for case in requested:
        norm = normalize_cover_case(case, calibration_enabled=calibration_enabled)
        if norm not in seen:
            seen.add(norm)
            out.append(case)
    return out


def _homogeneous_cover_value(
    value_spec: str,
    *,
    direct_cover: dict[str, Any],
    ellipse_cover: dict[str, Any],
) -> tuple[float, str]:
    spec = str(value_spec).strip().lower()
    if spec in {"direct_mean", "mean", "native_mean"}:
        return float(np.mean(np.asarray(direct_cover["F_low"], dtype=float))), "direct_mean"
    if spec in {"ellipse_mean", "ell_mean"}:
        return float(np.mean(np.asarray(ellipse_cover["F_low"], dtype=float))), "ellipse_mean"
    try:
        value = float(spec)
    except ValueError as exc:
        raise ValueError(
            "homogeneous_value must be direct_mean, ellipse_mean, or a numeric value in [0,1]; "
            f"got {value_spec!r}."
        ) from exc
    return float(np.clip(value, 0.0, 1.0)), "constant"


def select_cover_for_case(
    *,
    cover_case: str,
    direct_npz: str | Path,
    ellipse_npz: str | Path,
    cover_dir: str | Path,
    direct_cover: dict[str, Any] | None = None,
    ellipse_cover: dict[str, Any] | None = None,
    sam_pipeline_dir: str | Path | None = None,
    homogeneous_value: str = "direct_mean",
    calibration: CalibrationOptions | None = None,
) -> SelectedCover:
    """Select, build, or calibrate the cover used for one solver case."""
    direct_npz = Path(direct_npz)
    ellipse_npz = Path(ellipse_npz)
    cover_dir = Path(cover_dir)
    cover_dir.mkdir(parents=True, exist_ok=True)
    direct_cover = load_cover_npz(direct_npz) if direct_cover is None else direct_cover
    ellipse_cover = load_cover_npz(ellipse_npz) if ellipse_cover is None else ellipse_cover
    calibration = calibration or CalibrationOptions()

    case = normalize_cover_case(cover_case, calibration_enabled=calibration.enabled)
    if case == "direct_native_mask":
        return SelectedCover(
            cover=direct_cover,
            npz_path=direct_npz,
            label="direct_native_mask",
            source="direct",
            scale_summary={
                "enabled": False,
                "best_scale": 1.0,
                "achieved_reference_diameter_m": None,
                "target_lambda_flex_m": None,
                "input_npz": direct_npz,
                "output_npz": direct_npz,
                "note": "Solver used native direct mask-derived F_low.",
            },
            scale_summary_json=None,
        )

    if case == "ellipse_native_mask":
        return SelectedCover(
            cover=ellipse_cover,
            npz_path=ellipse_npz,
            label="ellipse_native_mask",
            source="ellipse",
            scale_summary={
                "enabled": False,
                "best_scale": 1.0,
                "achieved_reference_diameter_m": None,
                "target_lambda_flex_m": None,
                "input_npz": ellipse_npz,
                "output_npz": ellipse_npz,
                "note": "Solver used native ellipse-approximation F_low.",
            },
            scale_summary_json=None,
        )

    if case in {"direct_lambda_flex_matched", "ellipse_lambda_flex_matched"}:
        source_npz = direct_npz if case.startswith("direct") else ellipse_npz
        source_label = "direct" if case.startswith("direct") else "ellipse"
        out_npz = cover_dir / f"cover_{source_label}_lambda_flex_matched.npz"
        mpp = infer_pixel_scale_m(
            None if sam_pipeline_dir is None else Path(sam_pipeline_dir) / "run_summary.json",
            override=calibration.meters_per_pixel,
            fallback=calibration.meters_per_pixel_fallback,
        )
        scale_summary = calibrate_cover_npz_to_flexural_wavelength(
            input_npz=source_npz,
            output_npz=out_npz,
            target_lambda_flex_m=calibration.target_lambda_flex_m,
            meters_per_pixel=mpp,
            reference_stat=calibration.reference_stat,
            min_area_px=calibration.min_area_px,
            match_tol_rel=calibration.match_tol_rel,
            max_scale_iters=calibration.max_scale_iters,
            preserve_mean_cover=calibration.preserve_mean_cover,
            write_summary=True,
        )
        return SelectedCover(
            cover=load_cover_npz(out_npz),
            npz_path=out_npz,
            label=case,
            source=source_label,
            scale_summary=scale_summary,
            scale_summary_json=out_npz.with_suffix(".json"),
        )

    if case == "homogeneous":
        value, source = _homogeneous_cover_value(homogeneous_value, direct_cover=direct_cover, ellipse_cover=ellipse_cover)
        F_low_template = np.asarray(direct_cover["F_low"], dtype=np.float32)
        F_hi_template = np.asarray(direct_cover["F_hi"], dtype=np.float32)
        value_tag = f"{value:.4g}".replace(".", "p").replace("-", "m")
        out_npz = cover_dir / f"cover_homogeneous_{source}_{value_tag}.npz"
        build_homogeneous_cover_npz(
            out_path=out_npz,
            value=value,
            N=int(F_low_template.shape[1]),
            M=int(F_low_template.shape[0]),
            F_hi_shape=tuple(F_hi_template.shape),
            source_label=source,
        )
        cover = load_cover_npz(out_npz)
        label = f"homogeneous_{source}"
        return SelectedCover(
            cover=cover,
            npz_path=out_npz,
            label=label,
            source="homogeneous",
            scale_summary={
                "enabled": False,
                "best_scale": 1.0,
                "achieved_reference_diameter_m": None,
                "target_lambda_flex_m": None,
                "input_npz": None,
                "output_npz": out_npz,
                "homogeneous_value": value,
                "homogeneous_value_source": source,
                "note": "Solver used a spatially homogeneous cover field.",
            },
            scale_summary_json=None,
        )

    raise AssertionError(f"Unhandled normalized cover case {case!r}")