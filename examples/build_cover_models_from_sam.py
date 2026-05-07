from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from waveslab.cover_core import build_cover_npzs_from_sam_pipeline, json_ready


DEFAULT_PIPELINE_DIR = Path("outputs/sea_ice_segmentation")


REQUIRED_CATALOG_COLUMNS = {"centroid_col", "centroid_row", "major_axis", "minor_axis", "angle"}


def _catalog_has_cover_schema(path: Path) -> bool:
    """Return True if an existing compatibility catalog already has old cover columns."""
    try:
        cols = set(pd.read_csv(path, nrows=1).columns)
    except Exception:
        return False
    return REQUIRED_CATALOG_COLUMNS.issubset(cols)


def _read_pixel_size_m(pipeline_dir: Path, default: float = 0.05) -> float:
    """Read the pixel size used by the clean segmentation script.

    The selected catalog stores major/minor axes in metres, while the cover
    builder rasterizes ellipses in pixels.  The metadata written by
    segment_sea_ice_fair_compare.py records both the orthorectified pixel size
    and the value used for metrics; prefer the latter when present.
    """
    candidates = [
        pipeline_dir / "metadata" / "run_config.json",
        pipeline_dir / "run_config.json",
        pipeline_dir / "run_summary.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("pixel_size_m_for_metrics", "pixel_size_m", "meters_per_pixel", "m_per_px"):
            value = meta.get(key)
            if value is not None:
                return float(value)
        ortho = meta.get("orthorectification")
        if isinstance(ortho, dict):
            value = ortho.get("pixel_size_m")
            if value is not None:
                return float(value)
    return float(default)


def _first_existing_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _normalise_catalog_for_cover_builder(
    source_catalog: Path,
    catalog_path: Path,
    *,
    pipeline_dir: Path,
) -> dict[str, object]:
    """Write a cover-builder-compatible ellipse catalog.

    The clean segmentation catalog uses descriptive column names such as
    centroid_x_px, centroid_y_px, major_axis_length_m, and minor_axis_length_m.
    waveslab.cover_core.build_cover_from_ellipse_crop expects the older names:
    centroid_col, centroid_row, major_axis, minor_axis, and angle.  This function
    preserves the clean columns and appends the compatibility columns.
    """
    df = pd.read_csv(source_catalog)
    if df.empty:
        # Keep an empty catalog but with the required schema so downstream code
        # can fail gracefully because there are no ellipses, not because columns
        # are missing.
        for col in sorted(REQUIRED_CATALOG_COLUMNS):
            df[col] = []
        df.to_csv(catalog_path, index=False)
        return {"source_catalog": source_catalog, "catalog_path": catalog_path, "n_rows": 0, "pixel_size_m": None}

    x_col = _first_existing_column(df, ("centroid_col", "centroid_x_px", "centroid_x", "x_px", "x"))
    y_col = _first_existing_column(df, ("centroid_row", "centroid_y_px", "centroid_y", "y_px", "y"))
    if x_col is None or y_col is None:
        raise KeyError(
            f"{source_catalog} must contain centroid columns. Found {list(df.columns)}. "
            "Expected centroid_x_px/centroid_y_px from the clean segmentation output."
        )

    pixel_size_m = _read_pixel_size_m(pipeline_dir)

    major_px_col = _first_existing_column(df, ("major_axis", "major_axis_length_px", "major_px"))
    minor_px_col = _first_existing_column(df, ("minor_axis", "minor_axis_length_px", "minor_px"))
    major_m_col = _first_existing_column(df, ("major_axis_length_m", "major_m"))
    minor_m_col = _first_existing_column(df, ("minor_axis_length_m", "minor_m"))

    if major_px_col is not None:
        major_axis = pd.to_numeric(df[major_px_col], errors="coerce")
    elif major_m_col is not None:
        major_axis = pd.to_numeric(df[major_m_col], errors="coerce") / max(pixel_size_m, 1e-12)
    else:
        raise KeyError(f"{source_catalog} has no major-axis column. Found {list(df.columns)}")

    if minor_px_col is not None:
        minor_axis = pd.to_numeric(df[minor_px_col], errors="coerce")
    elif minor_m_col is not None:
        minor_axis = pd.to_numeric(df[minor_m_col], errors="coerce") / max(pixel_size_m, 1e-12)
    else:
        raise KeyError(f"{source_catalog} has no minor-axis column. Found {list(df.columns)}")

    angle_col = _first_existing_column(df, ("angle", "orientation", "orientation_rad", "theta", "theta_rad"))
    if angle_col is not None:
        angle = pd.to_numeric(df[angle_col], errors="coerce").fillna(0.0)
    else:
        # The clean catalog does not store orientation.  Use axis-aligned
        # ellipses rather than failing.  This is adequate for the ellipse
        # approximation and keeps the direct mask unchanged.
        angle = 0.0

    df["centroid_col"] = pd.to_numeric(df[x_col], errors="coerce")
    df["centroid_row"] = pd.to_numeric(df[y_col], errors="coerce")
    df["major_axis"] = major_axis
    df["minor_axis"] = minor_axis
    df["angle"] = angle

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["centroid_col", "centroid_row", "major_axis", "minor_axis", "angle"])
    df = df[(df["major_axis"] > 0.0) & (df["minor_axis"] > 0.0)].copy()
    df.to_csv(catalog_path, index=False)
    return {
        "source_catalog": source_catalog,
        "catalog_path": catalog_path,
        "n_rows_source": int(before),
        "n_rows_written": int(len(df)),
        "pixel_size_m": float(pixel_size_m),
        "assumed_axis_aligned_ellipses": bool(angle_col is None),
    }


def _npz_scalar_to_str(value: object, default: str) -> str:
    """Convert an NPZ scalar/string field to a normal Python string."""
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
        if arr.size == 1:
            return str(arr.reshape(-1)[0])
    except Exception:
        pass
    return default


def _load_selected_label_map(pipeline_dir: Path, selected_method: str | None) -> tuple[np.ndarray, np.ndarray, str, Path]:
    """Load the selected or requested segmentation mask from the new clean layout.

    The current segmentation script writes either

        data/selected_floe_labels.npz

    or method-specific files such as

        data/label_maps/sam_auto.npz.

    The cover builder still expects a single ``data/pipeline_arrays.npz`` file,
    so this function returns arrays that can be written to that compatibility
    file.
    """
    data_dir = pipeline_dir / "data"

    if selected_method:
        label_path = data_dir / "label_maps" / f"{selected_method}.npz"
        method_name = selected_method
    else:
        label_path = data_dir / "selected_floe_labels.npz"
        method_name = "selected"

    if not label_path.exists():
        available = sorted(p.stem for p in (data_dir / "label_maps").glob("*.npz"))
        raise FileNotFoundError(
            f"Could not find {label_path}. Available method label maps: {available}. "
            "Run examples/segment_sea_ice_fair_compare.py first, or pass --selected-method."
        )

    with np.load(label_path, allow_pickle=True) as z:
        if "label_map" not in z.files:
            raise KeyError(f"{label_path} does not contain a 'label_map' array.")
        label_map = np.asarray(z["label_map"], dtype=np.int32)
        if "binary_mask" in z.files:
            binary_mask = np.asarray(z["binary_mask"]).astype(np.uint8)
        else:
            binary_mask = (label_map > 0).astype(np.uint8)
        if not selected_method and "selected_method" in z.files:
            method_name = _npz_scalar_to_str(z["selected_method"], default=method_name)
        elif "method" in z.files:
            method_name = _npz_scalar_to_str(z["method"], default=method_name)

    return label_map, binary_mask, method_name, label_path


def ensure_cover_builder_inputs(
    pipeline_dir: Path,
    *,
    selected_method: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create the legacy cover-builder inputs from the new segmentation layout.

    New segmentation output layout:

        data/selected_floe_labels.npz
        data/selected_floe_catalog.csv
        data/label_maps/<method>.npz
        data/catalogs/<method>.csv

    Cover builder input layout expected by ``waveslab.cover_core``:

        data/pipeline_arrays.npz
        data/floe_catalog.csv

    This helper writes the expected files only when they are missing, unless
    ``overwrite=True`` is passed.
    """
    pipeline_dir = Path(pipeline_dir).expanduser().resolve()
    data_dir = pipeline_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    actions: list[str] = []
    arrays_path = data_dir / "pipeline_arrays.npz"
    catalog_path = data_dir / "floe_catalog.csv"

    if overwrite or not arrays_path.exists():
        label_map, binary_mask, method_name, source_label_path = _load_selected_label_map(pipeline_dir, selected_method)
        np.savez_compressed(
            arrays_path,
            accepted_label_map=label_map.astype(np.int32),
            final_accepted_label_map=label_map.astype(np.int32),
            accepted_binary_mask=binary_mask.astype(np.uint8),
            binary_mask=binary_mask.astype(np.uint8),
            selected_method=str(method_name),
        )
        actions.append(f"wrote {arrays_path} from {source_label_path}")
    else:
        actions.append(f"kept existing {arrays_path}")

    needs_catalog_rewrite = overwrite or not catalog_path.exists() or not _catalog_has_cover_schema(catalog_path)
    if needs_catalog_rewrite:
        if selected_method:
            source_catalog = data_dir / "catalogs" / f"{selected_method}.csv"
        else:
            source_catalog = data_dir / "selected_floe_catalog.csv"
        if not source_catalog.exists():
            available = sorted(p.name for p in (data_dir / "catalogs").glob("*.csv"))
            raise FileNotFoundError(
                f"Could not find {source_catalog}. Available method catalogs: {available}. "
                "Run examples/segment_sea_ice_fair_compare.py first, or pass --selected-method."
            )
        catalog_meta = _normalise_catalog_for_cover_builder(
            source_catalog,
            catalog_path,
            pipeline_dir=pipeline_dir,
        )
        actions.append(f"wrote normalized {catalog_path} from {source_catalog}")
    else:
        catalog_meta = {"catalog_path": catalog_path, "schema": "cover_builder_compatible"}
        actions.append(f"kept existing compatible {catalog_path}")

    return {
        "pipeline_dir": pipeline_dir,
        "arrays_path": arrays_path,
        "catalog_path": catalog_path,
        "selected_method": selected_method,
        "catalog_meta": catalog_meta,
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build solver-ready cover NPZs from SAM sea-ice outputs.")
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help=(
            "Segmentation output folder. The new default is outputs/sea_ice_segmentation. "
            "The script accepts the clean layout produced by segment_sea_ice_fair_compare.py."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/cover_models"))
    parser.add_argument(
        "--selected-method",
        default=None,
        help=(
            "Optional method name to use instead of data/selected_floe_labels.npz, "
            "for example sam_auto, morphology, or kmeans_watershed."
        ),
    )
    parser.add_argument("--direct-source", default="accepted_label_map")
    parser.add_argument("--crop-x0", type=int, default=256 * 3)
    parser.add_argument("--crop-y0", type=int, default=256)
    parser.add_argument("--crop-w", type=int, default=256 * 3)
    parser.add_argument("--crop-h", type=int, default=256 * 3)
    parser.add_argument("--smooth-width-px", type=float, default=2.0)
    parser.add_argument("--N", type=int, default=240)
    parser.add_argument("--M", type=int, default=120)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--overwrite-compat",
        action="store_true",
        help="Rewrite data/pipeline_arrays.npz and data/floe_catalog.csv from the selected clean outputs.",
    )
    args = parser.parse_args()

    compatibility = ensure_cover_builder_inputs(
        args.pipeline_dir,
        selected_method=args.selected_method,
        overwrite=args.overwrite_compat,
    )

    meta = build_cover_npzs_from_sam_pipeline(
        pipeline_dir=args.pipeline_dir,
        out_dir=args.out,
        direct_source=args.direct_source,
        crop_x0=args.crop_x0,
        crop_y0=args.crop_y0,
        crop_w=args.crop_w,
        crop_h=args.crop_h,
        smooth_width_px=args.smooth_width_px,
        N=args.N,
        M=args.M,
        write_full_binary_exports=True,
        write_preview=not args.no_preview,
    )
    meta["compatibility_inputs"] = compatibility

    summary_path = args.out / "cover_build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(json_ready(meta), indent=2), encoding="utf-8")

    print("[cover inputs]")
    for action in compatibility["actions"]:
        print(f"  - {action}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()