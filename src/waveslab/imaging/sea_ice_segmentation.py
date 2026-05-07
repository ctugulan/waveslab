from __future__ import annotations

"""
Fair-comparison pipeline for pancake-floe segmentation.

Goal
----
Use one common input preparation step and then compare:
  1. baseline_sam_auto: SAM automatic mask generation only
  2. alberello_threshold_morph: rectified-image threshold + morphology baseline
  3. kmeans_distance_watershed: k-means ice extraction + distance-transform splitting baseline

This deliberately removes the optional/adaptive pieces from the larger exploratory
pipeline: no illumination-correction branch selection, no CLAHE/SAM-input ablation,
no candidate-guided prompted SAM pass, no heuristic "best variant" selection,
and no manual post-hoc method selection.

The only shared preprocessing is geometric image preparation plus a common
boat/artifact removal step. The latter follows the backbone script: SAM is used
with two explicit box prompts to mask non-floe regions before all segmentation
methods see the same masked image.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
from typing import Any

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import color, measure, segmentation
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects

try:
    import cameratransform as ct
except Exception:  # pragma: no cover - optional environment dependency
    ct = None

try:
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except Exception:  # pragma: no cover - optional environment dependency
    torch = None
    sam_model_registry = None
    SamAutomaticMaskGenerator = None
    SamPredictor = None


# -----------------------------------------------------------------------------
# Edit these paths, then press play in VS Code.
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve()
SCRIPT_DIR = HERE.parent

# Use the same defaults as the exploratory script, assuming this file is placed
# beside it. You can also replace these with absolute paths.
RAW_IMAGE_PATH = SCRIPT_DIR / "2017SeaIceImage" / "2017-07-04" / "17-07-04 10-00-53.bmp"
SAM_WEIGHTS_PATH = SCRIPT_DIR / "weights" / "sam_vit_h_4b8939.pth"
OUT_DIR = SCRIPT_DIR / "results" / "pancake_fair_compare"

# The Cryosphere pancake paper reports 29 px/m after rectification.
PIXEL_SCALE_M = 1.0 / 29.0

# Set False if RAW_IMAGE_PATH already points to a rectified/calibrated image.
USE_ORTHORECTIFICATION = True

# Common non-floe masking, copied from the corrected backbone. Coordinates are
# xyxy on the orthorectified image grid. SAM is used only here with explicit box
# prompts for non-floe regions, then all methods receive the same masked image.
BOAT_BOX_XYXY = np.array([1393, 250, 2218, 1291], dtype=np.float32)
BOAT_MASK_DILATE_PX = 8
BOAT_MASK_FILL_VALUE = 0

ARTIFACT_BOX_XYXY = np.array([720, 1093, 1003, 1299], dtype=np.float32)
ARTIFACT_MASK_DILATE_PX = 0
ARTIFACT_MASK_FILL_VALUE = 0

# Shared filtering. These are objective measurement rules, not method-specific
# tuning knobs.
INTERIOR_MARGIN_PX = 6
MIN_FLOE_AREA_PX = 45
MAX_FLOE_AREA_FRAC = 0.08
MAX_ASPECT_RATIO = 5.0
MIN_FILL_RATIO = 0.20

# Alberello-style threshold + morphology baseline.
MORPH_ERODE_RADIUS_PX = 2
MORPH_DILATE_RADIUS_PX = 2
MORPH_MIN_HOLE_AREA_PX = 40

# Zhang/Skjetne-inspired automatic seed splitting baseline.
KMEANS_K = 3
WATERSHED_MIN_DISTANCE_PX = 8
WATERSHED_PEAK_REL = 0.25

# SAM automatic baseline. No prompts, no candidate boxes, no final pass.
RUN_SAM_AUTO = True
SAM_AUTO_CFG: dict[str, Any] = dict(
    points_per_side=32,
    pred_iou_thresh=0.50,
    stability_score_thresh=0.95,
    crop_n_layers=2,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)
SAM_MAX_MASK_FRAC = 0.02
SAM_MAX_OVERLAP_FRAC = 0.15

# LaTeX pipeline export. This saves the six selected methodology panels as
# individual PNG files with matching height, suitable for a compact 2 x 3 grid.
PIPELINE_EXPORT_DIRNAME = "latex_pipeline_2x3"
PIPELINE_EXPORT_METHOD = "baseline_sam_auto"  # alternatives: alberello_threshold_morph, kmeans_distance_watershed
PIPELINE_EXPORT_HEIGHT_PX = 900


@dataclass(frozen=True)
class MethodResult:
    name: str
    label_map: np.ndarray
    catalog: pd.DataFrame
    summary: dict[str, Any]


# -----------------------------------------------------------------------------
# Basic image helpers
# -----------------------------------------------------------------------------


def load_bgr(path: str | Path) -> np.ndarray:
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


# This matches the camera transform used in the exploratory pipeline.
def orthorectify_like_io_image(img_bgr: np.ndarray) -> np.ndarray:
    if ct is None:
        raise RuntimeError(
            "cameratransform is not installed. Either install it or set "
            "USE_ORTHORECTIFICATION=False and provide an already-rectified image."
        )

    fx, fy = 1453.86, 1448.71
    cx, cy = 1234.44, 1011.30
    elev_m, tilt_deg, roll_deg = 24, 75, 5
    extent = [-65, 46, 20, 85]
    scale_m_per_px = 0.05

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    proj = ct.RectilinearProjection(
        focallength_x_px=fx,
        focallength_y_px=fy,
        center_x_px=cx,
        center_y_px=cy,
        image=rgb,
    )
    orient = ct.SpatialOrientation(
        elevation_m=elev_m,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
    )
    cam = ct.Camera(proj, orient)
    top_rgb = cam.getTopViewOfImage(
        image=rgb,
        extent=extent,
        scaling=scale_m_per_px,
        do_plot=False,
    )
    return cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)


def disk_kernel(radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    if radius <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    return ((x * x + y * y) <= radius * radius).astype(np.uint8)



def largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask.astype(bool), connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def set_predictor_image(predictor: Any, image_bgr: np.ndarray) -> None:
    if predictor is None:
        raise RuntimeError("SAM predictor is not available.")
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def build_prompted_mask_from_box(
    ortho_bgr: np.ndarray,
    predictor: Any,
    box_xyxy: np.ndarray,
    *,
    object_name: str,
    dilate_px: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use SAM with one box prompt and keep the highest-scoring mask."""
    set_predictor_image(predictor, ortho_bgr)
    box_xyxy = np.asarray(box_xyxy, dtype=np.float32).ravel()
    masks, scores, _ = predictor.predict(box=box_xyxy, multimask_output=True)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise RuntimeError(f"SAM did not return a valid {object_name} mask.")

    best_i = int(np.argmax(scores))
    mask = largest_component(masks[best_i].astype(bool))

    if dilate_px > 0:
        mask = ndi.binary_dilation(mask, structure=disk_kernel(dilate_px).astype(bool))

    area_px = int(np.count_nonzero(mask))
    return mask.astype(bool), {
        "object_name": object_name,
        "box_xyxy": [float(v) for v in box_xyxy],
        "mask_score": float(scores[best_i]),
        "mask_index": best_i,
        "mask_area_px": area_px,
        "mask_dilate_px": int(dilate_px),
    }


def apply_mask(img_bgr: np.ndarray, mask: np.ndarray, *, fill_value: int = 0) -> np.ndarray:
    out = img_bgr.copy()
    out[mask.astype(bool)] = int(fill_value)
    return out


def build_common_masked_input(
    ortho_bgr: np.ndarray,
    predictor: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Build boat and artifact masks with the corrected backbone box prompts."""
    boat_mask, boat_meta = build_prompted_mask_from_box(
        ortho_bgr,
        predictor,
        BOAT_BOX_XYXY,
        object_name="boat",
        dilate_px=BOAT_MASK_DILATE_PX,
    )
    boat_bgr = apply_mask(ortho_bgr, boat_mask, fill_value=BOAT_MASK_FILL_VALUE)

    artifact_mask, artifact_meta = build_prompted_mask_from_box(
        ortho_bgr,
        predictor,
        ARTIFACT_BOX_XYXY,
        object_name="artifact",
        dilate_px=ARTIFACT_MASK_DILATE_PX,
    )
    masked_bgr = apply_mask(boat_bgr, artifact_mask, fill_value=ARTIFACT_MASK_FILL_VALUE)
    invalid_mask = boat_mask.astype(bool) | artifact_mask.astype(bool)
    return boat_bgr, masked_bgr, invalid_mask, boat_mask, artifact_mask, boat_meta, artifact_meta


def _clip_box_xyxy(box_xyxy: np.ndarray, shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape_hw
    x0, y0, x1, y1 = [int(round(float(v))) for v in np.asarray(box_xyxy).ravel()]
    x0 = int(np.clip(x0, 0, max(w - 1, 0)))
    x1 = int(np.clip(x1, 0, max(w - 1, 0)))
    y0 = int(np.clip(y0, 0, max(h - 1, 0)))
    y1 = int(np.clip(y1, 0, max(h - 1, 0)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    dil = ndi.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
    ero = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    return dil ^ ero


def draw_semtransparent_mask_and_box(
    base_bgr: np.ndarray,
    mask: np.ndarray,
    box_xyxy: np.ndarray,
    *,
    color_bgr: tuple[int, int, int],
    alpha: float = 0.38,
    thickness: int = 3,
) -> np.ndarray:
    out = base_bgr.copy().astype(np.float32)
    mask_bool = mask.astype(bool)
    color = np.array(color_bgr, dtype=np.float32)
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color
    out[mask_boundary(mask_bool)] = color
    out = np.clip(out, 0, 255).astype(np.uint8)
    x0, y0, x1, y1 = _clip_box_xyxy(box_xyxy, out.shape[:2])
    cv2.rectangle(out, (x0, y0), (x1, y1), color_bgr, int(thickness))
    return out


def resize_to_height(img_bgr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h == target_h:
        return img_bgr.copy()
    scale = float(target_h) / float(max(h, 1))
    target_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (target_w, int(target_h)), interpolation=interp)


def add_panel_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    pad = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, out.shape[0] / 900.0)
    thickness = max(2, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, th + pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def save_boat_artifact_overlay_panel(
    *,
    raw_bgr: np.ndarray,
    ortho_bgr: np.ndarray,
    boat_mask: np.ndarray,
    artifact_mask: np.ndarray,
    out_path: Path,
) -> Path:
    """Save one diagnostic panel: raw image plus box-prompt masks on rectified image."""
    overlay = ortho_bgr.copy()
    overlay = draw_semtransparent_mask_and_box(
        overlay,
        boat_mask,
        BOAT_BOX_XYXY,
        color_bgr=(0, 0, 255),
        alpha=0.35,
        thickness=3,
    )
    overlay = draw_semtransparent_mask_and_box(
        overlay,
        artifact_mask,
        ARTIFACT_BOX_XYXY,
        color_bgr=(0, 210, 255),
        alpha=0.45,
        thickness=3,
    )

    raw_scaled = resize_to_height(raw_bgr, overlay.shape[0])
    raw_scaled = add_panel_label(raw_scaled, "raw input (height matched)")
    overlay = add_panel_label(overlay, "orthorectified + SAM box masks")
    sep = np.full((overlay.shape[0], 18, 3), 255, dtype=np.uint8)
    panel = np.hstack([raw_scaled, sep, overlay])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
    return out_path

def compute_regions(masked_bgr: np.ndarray, invalid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    valid = (gray > 2) & (~invalid_mask)
    valid = remove_small_objects(valid.astype(bool), min_size=MIN_FLOE_AREA_PX)
    interior = ndi.binary_erosion(valid, structure=disk_kernel(INTERIOR_MARGIN_PX).astype(bool))
    return valid.astype(bool), interior.astype(bool)


# -----------------------------------------------------------------------------
# Label-map cleanup, catalog, and plotting
# -----------------------------------------------------------------------------


def relabel_filtered(label_map: np.ndarray, valid_region: np.ndarray, interior_region: np.ndarray) -> np.ndarray:
    """Apply identical physical/measurement filters to every method."""
    h, w = label_map.shape
    total_px = h * w
    out = np.zeros_like(label_map, dtype=np.int32)
    next_label = 1

    labels = measure.label(label_map > 0, connectivity=2)
    for prop in measure.regionprops(labels):
        mask = labels == prop.label
        area = int(prop.area)
        if area < MIN_FLOE_AREA_PX:
            continue
        if area > MAX_FLOE_AREA_FRAC * total_px:
            continue
        if not np.all(mask <= valid_region):
            continue
        # Remove boundary-touching and invalid-region-touching floes.
        if not np.all(mask <= interior_region):
            continue
        minr, minc, maxr, maxc = prop.bbox
        bw = maxc - minc
        bh = maxr - minr
        if min(bw, bh) <= 0:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > MAX_ASPECT_RATIO:
            continue
        if area / float(max(1, bw * bh)) < MIN_FILL_RATIO:
            continue
        out[mask] = next_label
        next_label += 1

    return out


def catalog_from_labels(label_map: np.ndarray, pixel_scale_m: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prop in measure.regionprops(label_map):
        area_px = int(prop.area)
        area_m2 = area_px * pixel_scale_m * pixel_scale_m
        perimeter_px = float(prop.perimeter) if prop.perimeter > 0 else 0.0
        circularity = (
            float(4.0 * math.pi * area_px / (perimeter_px * perimeter_px))
            if perimeter_px > 0 else np.nan
        )
        rows.append(
            {
                "label": int(prop.label),
                "area_px": area_px,
                "area_m2": area_m2,
                "equivalent_diameter_m": float(math.sqrt(4.0 * area_m2 / math.pi)),
                "major_axis_length_m": float(prop.major_axis_length * pixel_scale_m),
                "minor_axis_length_m": float(prop.minor_axis_length * pixel_scale_m),
                "aspect_ratio": float(prop.major_axis_length / max(prop.minor_axis_length, 1e-9)),
                "eccentricity": float(prop.eccentricity),
                "circularity": circularity,
                "centroid_x_px": float(prop.centroid[1]),
                "centroid_y_px": float(prop.centroid[0]),
                "bbox_min_row": int(prop.bbox[0]),
                "bbox_min_col": int(prop.bbox[1]),
                "bbox_max_row": int(prop.bbox[2]),
                "bbox_max_col": int(prop.bbox[3]),
            }
        )
    return pd.DataFrame(rows)


def summary_from_catalog(name: str, catalog: pd.DataFrame, valid_region: np.ndarray, pixel_scale_m: float) -> dict[str, Any]:
    valid_area_m2 = float(np.count_nonzero(valid_region) * pixel_scale_m * pixel_scale_m)
    if catalog.empty:
        return {
            "method": name,
            "n_floes": 0,
            "valid_area_m2": valid_area_m2,
            "ice_area_m2": 0.0,
            "concentration_in_valid_region": 0.0,
            "mean_equivalent_diameter_m": np.nan,
            "median_equivalent_diameter_m": np.nan,
            "p10_equivalent_diameter_m": np.nan,
            "p90_equivalent_diameter_m": np.nan,
        }

    diam = catalog["equivalent_diameter_m"].to_numpy(float)
    ice_area = float(catalog["area_m2"].sum())
    return {
        "method": name,
        "n_floes": int(len(catalog)),
        "valid_area_m2": valid_area_m2,
        "ice_area_m2": ice_area,
        "concentration_in_valid_region": float(ice_area / max(valid_area_m2, 1e-12)),
        "mean_equivalent_diameter_m": float(np.mean(diam)),
        "median_equivalent_diameter_m": float(np.median(diam)),
        "p10_equivalent_diameter_m": float(np.percentile(diam, 10)),
        "p90_equivalent_diameter_m": float(np.percentile(diam, 90)),
    }


def area_value_label_image(label_map: np.ndarray, catalog: pd.DataFrame) -> np.ndarray:
    values = np.zeros(label_map.shape, dtype=np.float32)
    if catalog.empty:
        return np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for _, row in catalog.iterrows():
        label = int(row["label"])
        # Area-compressed visualization: small floes remain visible, large floes saturate.
        v = 1.0 - math.exp(-float(row["area_px"]) / 1800.0)
        values[label_map == label] = v
    rgba = plt.get_cmap("viridis")(values)
    rgb = (255 * rgba[..., :3]).astype(np.uint8)
    rgb[label_map == 0] = 0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def overlay_labels(img_bgr: np.ndarray, label_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    label_bgr = area_value_label_image(label_map, catalog_from_labels(label_map, PIXEL_SCALE_M))
    overlay = img_bgr.copy()
    mask = label_map > 0
    overlay[mask] = cv2.addWeighted(img_bgr, 1.0 - alpha, label_bgr, alpha, 0)[mask]

    contours = measure.find_contours(label_map > 0, 0.5)
    for contour in contours:
        pts = np.fliplr(contour).astype(np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
    return overlay


def save_png(path: Path, image: np.ndarray, *, bgr: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bgr:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), image.astype(np.uint8))


# -----------------------------------------------------------------------------
# Methods being compared
# -----------------------------------------------------------------------------


def method_alberello_threshold_morph(masked_bgr: np.ndarray, valid_region: np.ndarray, interior_region: np.ndarray) -> np.ndarray:
    """Rectified image -> global Otsu threshold -> clean/erode/fill/dilate/clear-border.

    This is the reproducible version of the pancake-paper morphology chain. Otsu
    replaces a hand-selected threshold so the comparison is not manually tuned.
    """
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    work = gray.copy()
    work[~valid_region] = 0

    vals = work[valid_region]
    if vals.size == 0:
        return np.zeros(gray.shape, dtype=np.int32)

    threshold_value, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (work >= float(threshold_value)) & valid_region

    binary = remove_small_objects(binary, min_size=MIN_FLOE_AREA_PX)
    binary = cv2.erode(binary.astype(np.uint8), disk_kernel(MORPH_ERODE_RADIUS_PX), iterations=1) > 0
    binary = ndi.binary_fill_holes(binary)
    binary = remove_small_holes(binary, area_threshold=MORPH_MIN_HOLE_AREA_PX)
    binary = cv2.dilate(binary.astype(np.uint8), disk_kernel(MORPH_DILATE_RADIUS_PX), iterations=1) > 0
    binary = binary & valid_region

    labels = measure.label(binary, connectivity=2)
    return relabel_filtered(labels, valid_region, interior_region)


def method_kmeans_distance_watershed(masked_bgr: np.ndarray, valid_region: np.ndarray, interior_region: np.ndarray) -> np.ndarray:
    """K-means ice extraction plus distance-transform watershed splitting.

    This is a lightweight, reproducible analogue of the traditional workflow that
    uses k-means for ice/water separation and distance-transform seeds for object
    separation. It does not implement full GVF snakes.
    """
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[valid_region].astype(np.float32).reshape(-1, 1)
    if pixels.size == 0:
        return np.zeros(gray.shape, dtype=np.int32)

    # Deterministic OpenCV k-means.
    cv2.setRNGSeed(7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, compact_labels, centers = cv2.kmeans(
        pixels,
        KMEANS_K,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.ravel()
    water_cluster = int(np.argmin(centers))

    cluster_img = np.full(gray.shape, water_cluster, dtype=np.int32)
    cluster_img[valid_region] = compact_labels.ravel().astype(np.int32)
    ice = (cluster_img != water_cluster) & valid_region
    ice = remove_small_objects(ice, min_size=MIN_FLOE_AREA_PX)
    ice = remove_small_holes(ice, area_threshold=MORPH_MIN_HOLE_AREA_PX)

    distance = ndi.distance_transform_edt(ice)
    if distance.max() <= 0:
        return np.zeros(gray.shape, dtype=np.int32)

    peaks = peak_local_max(
        distance,
        labels=ice,
        min_distance=WATERSHED_MIN_DISTANCE_PX,
        threshold_abs=WATERSHED_PEAK_REL * float(distance.max()),
        exclude_border=False,
    )
    markers = np.zeros(gray.shape, dtype=np.int32)
    for i, (rr, cc) in enumerate(peaks, start=1):
        markers[rr, cc] = i

    if markers.max() == 0:
        labels = measure.label(ice, connectivity=2)
    else:
        labels = segmentation.watershed(-distance, markers=markers, mask=ice)

    return relabel_filtered(labels, valid_region, interior_region)


def load_sam_model(weights_path: Path):
    if torch is None or sam_model_registry is None or SamAutomaticMaskGenerator is None or SamPredictor is None:
        raise RuntimeError("segment_anything and torch are not available in this environment.")
    if not weights_path.exists():
        raise FileNotFoundError(f"SAM weights not found: {weights_path}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=str(weights_path))
    sam.to(device=torch.device(device))
    sam.eval()
    predictor = SamPredictor(sam)
    return sam, predictor, device


def method_baseline_sam_auto(
    masked_bgr: np.ndarray,
    valid_region: np.ndarray,
    interior_region: np.ndarray,
    *,
    sam_model: Any | None = None,
    device: str | None = None,
) -> np.ndarray:
    if sam_model is None:
        sam_model, _, device = load_sam_model(SAM_WEIGHTS_PATH)
    print(f"[sam] running automatic mask generator on {device or 'unknown device'}")
    generator = SamAutomaticMaskGenerator(sam_model, **SAM_AUTO_CFG)
    raw_masks = generator.generate(cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB))

    h, w = valid_region.shape
    total_px = h * w
    occupied = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=np.int32)
    next_label = 1

    def sort_key(m: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(m.get("predicted_iou", 0.0)),
            float(m.get("stability_score", 0.0)),
            float(m.get("area", 0.0)),
        )

    for m in sorted(raw_masks, key=sort_key, reverse=True):
        seg = m["segmentation"].astype(bool)
        seg = seg & valid_region
        area = int(np.count_nonzero(seg))
        if area < MIN_FLOE_AREA_PX:
            continue
        if area > SAM_MAX_MASK_FRAC * total_px:
            continue
        if not np.all(seg <= interior_region):
            continue
        overlap = np.count_nonzero(seg & occupied) / float(max(area, 1))
        if overlap > SAM_MAX_OVERLAP_FRAC:
            continue
        seg = seg & (~occupied)
        seg = remove_small_objects(seg, min_size=MIN_FLOE_AREA_PX)
        if np.count_nonzero(seg) < MIN_FLOE_AREA_PX:
            continue
        out[seg] = next_label
        occupied |= seg
        next_label += 1

    return relabel_filtered(out, valid_region, interior_region)


# -----------------------------------------------------------------------------
# Saving and orchestration
# -----------------------------------------------------------------------------


def run_method(name: str, label_map: np.ndarray, valid_region: np.ndarray) -> MethodResult:
    catalog = catalog_from_labels(label_map, PIXEL_SCALE_M)
    summary = summary_from_catalog(name, catalog, valid_region, PIXEL_SCALE_M)
    return MethodResult(name=name, label_map=label_map, catalog=catalog, summary=summary)


def select_pipeline_export_result(results: list[MethodResult], preferred_method: str) -> MethodResult | None:
    """Pick the method used for the representative six-panel pipeline export."""
    if not results:
        return None
    for result in results:
        if result.name == preferred_method:
            return result
    # If SAM was skipped or the preferred method was not run, use the last
    # successful method so the script still produces a complete figure set.
    return results[-1]


def save_resized_png(path: Path, image_bgr: np.ndarray, *, target_height_px: int) -> str:
    """Save an image after resizing to a common height while preserving aspect ratio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = resize_to_height(image_bgr, target_height_px)
    cv2.imwrite(str(path), out)
    return str(path)


def binary_mask_image(label_map: np.ndarray) -> np.ndarray:
    """Return a clean black/white binary floe mask as a 3-channel image."""
    binary = ((label_map > 0).astype(np.uint8) * 255)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def save_latex_pipeline_steps(
    *,
    raw_bgr: np.ndarray,
    ortho_bgr: np.ndarray,
    boat_mask: np.ndarray,
    artifact_mask: np.ndarray,
    masked_bgr: np.ndarray,
    result: MethodResult,
    out_dir: Path,
    target_height_px: int = PIPELINE_EXPORT_HEIGHT_PX,
) -> dict[str, Any]:
    """Save the six representative pipeline panels for a LaTeX 2 x 3 grid.

    Panels:
      00 raw image input
      01 orthorectified image
      02 boat/artifact masking overlay
      03 common masked segmentation input
      04 floe instance overlay for the selected method
      05 final binary floe mask for the selected method
    """
    pipeline_dir = out_dir / PIPELINE_EXPORT_DIRNAME
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    mask_overlay = ortho_bgr.copy()
    mask_overlay = draw_semtransparent_mask_and_box(
        mask_overlay,
        boat_mask,
        BOAT_BOX_XYXY,
        color_bgr=(0, 0, 255),
        alpha=0.35,
        thickness=3,
    )
    mask_overlay = draw_semtransparent_mask_and_box(
        mask_overlay,
        artifact_mask,
        ARTIFACT_BOX_XYXY,
        color_bgr=(0, 210, 255),
        alpha=0.45,
        thickness=3,
    )

    instance_overlay = overlay_labels(masked_bgr, result.label_map)
    binary_output = binary_mask_image(result.label_map)

    files = {
        "a_raw_image": save_resized_png(
            pipeline_dir / "00_raw_image_input.png",
            raw_bgr,
            target_height_px=target_height_px,
        ),
        "b_orthorectified": save_resized_png(
            pipeline_dir / "01_orthorectified_input.png",
            ortho_bgr,
            target_height_px=target_height_px,
        ),
        "c_non_floe_masking": save_resized_png(
            pipeline_dir / "02_non_floe_masking_overlay.png",
            mask_overlay,
            target_height_px=target_height_px,
        ),
        "d_common_masked_input": save_resized_png(
            pipeline_dir / "03_common_masked_input.png",
            masked_bgr,
            target_height_px=target_height_px,
        ),
        "e_floe_instances": save_resized_png(
            pipeline_dir / "04_floe_instance_overlay.png",
            instance_overlay,
            target_height_px=target_height_px,
        ),
        "f_binary_mask": save_resized_png(
            pipeline_dir / "05_binary_floe_mask.png",
            binary_output,
            target_height_px=target_height_px,
        ),
    }

    manifest = {
        "selected_method": result.name,
        "target_height_px": int(target_height_px),
        "panel_order": [
            ["(a)", "Raw image", files["a_raw_image"]],
            ["(b)", "Orthorectified view", files["b_orthorectified"]],
            ["(c)", "Non-floe masking", files["c_non_floe_masking"]],
            ["(d)", "Masked input", files["d_common_masked_input"]],
            ["(e)", "Floe instances", files["e_floe_instances"]],
            ["(f)", "Binary mask", files["f_binary_mask"]],
        ],
        "files": files,
    }
    with open(pipeline_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[latex] wrote six pipeline panels to {pipeline_dir}")
    print(f"[latex] selected method for panels (e)-(f): {result.name}")
    return manifest


def save_method_result(result: MethodResult, masked_bgr: np.ndarray, out_dir: Path) -> dict[str, str]:
    method_dir = out_dir / result.name
    method_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = method_dir / "floe_catalog.csv"
    summary_path = method_dir / "summary.json"
    labels_path = method_dir / "labels.npz"
    label_png_path = method_dir / "area_coloured_labels.png"
    overlay_path = method_dir / "overlay.png"

    result.catalog.to_csv(catalog_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result.summary, f, indent=2)
    np.savez_compressed(labels_path, label_map=result.label_map.astype(np.int32))
    save_png(label_png_path, area_value_label_image(result.label_map, result.catalog))
    save_png(overlay_path, overlay_labels(masked_bgr, result.label_map))

    return {
        "catalog": str(catalog_path),
        "summary": str(summary_path),
        "labels": str(labels_path),
        "area_coloured_labels": str(label_png_path),
        "overlay": str(overlay_path),
    }


def save_comparison_panel(results: list[MethodResult], masked_bgr: np.ndarray, out_path: Path) -> None:
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    axes[0].imshow(cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("common input")
    axes[0].set_axis_off()

    for ax, result in zip(axes[1:], results):
        overlay = overlay_labels(masked_bgr, result.label_map)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{result.name}\nn={result.summary['n_floes']}")
        ax.set_axis_off()

    fig.tight_layout(pad=0.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_bgr(RAW_IMAGE_PATH)
    if USE_ORTHORECTIFICATION:
        ortho = orthorectify_like_io_image(raw)
    else:
        ortho = raw

    sam_model, predictor, device = load_sam_model(SAM_WEIGHTS_PATH)
    boat_bgr, masked_bgr, invalid_mask, boat_mask, artifact_mask, boat_meta, artifact_meta = build_common_masked_input(
        ortho,
        predictor,
    )
    valid_region, interior_region = compute_regions(masked_bgr, invalid_mask)

    mask_panel_path = save_boat_artifact_overlay_panel(
        raw_bgr=raw,
        ortho_bgr=ortho,
        boat_mask=boat_mask,
        artifact_mask=artifact_mask,
        out_path=OUT_DIR / "01_boat_artifact_box_mask_overlay_panel.png",
    )

    save_png(OUT_DIR / "00_common_orthorectified_input.png", ortho)
    save_png(OUT_DIR / "01a_boat_masked_input.png", boat_bgr)
    save_png(OUT_DIR / "01_common_masked_input.png", masked_bgr)
    save_png(OUT_DIR / "02_valid_region.png", (valid_region.astype(np.uint8) * 255), bgr=False)
    save_png(OUT_DIR / "03_interior_region.png", (interior_region.astype(np.uint8) * 255), bgr=False)

    results: list[MethodResult] = []
    outputs: dict[str, Any] = {}

    print("[method] alberello_threshold_morph")
    labels = method_alberello_threshold_morph(masked_bgr, valid_region, interior_region)
    result = run_method("alberello_threshold_morph", labels, valid_region)
    results.append(result)
    outputs[result.name] = save_method_result(result, masked_bgr, OUT_DIR)

    print("[method] kmeans_distance_watershed")
    labels = method_kmeans_distance_watershed(masked_bgr, valid_region, interior_region)
    result = run_method("kmeans_distance_watershed", labels, valid_region)
    results.append(result)
    outputs[result.name] = save_method_result(result, masked_bgr, OUT_DIR)

    if RUN_SAM_AUTO:
        try:
            print("[method] baseline_sam_auto")
            labels = method_baseline_sam_auto(
                masked_bgr,
                valid_region,
                interior_region,
                sam_model=sam_model,
                device=device,
            )
            result = run_method("baseline_sam_auto", labels, valid_region)
            results.append(result)
            outputs[result.name] = save_method_result(result, masked_bgr, OUT_DIR)
        except Exception as exc:
            outputs["baseline_sam_auto_error"] = repr(exc)
            print(f"[warn] baseline_sam_auto skipped: {exc}")

    pipeline_result = select_pipeline_export_result(results, PIPELINE_EXPORT_METHOD)
    pipeline_manifest: dict[str, Any] | None = None
    if pipeline_result is not None:
        pipeline_manifest = save_latex_pipeline_steps(
            raw_bgr=raw,
            ortho_bgr=ortho,
            boat_mask=boat_mask,
            artifact_mask=artifact_mask,
            masked_bgr=masked_bgr,
            result=pipeline_result,
            out_dir=OUT_DIR,
            target_height_px=PIPELINE_EXPORT_HEIGHT_PX,
        )
        outputs["latex_pipeline_2x3"] = pipeline_manifest

    summary_df = pd.DataFrame([r.summary for r in results])
    summary_path = OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    comparison_path = OUT_DIR / "comparison_panel.png"
    save_comparison_panel(results, masked_bgr, comparison_path)

    config = {
        "raw_image_path": str(RAW_IMAGE_PATH),
        "sam_weights_path": str(SAM_WEIGHTS_PATH),
        "pixel_scale_m": PIXEL_SCALE_M,
        "use_orthorectification": USE_ORTHORECTIFICATION,
        "common_masking": {
            "source": "SAM box-prompt masks from corrected backbone",
            "device": str(device),
            "boat": boat_meta,
            "artifact": artifact_meta,
            "boat_box_xyxy": [float(v) for v in BOAT_BOX_XYXY],
            "artifact_box_xyxy": [float(v) for v in ARTIFACT_BOX_XYXY],
            "diagnostic_panel": str(mask_panel_path),
        },
        "shared_filters": {
            "interior_margin_px": INTERIOR_MARGIN_PX,
            "min_floe_area_px": MIN_FLOE_AREA_PX,
            "max_floe_area_frac": MAX_FLOE_AREA_FRAC,
            "max_aspect_ratio": MAX_ASPECT_RATIO,
            "min_fill_ratio": MIN_FILL_RATIO,
        },
        "alberello_threshold_morph": {
            "threshold": "Otsu on valid grayscale pixels",
            "erode_radius_px": MORPH_ERODE_RADIUS_PX,
            "dilate_radius_px": MORPH_DILATE_RADIUS_PX,
            "min_hole_area_px": MORPH_MIN_HOLE_AREA_PX,
        },
        "kmeans_distance_watershed": {
            "kmeans_k": KMEANS_K,
            "watershed_min_distance_px": WATERSHED_MIN_DISTANCE_PX,
            "watershed_peak_rel": WATERSHED_PEAK_REL,
        },
        "baseline_sam_auto": {
            "run": RUN_SAM_AUTO,
            "auto_cfg": SAM_AUTO_CFG,
            "max_mask_frac": SAM_MAX_MASK_FRAC,
            "max_overlap_frac": SAM_MAX_OVERLAP_FRAC,
            "note": "SAM automatic mask generation only; no prompts or candidate-map matching.",
        },
        "outputs": outputs,
        "mask_prompt_panel": str(mask_panel_path),
        "summary_csv": str(summary_path),
        "comparison_panel": str(comparison_path),
    }
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[done] wrote comparison to {OUT_DIR}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
    