from __future__ import annotations

"""
Clean thesis-output pipeline for one shipborne sea-ice image.

This script keeps the workflow used in the thesis chapter, but writes a much
smaller and more predictable set of files:

outputs/sea_ice_segmentation/
  figures/
    pipeline/       # panels for the 2 x 3 processing figure
    sensitivity/    # method-comparison masks
  data/             # reusable masks, label maps, catalogs, and summary table
  metadata/         # run configuration and notes

The script intentionally avoids writing every intermediate image to the root
output directory. Extra debugging images are only written with --save-diagnostics.

CUDA_VISIBLE_DEVICES=0 python examples/segment_sea_ice_fair_compare.py \
  --image "examples/2017SeaIceImage/2017-07-04/17-07-04 10-00-53.bmp" \
  --sam-weights "examples/weights/sam_vit_h_4b8939.pth" \
  --out outputs/sea_ice_segmentation \
  --selected-method sam_auto \
  --save-diagnostics
"""

import argparse
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, segmentation
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects

try:
    import cameratransform as ct
except Exception:  # pragma: no cover - optional dependency
    ct = None

try:
    import torch
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
except Exception:  # pragma: no cover - optional dependency
    torch = None
    SamAutomaticMaskGenerator = None
    SamPredictor = None
    sam_model_registry = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "examples" else Path.cwd()

DEFAULT_IMAGE = SCRIPT_DIR / "2017SeaIceImage" / "2017-07-04" / "17-07-04 10-00-53.bmp"
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "sam_vit_h_4b8939.pth"
if not DEFAULT_WEIGHTS.exists():
    DEFAULT_WEIGHTS = SCRIPT_DIR / "weights" / "sam_vit_h_4b8939.pth"
DEFAULT_OUT = REPO_ROOT / "outputs" / "sea_ice_segmentation"

# Thesis camera setup for the orthorectified top view.
CAMERA_INTRINSICS = dict(fx=1453.86, fy=1448.71, cx=1234.44, cy=1011.30)
CAMERA_EXTRINSICS = dict(elevation_m=24.0, tilt_deg=75.0, roll_deg=5.0)
ORTHO_EXTENT_M = [-65.0, 46.0, 20.0, 85.0]
ORTHO_PIXEL_SIZE_M = 0.05

# Non-floe regions on the orthorectified image, xyxy pixel coordinates.
BOAT_BOX_XYXY = np.array([1393, 250, 2218, 1291], dtype=np.float32)
ARTIFACT_BOX_XYXY = np.array([720, 1093, 1003, 1299], dtype=np.float32)
BOAT_MASK_DILATE_PX = 8
ARTIFACT_MASK_DILATE_PX = 0

# Shared filtering rules. These are applied to all methods after segmentation.
INTERIOR_MARGIN_PX = 6
MIN_FLOE_AREA_PX = 45
MAX_FLOE_AREA_FRAC = 0.08
MAX_ASPECT_RATIO = 5.0
MIN_FILL_RATIO = 0.20

# Classical morphology baseline.
MORPH_ERODE_RADIUS_PX = 2
MORPH_DILATE_RADIUS_PX = 2
MORPH_MIN_HOLE_AREA_PX = 40

# k-means + distance-watershed baseline.
KMEANS_K = 3
WATERSHED_MIN_DISTANCE_PX = 8
WATERSHED_PEAK_REL = 0.25

# SAM automatic baseline.
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

# Stricter automatic SAM setting used for the sensitivity panel.
SAM_AUTO_STRICT_CFG: dict[str, Any] = dict(
    points_per_side=32,
    pred_iou_thresh=0.70,
    stability_score_thresh=0.97,
    crop_n_layers=2,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=150,
)
SAM_AUTO_STRICT_MAX_MASK_FRAC = 0.015
SAM_AUTO_STRICT_MAX_OVERLAP_FRAC = 0.05

# Candidate-guided prompted SAM cases. Classical candidate regions are generated
# first, then each candidate is refined using a SAM box plus one positive point.
PROMPT_MAX_CANDIDATES = 500
PROMPT_PAD_PX = 4
PROMPT_MIN_IOU_WITH_CANDIDATE = 0.08
PROMPT_MAX_MASK_FRAC = 0.025
PROMPT_MAX_OVERLAP_FRAC = 0.08

SENSITIVITY_ORDER = [
    "morphology",
    "kmeans_watershed",
    "sam_auto",
    "sam_auto_strict",
    "prompted_illumination",
    "prompted_clahe",
]

SENSITIVITY_LABELS = {
    "morphology": ("a Morphology", "a_morphology.png"),
    "kmeans_watershed": ("b k-means + watershed", "b_kmeans_watershed.png"),
    "sam_auto": ("c SAM auto", "c_sam_auto.png"),
    "sam_auto_strict": ("d Strict SAM", "d_strict_sam.png"),
    "prompted_illumination": ("e Prompted, corrected", "e_prompted_corrected.png"),
    "prompted_clahe": ("f Prompted, alternate", "f_prompted_alternate.png"),
}


@dataclass(frozen=True)
class MethodResult:
    name: str
    label_map: np.ndarray
    catalog: pd.DataFrame
    summary: dict[str, Any]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_bgr(path: Path, image_bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")
    return str(path)


def write_gray(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image.astype(np.uint8))
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")
    return str(path)


def resize_to_height(image_bgr: np.ndarray, height: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h == height:
        return image_bgr.copy()
    scale = height / max(h, 1)
    width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (width, height), interpolation=interpolation)


def disk_kernel(radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    if radius == 0:
        return np.ones((1, 1), dtype=np.uint8)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((x * x + y * y) <= radius * radius).astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask.astype(bool), connectivity=2)
    if labels.max() == 0:
        return np.zeros(mask.shape, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def safe_axis_lengths(prop: Any) -> tuple[float, float]:
    major = getattr(prop, "axis_major_length", None)
    minor = getattr(prop, "axis_minor_length", None)
    if major is None:
        major = prop.major_axis_length
    if minor is None:
        minor = prop.minor_axis_length
    return float(major), float(minor)


_HAS_SKIMAGE_MAX_SIZE_OBJECTS = "max_size" in inspect.signature(remove_small_objects).parameters
_HAS_SKIMAGE_MAX_SIZE_HOLES = "max_size" in inspect.signature(remove_small_holes).parameters


def remove_objects_smaller_than(mask: np.ndarray, min_size_px: int) -> np.ndarray:
    """Remove connected components with area < min_size_px.

    scikit-image 0.26 renamed the threshold argument from ``min_size`` to
    ``max_size`` and changed the comparison from ``<`` to ``<=``.  Subtracting
    one keeps the historical behaviour used by this thesis script.
    """
    min_size_px = int(min_size_px)
    if _HAS_SKIMAGE_MAX_SIZE_OBJECTS:
        return remove_small_objects(mask.astype(bool), max_size=max(min_size_px - 1, 0))
    return remove_small_objects(mask.astype(bool), min_size=min_size_px)


def fill_holes_smaller_than(mask: np.ndarray, min_hole_area_px: int) -> np.ndarray:
    """Fill holes with area < min_hole_area_px, across scikit-image versions."""
    min_hole_area_px = int(min_hole_area_px)
    if _HAS_SKIMAGE_MAX_SIZE_HOLES:
        return remove_small_holes(mask.astype(bool), max_size=max(min_hole_area_px - 1, 0))
    return remove_small_holes(mask.astype(bool), area_threshold=min_hole_area_px)


# -----------------------------------------------------------------------------
# Image preparation
# -----------------------------------------------------------------------------


def orthorectify(raw_bgr: np.ndarray) -> np.ndarray:
    if ct is None:
        raise RuntimeError(
            "cameratransform is not installed. Install it or use --no-orthorectify "
            "with an already-rectified image."
        )

    rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    projection = ct.RectilinearProjection(
        focallength_x_px=CAMERA_INTRINSICS["fx"],
        focallength_y_px=CAMERA_INTRINSICS["fy"],
        center_x_px=CAMERA_INTRINSICS["cx"],
        center_y_px=CAMERA_INTRINSICS["cy"],
        image=rgb,
    )
    orientation = ct.SpatialOrientation(**CAMERA_EXTRINSICS)
    camera = ct.Camera(projection, orientation)
    top_rgb = camera.getTopViewOfImage(
        image=rgb,
        extent=ORTHO_EXTENT_M,
        scaling=ORTHO_PIXEL_SIZE_M,
        do_plot=False,
    )
    return cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)


def load_sam(weights_path: Path) -> tuple[Any, Any, str]:
    if torch is None or sam_model_registry is None or SamPredictor is None:
        raise RuntimeError("torch and segment_anything are required for this script.")
    if not weights_path.exists():
        raise FileNotFoundError(f"SAM weights not found: {weights_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry["vit_h"](checkpoint=str(weights_path))
    model.to(device=torch.device(device))
    model.eval()
    return model, SamPredictor(model), device


def prompted_box_mask(
    image_bgr: np.ndarray,
    predictor: Any,
    box_xyxy: np.ndarray,
    *,
    name: str,
    dilate_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(box=np.asarray(box_xyxy, dtype=np.float32), multimask_output=True)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise RuntimeError(f"SAM did not return a valid mask for {name}.")

    best = int(np.argmax(scores))
    mask = largest_component(masks[best].astype(bool))
    if dilate_px > 0:
        mask = ndi.binary_dilation(mask, structure=disk_kernel(dilate_px).astype(bool))

    return mask.astype(bool), {
        "name": name,
        "box_xyxy": [float(v) for v in np.asarray(box_xyxy).ravel()],
        "sam_score": float(scores[best]),
        "sam_mask_index": best,
        "dilate_px": int(dilate_px),
        "area_px": int(np.count_nonzero(mask)),
    }


def apply_black_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    out[mask.astype(bool)] = 0
    return out


def prepare_common_input(
    ortho_bgr: np.ndarray,
    predictor: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    boat_mask, boat_meta = prompted_box_mask(
        ortho_bgr,
        predictor,
        BOAT_BOX_XYXY,
        name="ship/deck",
        dilate_px=BOAT_MASK_DILATE_PX,
    )
    image_without_boat = apply_black_mask(ortho_bgr, boat_mask)

    artifact_mask, artifact_meta = prompted_box_mask(
        ortho_bgr,
        predictor,
        ARTIFACT_BOX_XYXY,
        name="foreground artifact",
        dilate_px=ARTIFACT_MASK_DILATE_PX,
    )
    masked_bgr = apply_black_mask(image_without_boat, artifact_mask)
    invalid_mask = boat_mask | artifact_mask

    meta = {"boat": boat_meta, "artifact": artifact_meta}
    return image_without_boat, masked_bgr, invalid_mask, boat_mask, artifact_mask, meta


def compute_regions(masked_bgr: np.ndarray, invalid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    valid = (gray > 2) & (~invalid_mask)
    valid = remove_objects_smaller_than(valid.astype(bool), MIN_FLOE_AREA_PX)
    interior = ndi.binary_erosion(valid, structure=disk_kernel(INTERIOR_MARGIN_PX).astype(bool))
    return valid.astype(bool), interior.astype(bool)


# -----------------------------------------------------------------------------
# Segmentation methods
# -----------------------------------------------------------------------------


def relabel_filtered(label_map: np.ndarray, valid_region: np.ndarray, interior_region: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    total_px = h * w
    labels = measure.label(label_map > 0, connectivity=2)
    out = np.zeros_like(labels, dtype=np.int32)
    next_label = 1

    for prop in measure.regionprops(labels):
        mask = labels == prop.label
        area = int(prop.area)
        if area < MIN_FLOE_AREA_PX:
            continue
        if area > MAX_FLOE_AREA_FRAC * total_px:
            continue
        if not np.all(mask <= valid_region):
            continue
        if not np.all(mask <= interior_region):
            continue

        minr, minc, maxr, maxc = prop.bbox
        box_h = maxr - minr
        box_w = maxc - minc
        if min(box_h, box_w) <= 0:
            continue
        if max(box_h, box_w) / max(1, min(box_h, box_w)) > MAX_ASPECT_RATIO:
            continue
        if area / float(max(1, box_h * box_w)) < MIN_FILL_RATIO:
            continue

        out[mask] = next_label
        next_label += 1

    return out


def segment_morphology(masked_bgr: np.ndarray, valid_region: np.ndarray, interior_region: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    work = gray.copy()
    work[~valid_region] = 0
    values = work[valid_region]
    if values.size == 0:
        return np.zeros(gray.shape, dtype=np.int32)

    threshold_value, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ice = (work >= float(threshold_value)) & valid_region
    ice = remove_objects_smaller_than(ice, MIN_FLOE_AREA_PX)
    ice = cv2.erode(ice.astype(np.uint8), disk_kernel(MORPH_ERODE_RADIUS_PX), iterations=1) > 0
    ice = ndi.binary_fill_holes(ice)
    ice = fill_holes_smaller_than(ice, MORPH_MIN_HOLE_AREA_PX)
    ice = cv2.dilate(ice.astype(np.uint8), disk_kernel(MORPH_DILATE_RADIUS_PX), iterations=1) > 0
    labels = measure.label(ice & valid_region, connectivity=2)
    return relabel_filtered(labels, valid_region, interior_region)


def segment_kmeans_watershed(masked_bgr: np.ndarray, valid_region: np.ndarray, interior_region: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[valid_region].astype(np.float32).reshape(-1, 1)
    if pixels.size == 0:
        return np.zeros(gray.shape, dtype=np.int32)

    cv2.setRNGSeed(7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, compact_labels, centers = cv2.kmeans(pixels, KMEANS_K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    water_cluster = int(np.argmin(centers.ravel()))

    cluster_image = np.full(gray.shape, water_cluster, dtype=np.int32)
    cluster_image[valid_region] = compact_labels.ravel().astype(np.int32)
    ice = (cluster_image != water_cluster) & valid_region
    ice = remove_objects_smaller_than(ice, MIN_FLOE_AREA_PX)
    ice = fill_holes_smaller_than(ice, MORPH_MIN_HOLE_AREA_PX)

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
    for i, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = i

    if markers.max() == 0:
        labels = measure.label(ice, connectivity=2)
    else:
        labels = segmentation.watershed(-distance, markers=markers, mask=ice)
    return relabel_filtered(labels, valid_region, interior_region)


def segment_sam_auto_cfg(
    image_bgr: np.ndarray,
    valid_region: np.ndarray,
    interior_region: np.ndarray,
    *,
    sam_model: Any,
    device: str,
    auto_cfg: dict[str, Any],
    max_mask_frac: float,
    max_overlap_frac: float,
    label: str,
) -> np.ndarray:
    if SamAutomaticMaskGenerator is None:
        raise RuntimeError("segment_anything is not available.")

    print(f"[sam] {label} automatic mask generation on {device}")
    generator = SamAutomaticMaskGenerator(sam_model, **auto_cfg)
    masks = generator.generate(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    h, w = valid_region.shape
    total_px = h * w
    occupied = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=np.int32)
    next_label = 1

    def quality_key(mask_dict: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(mask_dict.get("predicted_iou", 0.0)),
            float(mask_dict.get("stability_score", 0.0)),
            float(mask_dict.get("area", 0.0)),
        )

    for mask_dict in sorted(masks, key=quality_key, reverse=True):
        seg = mask_dict["segmentation"].astype(bool) & valid_region
        area = int(np.count_nonzero(seg))
        if area < MIN_FLOE_AREA_PX:
            continue
        if area > float(max_mask_frac) * total_px:
            continue
        if not np.all(seg <= interior_region):
            continue
        if np.count_nonzero(seg & occupied) / float(max(area, 1)) > float(max_overlap_frac):
            continue
        seg = remove_objects_smaller_than(seg & (~occupied), MIN_FLOE_AREA_PX)
        if np.count_nonzero(seg) < MIN_FLOE_AREA_PX:
            continue
        out[seg] = next_label
        occupied |= seg
        next_label += 1

    return relabel_filtered(out, valid_region, interior_region)


def segment_sam_auto(
    masked_bgr: np.ndarray,
    valid_region: np.ndarray,
    interior_region: np.ndarray,
    *,
    sam_model: Any,
    device: str,
) -> np.ndarray:
    return segment_sam_auto_cfg(
        masked_bgr,
        valid_region,
        interior_region,
        sam_model=sam_model,
        device=device,
        auto_cfg=SAM_AUTO_CFG,
        max_mask_frac=SAM_MAX_MASK_FRAC,
        max_overlap_frac=SAM_MAX_OVERLAP_FRAC,
        label="default",
    )


def segment_sam_auto_strict(
    masked_bgr: np.ndarray,
    valid_region: np.ndarray,
    interior_region: np.ndarray,
    *,
    sam_model: Any,
    device: str,
) -> np.ndarray:
    return segment_sam_auto_cfg(
        masked_bgr,
        valid_region,
        interior_region,
        sam_model=sam_model,
        device=device,
        auto_cfg=SAM_AUTO_STRICT_CFG,
        max_mask_frac=SAM_AUTO_STRICT_MAX_MASK_FRAC,
        max_overlap_frac=SAM_AUTO_STRICT_MAX_OVERLAP_FRAC,
        label="strict",
    )


def illumination_correct_bgr(image_bgr: np.ndarray, valid_region: np.ndarray, *, blur_sigma: float = 45.0) -> np.ndarray:
    """Flatten slow illumination changes without changing the image geometry."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    valid = valid_region.astype(bool)
    fill_value = float(np.median(gray[valid])) if np.any(valid) else float(np.median(gray))

    work = gray.copy()
    work[~valid] = fill_value
    ksize = int(max(3, 2 * round(blur_sigma * 2) + 1))
    background = cv2.GaussianBlur(work, (ksize, ksize), blur_sigma)
    corrected = (work / np.maximum(background, 1.0)) * fill_value
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected[~valid] = 0
    return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)


def clahe_enhance_bgr(
    image_bgr: np.ndarray,
    valid_region: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Local contrast enhancement used as the alternate prompted-SAM input."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l_out = clahe.apply(l_chan)
    out = cv2.cvtColor(cv2.merge([l_out, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    out[~valid_region.astype(bool)] = 0
    return out


def candidate_labels_for_prompting(
    candidate_bgr: np.ndarray,
    valid_region: np.ndarray,
    interior_region: np.ndarray,
) -> np.ndarray:
    """Classical proposal map used only to define local SAM prompts."""
    return segment_kmeans_watershed(candidate_bgr, valid_region, interior_region)


def padded_bbox_from_region(
    bbox: tuple[int, int, int, int],
    shape_hw: tuple[int, int],
    *,
    pad_px: int,
) -> np.ndarray:
    minr, minc, maxr, maxc = bbox
    h, w = shape_hw
    x0 = max(0, int(minc) - int(pad_px))
    y0 = max(0, int(minr) - int(pad_px))
    x1 = min(w - 1, int(maxc) + int(pad_px))
    y1 = min(h - 1, int(maxr) + int(pad_px))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def segment_prompted_sam_from_candidates(
    sam_input_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    valid_region: np.ndarray,
    interior_region: np.ndarray,
    *,
    predictor: Any,
    max_candidates: int = PROMPT_MAX_CANDIDATES,
    prompt_pad_px: int = PROMPT_PAD_PX,
    max_mask_frac: float = PROMPT_MAX_MASK_FRAC,
    max_overlap_frac: float = PROMPT_MAX_OVERLAP_FRAC,
    min_iou_with_candidate: float = PROMPT_MIN_IOU_WITH_CANDIDATE,
) -> np.ndarray:
    """Candidate-guided SAM using one local box and one positive centroid point."""
    if predictor is None:
        raise RuntimeError("SAM predictor is required for prompted SAM cases.")

    candidates = candidate_labels_for_prompting(candidate_bgr, valid_region, interior_region)
    props = sorted(measure.regionprops(candidates), key=lambda prop: prop.area, reverse=True)
    if max_candidates is not None:
        props = props[: int(max_candidates)]

    predictor.set_image(cv2.cvtColor(sam_input_bgr, cv2.COLOR_BGR2RGB))

    h, w = valid_region.shape
    total_px = h * w
    occupied = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=np.int32)
    next_label = 1

    for prop in props:
        candidate_mask = candidates == int(prop.label)
        if int(np.count_nonzero(candidate_mask)) < MIN_FLOE_AREA_PX:
            continue

        box = padded_bbox_from_region(prop.bbox, (h, w), pad_px=prompt_pad_px)
        cy, cx = prop.centroid
        point_coords = np.array([[float(cx), float(cy)]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
        except Exception as exc:
            print(f"[warn] prompted SAM skipped one candidate: {exc}")
            continue

        if masks.ndim != 3 or masks.shape[0] == 0:
            continue

        best_seg: np.ndarray | None = None
        best_key: tuple[float, float] | None = None
        for mask, score in zip(masks, scores):
            seg = mask.astype(bool) & valid_region
            area = int(np.count_nonzero(seg))
            if area < MIN_FLOE_AREA_PX:
                continue
            if area > float(max_mask_frac) * total_px:
                continue
            if not np.all(seg <= interior_region):
                continue

            intersection = int(np.count_nonzero(seg & candidate_mask))
            union = int(np.count_nonzero(seg | candidate_mask))
            iou = intersection / float(max(union, 1))
            if iou < float(min_iou_with_candidate):
                continue

            overlap = np.count_nonzero(seg & occupied) / float(max(area, 1))
            if overlap > float(max_overlap_frac):
                continue

            key = (float(score), float(iou))
            if best_key is None or key > best_key:
                best_key = key
                best_seg = seg

        if best_seg is None:
            continue

        best_seg = remove_objects_smaller_than(best_seg & (~occupied), MIN_FLOE_AREA_PX)
        if np.count_nonzero(best_seg) < MIN_FLOE_AREA_PX:
            continue

        out[best_seg] = next_label
        occupied |= best_seg
        next_label += 1

    return relabel_filtered(out, valid_region, interior_region)


# -----------------------------------------------------------------------------
# Metrics and figures
# -----------------------------------------------------------------------------


def catalog_from_labels(label_map: np.ndarray, pixel_size_m: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prop in measure.regionprops(label_map):
        area_px = int(prop.area)
        area_m2 = area_px * pixel_size_m * pixel_size_m
        perimeter_px = float(prop.perimeter) if prop.perimeter > 0 else 0.0
        circularity = 4.0 * math.pi * area_px / (perimeter_px * perimeter_px) if perimeter_px > 0 else np.nan
        major_px, minor_px = safe_axis_lengths(prop)
        rows.append(
            {
                "label": int(prop.label),
                "area_px": area_px,
                "area_m2": area_m2,
                "equivalent_diameter_m": math.sqrt(4.0 * area_m2 / math.pi),
                "major_axis_length_m": major_px * pixel_size_m,
                "minor_axis_length_m": minor_px * pixel_size_m,
                "aspect_ratio": major_px / max(minor_px, 1e-9),
                "eccentricity": float(prop.eccentricity),
                "circularity": float(circularity),
                "centroid_x_px": float(prop.centroid[1]),
                "centroid_y_px": float(prop.centroid[0]),
            }
        )
    return pd.DataFrame(rows)


def summary_from_catalog(name: str, catalog: pd.DataFrame, valid_region: np.ndarray, pixel_size_m: float) -> dict[str, Any]:
    valid_area_m2 = float(np.count_nonzero(valid_region) * pixel_size_m * pixel_size_m)
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
    ice_area_m2 = float(catalog["area_m2"].sum())
    return {
        "method": name,
        "n_floes": int(len(catalog)),
        "valid_area_m2": valid_area_m2,
        "ice_area_m2": ice_area_m2,
        "concentration_in_valid_region": ice_area_m2 / max(valid_area_m2, 1e-12),
        "mean_equivalent_diameter_m": float(np.mean(diam)),
        "median_equivalent_diameter_m": float(np.median(diam)),
        "p10_equivalent_diameter_m": float(np.percentile(diam, 10)),
        "p90_equivalent_diameter_m": float(np.percentile(diam, 90)),
    }


def make_result(name: str, label_map: np.ndarray, valid_region: np.ndarray, pixel_size_m: float) -> MethodResult:
    catalog = catalog_from_labels(label_map, pixel_size_m)
    summary = summary_from_catalog(name, catalog, valid_region, pixel_size_m)
    return MethodResult(name=name, label_map=label_map.astype(np.int32), catalog=catalog, summary=summary)


def binary_mask_bgr(label_map: np.ndarray) -> np.ndarray:
    mask = ((label_map > 0).astype(np.uint8) * 255)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def label_image_bgr(label_map: np.ndarray, catalog: pd.DataFrame) -> np.ndarray:
    values = np.zeros(label_map.shape, dtype=np.float32)
    for _, row in catalog.iterrows():
        label = int(row["label"])
        values[label_map == label] = 1.0 - math.exp(-float(row["area_px"]) / 1800.0)
    rgb = (255 * plt.get_cmap("viridis")(values)[..., :3]).astype(np.uint8)
    rgb[label_map == 0] = 0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def overlay_labels(image_bgr: np.ndarray, label_map: np.ndarray, catalog: pd.DataFrame, alpha: float = 0.45) -> np.ndarray:
    color_labels = label_image_bgr(label_map, catalog)
    overlay = image_bgr.copy()
    mask = label_map > 0
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color_labels, alpha, 0)
    overlay[mask] = blended[mask]

    for contour in measure.find_contours(label_map > 0, 0.5):
        points = np.fliplr(contour).astype(np.int32)
        cv2.polylines(overlay, [points], isClosed=True, color=(255, 255, 255), thickness=1)
    return overlay


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    return ndi.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)) ^ ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))


def draw_mask_and_box(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    box_xyxy: np.ndarray,
    *,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    out = image_bgr.copy().astype(np.float32)
    mask = mask.astype(bool)
    color = np.asarray(color_bgr, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    out[mask_boundary(mask)] = color
    out = np.clip(out, 0, 255).astype(np.uint8)

    h, w = out.shape[:2]
    x0, y0, x1, y1 = [int(round(float(v))) for v in box_xyxy]
    x0, x1 = sorted((int(np.clip(x0, 0, w - 1)), int(np.clip(x1, 0, w - 1))))
    y0, y1 = sorted((int(np.clip(y0, 0, h - 1)), int(np.clip(y1, 0, h - 1))))
    cv2.rectangle(out, (x0, y0), (x1, y1), color_bgr, 3)
    return out


def non_floe_overlay(ortho_bgr: np.ndarray, boat_mask: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
    overlay = draw_mask_and_box(ortho_bgr, boat_mask, BOAT_BOX_XYXY, color_bgr=(0, 0, 255), alpha=0.35)
    overlay = draw_mask_and_box(overlay, artifact_mask, ARTIFACT_BOX_XYXY, color_bgr=(0, 210, 255), alpha=0.45)
    return overlay


def save_grid(path: Path, panels: list[tuple[str, np.ndarray]], *, ncols: int) -> str:
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (title, image_bgr) in zip(axes_flat, panels):
        ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    for ax in axes_flat[len(panels) :]:
        ax.set_axis_off()
    fig.tight_layout(pad=0.15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(path)


def save_pipeline_outputs(
    out_dir: Path,
    *,
    raw_bgr: np.ndarray,
    ortho_bgr: np.ndarray,
    non_floe_bgr: np.ndarray,
    masked_bgr: np.ndarray,
    selected: MethodResult,
    height_px: int,
) -> dict[str, str]:
    fig_dir = out_dir / "figures" / "pipeline"
    overlay = overlay_labels(masked_bgr, selected.label_map, selected.catalog)
    binary = binary_mask_bgr(selected.label_map)

    panels = [
        ("a Raw image", raw_bgr, "a_raw_image.png"),
        ("b Orthorectified", ortho_bgr, "b_orthorectified.png"),
        ("c Non-floe masks", non_floe_bgr, "c_non_floe_masks.png"),
        ("d Masked input", masked_bgr, "d_masked_input.png"),
        ("e Floe instances", overlay, "e_floe_instances.png"),
        ("f Binary mask", binary, "f_binary_mask.png"),
    ]

    files: dict[str, str] = {}
    grid_panels: list[tuple[str, np.ndarray]] = []
    for title, image, filename in panels:
        resized = resize_to_height(image, height_px)
        key = filename.removesuffix(".png")
        files[key] = write_bgr(fig_dir / filename, resized)
        grid_panels.append((title, resized))
    files["pipeline_2x3"] = save_grid(fig_dir / "pipeline_2x3.png", grid_panels, ncols=3)
    return files


def save_sensitivity_outputs(out_dir: Path, results: list[MethodResult], *, height_px: int) -> dict[str, str]:
    fig_dir = out_dir / "figures" / "sensitivity"
    result_by_name = {result.name: result for result in results}

    files: dict[str, str] = {}
    grid_panels: list[tuple[str, np.ndarray]] = []

    ordered_names = [name for name in SENSITIVITY_ORDER if name in result_by_name]
    ordered_names.extend(result.name for result in results if result.name not in ordered_names)

    for name in ordered_names:
        result = result_by_name[name]
        title, filename = SENSITIVITY_LABELS.get(result.name, (result.name.replace("_", " "), f"{result.name}.png"))
        image = resize_to_height(binary_mask_bgr(result.label_map), height_px)
        files[result.name] = write_bgr(fig_dir / filename, image)
        grid_panels.append((title, image))

    if grid_panels:
        files["sensitivity_panel"] = save_grid(fig_dir / "sensitivity_2x3.png", grid_panels, ncols=min(3, len(grid_panels)))

    note = (
        "Sensitivity figure panels are binary floe masks from the same prepared input. "
        "The intended six-panel thesis set is morphology, k-means + watershed, SAM auto, "
        "strict SAM auto, prompted SAM on the illumination-corrected input, and prompted SAM "
        "on the CLAHE-enhanced alternate input. If a SAM option is skipped, the corresponding "
        "panel is omitted but the available panels are still saved.\n"
    )
    (fig_dir / "README.txt").write_text(note, encoding="utf-8")
    files["readme"] = str(fig_dir / "README.txt")
    return files


def save_data_outputs(out_dir: Path, results: list[MethodResult], selected: MethodResult) -> dict[str, str]:
    data_dir = out_dir / "data"
    catalog_dir = data_dir / "catalogs"
    label_dir = data_dir / "label_maps"
    mask_dir = data_dir / "binary_masks"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for result in results:
        catalog_path = catalog_dir / f"{result.name}.csv"
        result.catalog.to_csv(catalog_path, index=False)
        files[f"{result.name}_catalog"] = str(catalog_path)

        labels_path = label_dir / f"{result.name}.npz"
        np.savez_compressed(
            labels_path,
            label_map=result.label_map.astype(np.int32),
            binary_mask=(result.label_map > 0).astype(np.uint8),
            method=result.name,
        )
        files[f"{result.name}_labels"] = str(labels_path)
        files[f"{result.name}_binary_png"] = write_bgr(mask_dir / f"{result.name}.png", binary_mask_bgr(result.label_map))

    summary = pd.DataFrame([r.summary for r in results])
    files["summary"] = str(data_dir / "summary.csv")
    summary.to_csv(files["summary"], index=False)

    files["selected_labels"] = str(data_dir / "selected_floe_labels.npz")
    np.savez_compressed(
        files["selected_labels"],
        label_map=selected.label_map.astype(np.int32),
        binary_mask=(selected.label_map > 0).astype(np.uint8),
        selected_method=selected.name,
    )
    files["selected_binary_png"] = write_bgr(data_dir / "selected_binary_mask.png", binary_mask_bgr(selected.label_map))
    files["selected_catalog"] = str(data_dir / "selected_floe_catalog.csv")
    selected.catalog.to_csv(files["selected_catalog"], index=False)
    return files


def save_diagnostics(
    out_dir: Path,
    *,
    image_without_boat: np.ndarray,
    valid_region: np.ndarray,
    interior_region: np.ndarray,
) -> dict[str, str]:
    diag_dir = out_dir / "diagnostics"
    return {
        "boat_removed_input": write_bgr(diag_dir / "boat_removed_input.png", image_without_boat),
        "valid_region": write_gray(diag_dir / "valid_region.png", (valid_region.astype(np.uint8) * 255)),
        "interior_region": write_gray(diag_dir / "interior_region.png", (interior_region.astype(np.uint8) * 255)),
    }


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the sea-ice segmentation figures for the thesis chapter.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Raw shipborne image.")
    parser.add_argument("--sam-weights", type=Path, default=DEFAULT_WEIGHTS, help="SAM ViT-H checkpoint path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Clean output directory.")
    parser.add_argument(
        "--selected-method",
        choices=[
            "sam_auto",
            "sam_auto_strict",
            "prompted_illumination",
            "prompted_clahe",
            "morphology",
            "kmeans_watershed",
        ],
        default="sam_auto",
        help="Method used for the pipeline floe-instance and binary-mask panels.",
    )
    parser.add_argument("--pixel-size-m", type=float, default=ORTHO_PIXEL_SIZE_M, help="Physical pixel size after orthorectification.")
    parser.add_argument("--panel-height", type=int, default=900, help="Height in pixels for exported thesis panels.")
    parser.add_argument("--no-orthorectify", action="store_true", help="Use this when --image is already orthorectified.")
    parser.add_argument("--skip-sam-auto", action="store_true", help="Skip the default SAM automatic mask method.")
    parser.add_argument("--skip-strict-sam", action="store_true", help="Skip the stricter SAM automatic-mask sensitivity case.")
    parser.add_argument("--skip-prompted-sam", action="store_true", help="Skip the two candidate-guided prompted-SAM sensitivity cases.")
    parser.add_argument("--save-diagnostics", action="store_true", help="Also save valid/interior masks, boat-removed input, and prompted-SAM inputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw_bgr = load_bgr(args.image)
    ortho_bgr = raw_bgr if args.no_orthorectify else orthorectify(raw_bgr)

    sam_model, predictor, device = load_sam(args.sam_weights)
    image_without_boat, masked_bgr, invalid_mask, boat_mask, artifact_mask, prompt_meta = prepare_common_input(ortho_bgr, predictor)
    valid_region, interior_region = compute_regions(masked_bgr, invalid_mask)
    non_floe_bgr = non_floe_overlay(ortho_bgr, boat_mask, artifact_mask)

    results: list[MethodResult] = []

    print("[method] morphology")
    morphology_labels = segment_morphology(masked_bgr, valid_region, interior_region)
    results.append(make_result("morphology", morphology_labels, valid_region, args.pixel_size_m))

    print("[method] kmeans_watershed")
    kmeans_labels = segment_kmeans_watershed(masked_bgr, valid_region, interior_region)
    results.append(make_result("kmeans_watershed", kmeans_labels, valid_region, args.pixel_size_m))

    if not args.skip_sam_auto:
        print("[method] sam_auto")
        sam_labels = segment_sam_auto(masked_bgr, valid_region, interior_region, sam_model=sam_model, device=device)
        results.append(make_result("sam_auto", sam_labels, valid_region, args.pixel_size_m))

    if not args.skip_strict_sam:
        try:
            print("[method] sam_auto_strict")
            strict_labels = segment_sam_auto_strict(masked_bgr, valid_region, interior_region, sam_model=sam_model, device=device)
            results.append(make_result("sam_auto_strict", strict_labels, valid_region, args.pixel_size_m))
        except Exception as exc:
            print(f"[warn] sam_auto_strict skipped: {exc}")

    prompted_inputs: dict[str, str] = {}
    if not args.skip_prompted_sam:
        illumination_bgr = illumination_correct_bgr(masked_bgr, valid_region)
        clahe_bgr = clahe_enhance_bgr(masked_bgr, valid_region)
        if args.save_diagnostics:
            diag_dir = args.out / "diagnostics"
            prompted_inputs["illumination_corrected"] = write_bgr(diag_dir / "prompt_input_illumination_corrected.png", illumination_bgr)
            prompted_inputs["clahe_enhanced"] = write_bgr(diag_dir / "prompt_input_clahe_enhanced.png", clahe_bgr)

        try:
            print("[method] prompted_illumination")
            labels = segment_prompted_sam_from_candidates(
                illumination_bgr,
                illumination_bgr,
                valid_region,
                interior_region,
                predictor=predictor,
            )
            results.append(make_result("prompted_illumination", labels, valid_region, args.pixel_size_m))
        except Exception as exc:
            print(f"[warn] prompted_illumination skipped: {exc}")

        try:
            print("[method] prompted_clahe")
            labels = segment_prompted_sam_from_candidates(
                clahe_bgr,
                clahe_bgr,
                valid_region,
                interior_region,
                predictor=predictor,
            )
            results.append(make_result("prompted_clahe", labels, valid_region, args.pixel_size_m))
        except Exception as exc:
            print(f"[warn] prompted_clahe skipped: {exc}")

    result_by_name = {result.name: result for result in results}
    if args.selected_method not in result_by_name:
        available = ", ".join(result_by_name)
        raise RuntimeError(f"Selected method {args.selected_method!r} was not run. Available methods: {available}")
    selected = result_by_name[args.selected_method]

    outputs: dict[str, Any] = {
        "pipeline_figures": save_pipeline_outputs(
            args.out,
            raw_bgr=raw_bgr,
            ortho_bgr=ortho_bgr,
            non_floe_bgr=non_floe_bgr,
            masked_bgr=masked_bgr,
            selected=selected,
            height_px=args.panel_height,
        ),
        "sensitivity_figures": save_sensitivity_outputs(args.out, results, height_px=args.panel_height),
        "data": save_data_outputs(args.out, results, selected),
    }

    if args.save_diagnostics:
        outputs["diagnostics"] = save_diagnostics(
            args.out,
            image_without_boat=image_without_boat,
            valid_region=valid_region,
            interior_region=interior_region,
        )
        outputs["diagnostics"].update(prompted_inputs)

    metadata = {
        "image": str(args.image),
        "sam_weights": str(args.sam_weights),
        "device": device,
        "orthorectified": not args.no_orthorectify,
        "orthorectification": {
            "camera_intrinsics": CAMERA_INTRINSICS,
            "camera_extrinsics": CAMERA_EXTRINSICS,
            "extent_m": ORTHO_EXTENT_M,
            "pixel_size_m": ORTHO_PIXEL_SIZE_M,
        },
        "selected_method": selected.name,
        "pixel_size_m_for_metrics": args.pixel_size_m,
        "prompted_non_floe_masks": prompt_meta,
        "shared_filters": {
            "interior_margin_px": INTERIOR_MARGIN_PX,
            "min_floe_area_px": MIN_FLOE_AREA_PX,
            "max_floe_area_frac": MAX_FLOE_AREA_FRAC,
            "max_aspect_ratio": MAX_ASPECT_RATIO,
            "min_fill_ratio": MIN_FILL_RATIO,
        },
        "methods": {
            "morphology": {
                "threshold": "Otsu on valid grayscale pixels",
                "erode_radius_px": MORPH_ERODE_RADIUS_PX,
                "dilate_radius_px": MORPH_DILATE_RADIUS_PX,
                "min_hole_area_px": MORPH_MIN_HOLE_AREA_PX,
            },
            "kmeans_watershed": {
                "kmeans_k": KMEANS_K,
                "watershed_min_distance_px": WATERSHED_MIN_DISTANCE_PX,
                "watershed_peak_rel": WATERSHED_PEAK_REL,
            },
            "sam_auto": {
                "run": not args.skip_sam_auto,
                "auto_cfg": SAM_AUTO_CFG,
                "max_mask_frac": SAM_MAX_MASK_FRAC,
                "max_overlap_frac": SAM_MAX_OVERLAP_FRAC,
            },
            "sam_auto_strict": {
                "run": not args.skip_strict_sam,
                "auto_cfg": SAM_AUTO_STRICT_CFG,
                "max_mask_frac": SAM_AUTO_STRICT_MAX_MASK_FRAC,
                "max_overlap_frac": SAM_AUTO_STRICT_MAX_OVERLAP_FRAC,
            },
            "prompted_sam": {
                "run": not args.skip_prompted_sam,
                "candidate_method": "kmeans_watershed proposal regions",
                "prompt": "SAM box plus positive centroid point for each proposal",
                "max_candidates": PROMPT_MAX_CANDIDATES,
                "prompt_pad_px": PROMPT_PAD_PX,
                "min_iou_with_candidate": PROMPT_MIN_IOU_WITH_CANDIDATE,
                "max_mask_frac": PROMPT_MAX_MASK_FRAC,
                "max_overlap_frac": PROMPT_MAX_OVERLAP_FRAC,
                "cases": ["prompted_illumination", "prompted_clahe"],
            },
        },
        "outputs": outputs,
    }
    metadata_dir = args.out / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "run_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_df = pd.DataFrame([r.summary for r in results])
    print(f"[done] wrote clean thesis outputs to {args.out}")
    print(f"[selected] {selected.name}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()