from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .env import slug_float
from .model_config import SamCoverSettings


def json_ready(obj: Any) -> Any:
    """Convert common numerical objects to JSON-safe values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):
        return json_ready(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def coerce_shape(array: np.ndarray, *, rows: int, cols: int, name: str = "array") -> np.ndarray:
    """Return an array with shape (rows, cols), accepting a transposed input."""
    A = np.asarray(array, dtype=float)
    if A.shape == (rows, cols):
        return A
    if A.T.shape == (rows, cols):
        return A.T
    raise ValueError(f"{name} has shape {A.shape}; expected {(rows, cols)} or {(cols, rows)}.")


def row_mean_cover(F: np.ndarray) -> np.ndarray:
    """Replace each row by its x-mean."""
    F = np.clip(np.asarray(F, dtype=float), 0.0, 1.0)
    row_mean = np.mean(F, axis=1, keepdims=True)
    return np.repeat(row_mean, F.shape[1], axis=1)


def shift_to_mean(F: np.ndarray, target_mean: float, *, maxiter: int = 80) -> np.ndarray:
    """Add a constant and clamp so the cover mean matches target_mean."""
    F0 = np.asarray(F, dtype=float)
    target = float(np.clip(target_mean, 0.0, 1.0))

    def mean_at(delta: float) -> float:
        return float(np.mean(np.clip(F0 + delta, 0.0, 1.0)))

    lo, hi = -1.0, 1.0
    if target <= mean_at(lo):
        return np.clip(F0 + lo, 0.0, 1.0)
    if target >= mean_at(hi):
        return np.clip(F0 + hi, 0.0, 1.0)

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return np.clip(F0 + 0.5 * (lo + hi), 0.0, 1.0)


def scale_to_mean(F: np.ndarray, target_mean: float) -> np.ndarray:
    """Scale and clip a cover to approximately match the target mean."""
    F = np.clip(np.asarray(F, dtype=float), 0.0, 1.0)
    mean = max(float(np.mean(F)), 1e-12)
    return np.clip(F * (float(target_mean) / mean), 0.0, 1.0)


def logistic_beta(y: np.ndarray, *, gamma: float, sigma: float) -> np.ndarray:
    """Fallback logistic ridge/channel profile.

    The preferred implementation is the project helper
    pywave.waves_helpers.cover_core.logistic_beta_y. This fallback keeps the
    public wrapper readable and testable, but the numerical reproduction should
    use the project helper when available.
    """
    y = np.asarray(y, dtype=float)
    return 1.0 / (1.0 + np.exp(-float(gamma) * (np.abs(y) - float(sigma))))


def logistic_cover(x: np.ndarray, y: np.ndarray, *, gamma: float, sigma: float) -> np.ndarray:
    beta = logistic_beta(y, gamma=gamma, sigma=sigma).reshape(-1, 1)
    return np.repeat(np.clip(beta, 0.0, 1.0), len(np.asarray(x).reshape(-1)), axis=1)


def compose_fragmented_channel(
    source_cover: np.ndarray,
    beta: np.ndarray,
    *,
    mode: str,
    texture_contrast: float = 0.45,
    row_mean_floor: float = 1e-4,
) -> np.ndarray:
    """Combine a smooth channel beta(y) with a data-derived floe field S(x,y)."""
    S = np.clip(np.asarray(source_cover, dtype=float), 0.0, 1.0)
    B = np.clip(np.asarray(beta, dtype=float), 0.0, 1.0)
    if B.ndim == 1:
        B = np.repeat(B.reshape(-1, 1), S.shape[1], axis=1)
    B = coerce_shape(B, rows=S.shape[0], cols=S.shape[1], name="beta")

    mode = mode.strip().lower()
    if mode == "continuous":
        return B
    if mode == "product":
        return np.clip(B * S, 0.0, 1.0)
    if mode == "texture":
        row_mean = np.mean(S, axis=1, keepdims=True)
        texture = S - row_mean
        return np.clip(B + float(texture_contrast) * texture, 0.0, 1.0)
    if mode == "row_normalized_product":
        row_mean = np.maximum(np.mean(S, axis=1, keepdims=True), float(row_mean_floor))
        return np.clip(B * S / row_mean, 0.0, 1.0)
    raise ValueError(f"Unknown composition mode {mode!r}.")


def centered_clamped_crop(
    *,
    scale: float,
    full_shape_hw: tuple[int, int],
    base_x0: int,
    base_y0: int,
    base_w: int,
    base_h: int,
) -> dict[str, Any]:
    """Scale a crop around its original centre and keep it inside the image."""
    H, W = int(full_shape_hw[0]), int(full_shape_hw[1])
    cx = float(base_x0) + 0.5 * float(base_w)
    cy = float(base_y0) + 0.5 * float(base_h)

    requested_w = max(4, int(round(float(base_w) * float(scale))))
    requested_h = max(4, int(round(float(base_h) * float(scale))))
    w = min(requested_w, W)
    h = min(requested_h, H)

    x0 = int(round(cx - 0.5 * w))
    y0 = int(round(cy - 0.5 * h))
    x0 = max(0, min(x0, W - w))
    y0 = max(0, min(y0, H - h))

    return {
        "requested_scale": float(scale),
        "effective_scale_x": float(w / max(float(base_w), 1.0)),
        "effective_scale_y": float(h / max(float(base_h), 1.0)),
        "x0": int(x0),
        "y0": int(y0),
        "w": int(w),
        "h": int(h),
        "was_clamped": bool(w != requested_w or h != requested_h),
        "full_mask_shape_hw": [int(H), int(W)],
    }


def crop_label(crop: dict[str, Any]) -> str:
    return f"crop{slug_float(crop['effective_scale_x'])}x{slug_float(crop['effective_scale_y'])}"


def cover_roughness(F: np.ndarray, *, dx: float = 1.0, dy: float = 1.0) -> dict[str, float]:
    """Simple cover-field summary used in comparison tables."""
    F = np.asarray(F, dtype=float)
    gx = np.diff(F, axis=1) / max(float(dx), 1e-12)
    gy = np.diff(F, axis=0) / max(float(dy), 1e-12)
    texture = F - np.mean(F, axis=1, keepdims=True)
    return {
        "cover_mean": float(np.mean(F)),
        "cover_std": float(np.std(F)),
        "cover_min": float(np.min(F)),
        "cover_max": float(np.max(F)),
        "x_texture_std": float(np.std(texture)),
        "mean_abs_grad_x": float(np.mean(np.abs(gx))) if gx.size else 0.0,
        "mean_abs_grad_y": float(np.mean(np.abs(gy))) if gy.size else 0.0,
        "max_abs_grad_x": float(np.max(np.abs(gx))) if gx.size else 0.0,
        "max_abs_grad_y": float(np.max(np.abs(gy))) if gy.size else 0.0,
    }


def field_metrics(Z: np.ndarray, reference: np.ndarray | None = None, *, prefix: str = "") -> dict[str, float | None]:
    """Simple solution metrics, optionally against a reference solution."""
    Z = np.asarray(Z, dtype=float)
    out: dict[str, float | None] = {
        f"{prefix}zeta_min": float(np.min(Z)),
        f"{prefix}zeta_max": float(np.max(Z)),
        f"{prefix}zeta_amp": float(np.max(Z) - np.min(Z)),
        f"{prefix}zeta_l2": float(np.sqrt(np.mean(Z * Z))),
    }
    if reference is None:
        return out
    R = np.asarray(reference, dtype=float)
    if R.shape != Z.shape:
        out[f"{prefix}rmse_vs_reference"] = None
        out[f"{prefix}relative_rmse_vs_reference"] = None
        out[f"{prefix}corr_vs_reference"] = None
        return out
    diff = Z - R
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sqrt(np.mean(R * R)))
    zc = Z.ravel() - float(np.mean(Z))
    rc = R.ravel() - float(np.mean(R))
    corr_denom = float(np.linalg.norm(zc) * np.linalg.norm(rc))
    out[f"{prefix}rmse_vs_reference"] = rmse
    out[f"{prefix}relative_rmse_vs_reference"] = rmse / max(denom, 1e-14)
    out[f"{prefix}corr_vs_reference"] = float(np.dot(zc, rc) / corr_denom) if corr_denom > 0 else None
    return out


def load_sam_cover_sources(
    *,
    adapter: Any,
    settings: SamCoverSettings,
    out_dir: Path,
    rows: int,
    cols: int,
    crop: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build and load direct and ellipse cover fields from a SAM pipeline folder."""
    crop = crop or {
        "x0": settings.crop_x0,
        "y0": settings.crop_y0,
        "w": settings.crop_w,
        "h": settings.crop_h,
    }
    pipeline_dir = adapter.resolve_sam_pipeline_dir(settings.pipeline_dir)
    meta = adapter.build_sam_cover_npzs(
        pipeline_dir=pipeline_dir,
        out_dir=out_dir,
        direct_source=settings.direct_source,
        crop_x0=int(crop["x0"]),
        crop_y0=int(crop["y0"]),
        crop_w=int(crop["w"]),
        crop_h=int(crop["h"]),
        smooth_width_px=float(settings.smooth_width_px),
        N=cols,
        M=rows,
    )

    sources: dict[str, dict[str, Any]] = {}
    if "direct" in settings.sources:
        data = adapter.load_cover_npz(Path(meta["direct_npz"]))
        sources["direct"] = {
            "F": coerce_shape(data["F_low"], rows=rows, cols=cols, name="direct F_low"),
            "path": Path(meta["direct_npz"]),
            "data": data,
        }
    if "ellipse" in settings.sources:
        data = adapter.load_cover_npz(Path(meta["ellipse_npz"]))
        sources["ellipse"] = {
            "F": coerce_shape(data["F_low"], rows=rows, cols=cols, name="ellipse F_low"),
            "path": Path(meta["ellipse_npz"]),
            "data": data,
        }
    return sources
