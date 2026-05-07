from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SolverSettings:
    """Numerical and physical settings for the steady solver."""

    N: int = 121
    M: int = 61
    dx: float = 10.0 / 31.0
    dy: float = 10.0 / 31.0
    x0: float = -20.0
    Fr: float = 0.7
    aleph: float | None = None
    mu: float | None = 0.0
    tauf: float = 0.0
    epsilon: float = 0.1
    Lx: float = 1.0
    Ly: float = 1.0
    full_domain: bool = False
    use_radiation: bool = False
    rigidity_min: float = 0.0
    rigidity_max_scale: float = 1.0
    upstream_bc: str = "centered"
    pad_mode: str = "zeros"
    block_builder: str = "analytic"
    skip_if_exists: bool = True


@dataclass(frozen=True)
class SamCoverSettings:
    """How a SAM-derived mask is cropped and mapped to the solver grid."""

    pipeline_dir: Path | None = None
    direct_source: str = "accepted_label_map"
    crop_x0: int = 256 * 3
    crop_y0: int = 256
    crop_w: int = 256 * 3
    crop_h: int = 256 * 3
    smooth_width_px: float = 2.0
    sources: tuple[str, ...] = ("direct", "ellipse")


@dataclass(frozen=True)
class LogisticSweepSettings:
    """Analytic ridge/channel cover fields, F(x,y)=beta(y)."""

    ridge_gamma: float = 3.0
    channel_gamma: float = -3.0
    ridge_sigmas: tuple[float, ...] = (-4.0, 0.0, 4.0, 8.0)
    channel_sigmas: tuple[float, ...] = (-4.0, 0.0, 4.0, 8.0)
    run_ridge: bool = True
    run_channel: bool = True


@dataclass(frozen=True)
class ImageScatteringSettings:
    """Raw image-derived covers compared with x-homogenized covers."""

    crop_scales: tuple[float, ...] = (1.0,)
    match_native_mean: bool = True
    run_homogenized: bool = True
    target_wavelength_json: Path | None = None
    target_kind: str = "short"


@dataclass(frozen=True)
class BridgeSettings:
    """Controlled bridge from smooth channels to fragmented covers."""

    gamma: float = -3.0
    sigmas: tuple[float, ...] = (4.0,)
    compose_modes: tuple[str, ...] = ("texture",)
    logistic_fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    alphas: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    texture_contrast: float = 0.45
    row_mean_floor: float = 1e-4
    run_endpoint_product: bool = False
