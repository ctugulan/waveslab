from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter


LAMBDA_STAR = 27.0 / 256.0


@dataclass(frozen=True)
class WavelengthSettings:
    N: int = 240
    M: int = 80
    dx: float = 0.3
    dy: float = 0.6
    Fr: float | None = None
    aleph: float = 0.5
    mu: float = 0.1
    tauf: float = 0.0
    epsilon: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0
    x0: float | None = None
    near_skip: float = 4.0
    edge_trim: float = 3.0


def classify_infinite_depth_regime(F: float, aleph: float) -> dict[str, float | str]:
    if F <= 0.0 or aleph <= 0.0:
        raise ValueError("F and aleph must be positive.")
    lam = aleph * (F ** 3)
    if lam < LAMBDA_STAR:
        regime = "c_min < U"
    elif lam > LAMBDA_STAR:
        regime = "U < c_min"
    else:
        regime = "U approximately c_min"
    return {
        "F": float(F),
        "aleph": float(aleph),
        "lambda": float(lam),
        "lambda_star": float(LAMBDA_STAR),
        "Fcrit": float((LAMBDA_STAR / aleph) ** (1.0 / 3.0)),
        "regime": regime,
    }


def auto_supercritical_F(aleph: float, *, frac: float = 0.92) -> float:
    return float(frac * (LAMBDA_STAR / float(aleph)) ** (1.0 / 3.0))


def _odd_window(n: int, *, frac: float = 0.08, min_w: int = 7) -> int:
    w = max(min_w, int(frac * n))
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if n % 2 == 0 else n
    if w < 5:
        w = 5
    if w % 2 == 0:
        w += 1
    return w


def detrend_oscillatory_signal(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    if len(z) < 9:
        return z - np.mean(z), np.zeros_like(z)
    w = _odd_window(len(z), frac=0.12, min_w=9)
    trend = savgol_filter(z, window_length=w, polyorder=3, mode="interp")
    return z - trend, trend


def interp_zero_crossings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.sign(y)
    s[s == 0.0] = 1.0
    idx = np.where(np.diff(s) != 0)[0]
    out: list[float] = []
    for i in idx:
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        if y1 == y0:
            out.append(0.5 * (x0 + x1))
        else:
            out.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return np.asarray(out, dtype=float)


def fft_wavelength(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 16:
        return None
    dx = float(np.median(np.diff(x)))
    yy = y - float(np.mean(y))
    if np.allclose(yy, 0.0):
        return None
    spec = np.fft.rfft(yy * np.hanning(len(yy)))
    freq = np.fft.rfftfreq(len(yy), d=dx)
    power = np.abs(spec) ** 2
    if len(power) <= 1:
        return None
    power[0] = 0.0
    j = int(np.argmax(power))
    if j == 0 or freq[j] <= 0.0:
        return None
    return float(1.0 / freq[j])


def estimate_side_wavelength(
    x: np.ndarray,
    z: np.ndarray,
    *,
    side: str,
    x_ref: float = 0.0,
    near_skip: float = 4.0,
    edge_trim: float = 3.0,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if side == "left":
        mask = x < float(x_ref) - float(near_skip)
        if np.any(mask):
            mask &= x > float(np.min(x[mask])) + float(edge_trim)
    elif side == "right":
        mask = x > float(x_ref) + float(near_skip)
        if np.any(mask):
            mask &= x < float(np.max(x[mask])) - float(edge_trim)
    else:
        raise ValueError("side must be 'left' or 'right'.")

    xs = x[mask]
    zs = z[mask]
    if len(xs) < 12:
        return {"side": side, "lambda_peak": None, "lambda_zero": None, "lambda_fft": None, "lambda_char": None}

    z_osc, _trend = detrend_oscillatory_signal(xs, zs)
    amp = float(np.max(np.abs(z_osc)))
    prominence = max(0.06 * amp, 1e-12)
    peak_idx, _ = find_peaks(z_osc, prominence=prominence)
    trough_idx, _ = find_peaks(-z_osc, prominence=prominence)

    estimates: list[float] = []
    peak_estimates: list[float] = []
    if len(peak_idx) >= 2:
        peak_estimates.append(float(np.median(np.diff(xs[peak_idx]))))
    if len(trough_idx) >= 2:
        peak_estimates.append(float(np.median(np.diff(xs[trough_idx]))))
    estimates.extend(peak_estimates)

    zero_crossings = interp_zero_crossings(xs, z_osc)
    lambda_zero = None
    if len(zero_crossings) >= 3:
        lambda_zero = float(np.median(zero_crossings[2:] - zero_crossings[:-2]))
        estimates.append(lambda_zero)

    lambda_fft = fft_wavelength(xs, z_osc)
    if lambda_fft is not None:
        estimates.append(lambda_fft)

    return {
        "side": side,
        "lambda_peak": float(np.median(peak_estimates)) if peak_estimates else None,
        "lambda_zero": lambda_zero,
        "lambda_fft": lambda_fft,
        "lambda_char": float(np.median(estimates)) if estimates else None,
    }


def summarize_wavelengths(settings: WavelengthSettings, x: np.ndarray, zeta_y0: np.ndarray) -> dict[str, Any]:
    Fr = settings.Fr if settings.Fr is not None else auto_supercritical_F(settings.aleph)
    regime = classify_infinite_depth_regime(Fr, settings.aleph)
    left = estimate_side_wavelength(
        x, zeta_y0, side="left", near_skip=settings.near_skip, edge_trim=settings.edge_trim
    )
    right = estimate_side_wavelength(
        x, zeta_y0, side="right", near_skip=settings.near_skip, edge_trim=settings.edge_trim
    )
    candidates = [v for v in (left["lambda_char"], right["lambda_char"]) if v is not None and np.isfinite(v)]
    lambda_short = float(min(candidates)) if candidates else None
    lambda_long = float(max(candidates)) if candidates else None
    return {
        "settings": asdict(settings),
        "Fr_used": float(Fr),
        "regime": regime,
        "left": left,
        "right": right,
        "characteristic_scales": {
            "lambda_flexural_like": lambda_short,
            "lambda_gravity_like": lambda_long,
            "floe_diameter_suggestions": None if not candidates else {
                "0.5x_short": None if lambda_short is None else 0.5 * lambda_short,
                "1.0x_short": lambda_short,
                "1.0x_long": lambda_long,
            },
        },
    }
