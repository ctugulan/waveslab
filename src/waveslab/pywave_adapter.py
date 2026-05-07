from __future__ import annotations

import importlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .model_config import SolverSettings


class MissingCustomDependency(RuntimeError):
    """Raised when a private project dependency is not importable."""


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise MissingCustomDependency(
            f"Could not import {name!r}. The requested WavesLab module could not be imported."
        ) from exc


class PyWaveAdapter:
    """Small boundary between public examples and the steady solver modules.

    The old scripts imported many helpers directly. This adapter keeps the public examples readable.
    """

    def __init__(self) -> None:
        os.environ.setdefault("VISCICE_USE_X64", "1")
        self.runner = _import("waveslab.steady.cover_runner")
        self.cover_core = _import("waveslab.cover_core")

    def solver_config(self, settings: SolverSettings) -> Any:
        cfg = self.runner.SOLVER_CFG
        kwargs = {
            "full_domain": bool(settings.full_domain),
            "block_builder": str(settings.block_builder),
            "Fr": float(settings.Fr),
            "tauf": float(settings.tauf),
            "epsilon": float(settings.epsilon),
            "Lx": float(settings.Lx),
            "Ly": float(settings.Ly),
            "N": int(settings.N),
            "M": int(settings.M),
            "dx": float(settings.dx),
            "dy": float(settings.dy),
            "x0": float(settings.x0),
            "upstream_bc": str(settings.upstream_bc),
            "pad_mode": str(settings.pad_mode),
            "use_radiation": bool(settings.use_radiation),
            "rigidity_min": float(settings.rigidity_min),
            "rigidity_max_scale": float(settings.rigidity_max_scale),
            "skip_if_exists": bool(settings.skip_if_exists),
        }
        if hasattr(cfg, "full_domain_y0"):
            kwargs["full_domain_y0"] = None if settings.full_domain else getattr(cfg, "full_domain_y0")
        if hasattr(cfg, "symmetric_solver_mode"):
            kwargs["symmetric_solver_mode"] = "never" if settings.full_domain else getattr(cfg, "symmetric_solver_mode")
        if settings.aleph is not None:
            kwargs["aleph"] = float(settings.aleph)
        if settings.mu is not None:
            kwargs["mu"] = float(settings.mu)
        return replace(cfg, **kwargs)

    def make_grid(self, cfg: Any) -> tuple[np.ndarray, np.ndarray]:
        grid = self.runner._make_grid(cfg)
        x = np.asarray(grid.x, dtype=float)
        y = np.asarray(grid.y, dtype=float)
        if x.ndim == 2:
            x = x[0, :]
        if y.ndim == 2:
            y = y[:, 0]
        return x.reshape(-1), y.reshape(-1)

    def resolve_sam_pipeline_dir(self, explicit: Path | None = None) -> Path:
        if explicit is not None:
            return Path(explicit).expanduser().resolve()
        resolver = getattr(self.runner, "resolve_sam_pipeline_dir", None)
        if callable(resolver):
            return Path(resolver()).expanduser().resolve()
        env = os.environ.get("SAM_PIPELINE_DIR") or os.environ.get("FLEXURAL_COVER_PIPELINE_DIR")
        if env:
            return Path(env).expanduser().resolve()
        raise FileNotFoundError("Set SAM_PIPELINE_DIR or pass a pipeline_dir explicitly.")

    def full_binary_shape(self, pipeline_dir: Path, *, direct_source: str) -> tuple[tuple[int, int], str]:
        loader = getattr(self.cover_core, "_load_direct_binary_from_pipeline", None)
        if loader is None:
            return (10_000, 10_000), "unknown_shape_fallback"
        binary, key, _ = loader(Path(pipeline_dir), direct_source=direct_source)
        return tuple(np.asarray(binary).shape), str(key)

    def build_sam_cover_npzs(self, **kwargs: Any) -> dict[str, Any]:
        return self.cover_core.build_cover_npzs_from_sam_pipeline(
            write_full_binary_exports=False,
            write_preview=False,
            **kwargs,
        )

    def load_cover_npz(self, path: Path) -> dict[str, Any]:
        return self.cover_core.load_cover_npz(Path(path))

    def logistic_beta_y(self, y: np.ndarray, *, gamma: float, sigma: float) -> np.ndarray:
        func = getattr(self.cover_core, "logistic_beta_y", None)
        if func is None:
            from .covers import logistic_beta

            return logistic_beta(y, gamma=gamma, sigma=sigma)
        return np.asarray(func(y, gamma=float(gamma), sigma=float(sigma)), dtype=float)

    def floe_diameter_stats(
        self,
        binary_crop: np.ndarray,
        *,
        meters_per_pixel: float,
        min_area_px: int = 4,
        exclude_border: bool = True,
    ) -> dict[str, Any]:
        func = getattr(self.cover_core, "floe_diameters_from_binary", None)
        if func is None:
            return {"diameter_error": "floe_diameters_from_binary is unavailable"}
        return func(
            np.asarray(binary_crop).astype(bool),
            meters_per_pixel=float(meters_per_pixel),
            min_area_px=int(min_area_px),
            exclude_border=bool(exclude_border),
        )

    def run_or_load(self, F: np.ndarray, run_dir: Path, cfg: Any, *, label: str) -> dict[str, Any]:
        run_dir = Path(run_dir)
        if (
            bool(getattr(cfg, "skip_if_exists", False))
            and (run_dir / "zeta.npy").exists()
            and (run_dir / "cover_F.npy").exists()
            and (run_dir / "meta.json").exists()
        ):
            return self.runner._load_existing_run(run_dir, cfg)

        domain_decision = {
            "cover_case_label": label,
            "requested_full_domain": bool(getattr(cfg, "full_domain", False)),
            "effective_full_domain": bool(getattr(cfg, "full_domain", False)),
            "used_symmetric_half_domain": not bool(getattr(cfg, "full_domain", False)),
            "reason": "clean public wrapper",
        }
        try:
            return self.runner._run_wave_case(F, run_dir, cfg, domain_decision=domain_decision)
        except TypeError:
            return self.runner._run_wave_case(F, run_dir, cfg)


def load_deep_runner() -> Any:
    """Return the steady deep-case runner for wavelength extraction."""
    return _import("waveslab.steady.driver")
