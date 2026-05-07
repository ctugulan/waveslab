from __future__ import annotations

"""
Single press-play driver for the simplified SAM -> cover -> flexural-wave workflow.

Target workflow
---------------
1. Read SAM outputs:
       pipeline_arrays.npz, floe_catalog.csv, run_summary.json
2. Build solver covers once:
       cover_direct_from_crop.npz, cover_from_ellipse_crop.npz
3. Run any enabled cover cases in one VS Code press-play session:
       homogeneous / constant, direct mask, ellipse approximation
4. For each enabled case, run solver:
       zeta.npy, cover_F.npy, meta.json
5. For each enabled case, save:
       publication-style surface-cover plot, final_summary.json
6. If multiple cases complete, save publication-style cross-section comparisons.
7. Save a multi-case batch summary:
       runs/flexural_cover_multi_case_summary.json

Place in:
    src/waveslab/steady/cover_runner.py
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# JAX config before importing jax / pywave modules that may import jax.
# -----------------------------------------------------------------------------
_USE_X64 = os.environ.get("VISCICE_USE_X64", "1") == "1"
from jax import config as jax_config

jax_config.update("jax_enable_x64", _USE_X64)
import jax
import jax.numpy as jnp


def _ensure_repo_on_path() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pythonwaves").is_dir() and ((p / "viscice").is_dir() or (p / "my-packages").is_dir()):
            for candidate in [p, p / "my-packages" / "pypure"]:
                if candidate.exists() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
            return p
    cwd = Path.cwd().resolve()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    pp = cwd / "my-packages" / "pypure"
    if pp.exists() and str(pp) not in sys.path:
        sys.path.insert(0, str(pp))
    return cwd


REPO_ROOT = _ensure_repo_on_path()

from waveslab.cover_core import (  # noqa: E402
    CalibrationOptions,
    RigidityField,
    build_cover_npzs_from_sam_pipeline,
    cover_cases_to_run,
    json_ready,
    load_cover_npz,
    select_cover_for_case,
)
from waveslab.cover_rendering import (  # noqa: E402
    save_publication_cross_section_comparison,
    save_publication_surface_with_cover,
)
from waveslab.steady.driver import (  # noqa: E402
    make_grid as make_half_grid,
    initial_guess as initial_guess_half,
    pressure as pressure_half,
)
from waveslab.steady.newton import NewtonKrylovSolver  # noqa: E402
from waveslab.steady.residuals import (  # noqa: E402
    jax_residual_deep as jax_residual_deep_half,
    reshaping2Unknowns as reshaping2Unknowns_half,
    allVals as allVals_half,
)
from waveslab.steady.blocks import build_blocks_infinite_depth as build_blocks_infinite_depth_half  # noqa: E402
from waveslab.steady.biharmonic import (  # noqa: E402
    BiharmonicPolicy,
    flexural_contribution as flexural_contribution_half,
)
from waveslab.steady.full_domain import (  # noqa: E402
    make_full_grid,
    initial_guess as initial_guess_full,
    pressure as pressure_full,
    jax_residual_deep_full_domain,
    reshaping2Unknowns as reshaping2Unknowns_full,
    allVals as allVals_full,
    build_blocks_infinite_depth_full_domain,
    flexural_contribution_full_domain,
)
from waveslab.steady.preconditioners import ldu2_prec_operator  # noqa: E402
from waveslab.steady.names import (  # noqa: E402
    flexural_cover_run_tag,
    flexural_cover_surface_path,
)


HERE = Path(__file__).resolve()


def _env_flag(name: str, default: bool) -> bool:
    """Read a permissive boolean flag from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_optional_single_case_requested() -> bool:
    """Return True only when the legacy single-case env variable was explicitly set."""
    return bool(os.environ.get("FLEXURAL_COVER_CASE", "").strip())


@dataclass(frozen=True)
class CoverBuildConfig:
    direct_source: str = "accepted_label_map"
    crop_x0: int = 256 * 3
    crop_y0: int = 256
    crop_w: int = 256 * 3
    crop_h: int = 256 * 3
    smooth_width_px: float = 2.0
    # By default the solver-cover grid follows SolverConfig.N/M so the field
    # shape, solver arrays, and generated run tag remain synchronized.
    N: int | None = None
    M: int | None = None
    write_full_binary_exports: bool = True


@dataclass(frozen=True)
class CoverSelectionConfig:
    # Select which solver-scale cover field to run.
    #
    # Recommended cases:
    #   direct          : native cover from accepted SAM mask
    #   ellipse         : native cover from ellipse approximation
    #   homogeneous     : spatially uniform baseline, defaulting to the direct-cover mean
    #
    # Also accepted:
    #   direct_native_mask, ellipse_native_mask, direct_lambda_flex_matched,
    #   ellipse_lambda_flex_matched, uniform, constant.
    case: str = os.environ.get("FLEXURAL_COVER_CASE", "direct").strip().lower()

    # For homogeneous cases, use one of:
    #   direct_mean, ellipse_mean, or a numeric value such as 1.0, 0.5, 0.0.
    homogeneous_value: str = os.environ.get("FLEXURAL_HOMOGENEOUS_VALUE", "direct_mean").strip().lower()


@dataclass(frozen=True)
class CoverRunPlanConfig:
    """Press-play case toggles for running several cover models in one script call.

    Edit these booleans directly for VS Code runs, or override them with:
        FLEXURAL_RUN_HOMOGENEOUS=0/1
        FLEXURAL_RUN_DIRECT=0/1
        FLEXURAL_RUN_ELLIPSE=0/1

    By default all three main comparison cases are run. Set
    FLEXURAL_USE_SINGLE_CASE=1, or explicitly set FLEXURAL_COVER_CASE, to use
    the older one-case behaviour.
    """

    # Main press-play flags. These are the lines to edit most often.
    run_homogeneous_constant: bool = _env_flag("FLEXURAL_RUN_HOMOGENEOUS", True)
    run_direct: bool = _env_flag("FLEXURAL_RUN_DIRECT", True)
    run_ellipse: bool = _env_flag("FLEXURAL_RUN_ELLIPSE", True)

    # Compatibility switch for old workflow: run only COVER_SELECT_CFG.case.
    use_single_case: bool = _env_flag(
        "FLEXURAL_USE_SINGLE_CASE",
        _env_optional_single_case_requested(),
    )

    # Default is fail-fast so numerical issues are visible immediately.
    continue_on_case_error: bool = _env_flag("FLEXURAL_CONTINUE_ON_ERROR", False)


@dataclass(frozen=True)
class CrossSectionPlotConfig:
    """Publication-style comparison plots across completed cover cases.

    These are saved only when at least two cover cases complete.  The default
    origin slice compares the same physical lines across cases:
        zeta(x, 0) and zeta(0, y).

    Set FLEXURAL_COVER_SLICE_KIND=absmax to reproduce the older
    plot_steady_all.py absmax slice convention instead.
    """

    enabled: bool = _env_flag("FLEXURAL_SAVE_CROSS_SECTIONS", True)
    slice_kind: str = os.environ.get("FLEXURAL_COVER_SLICE_KIND", "origin").strip().lower()
    out_stem: str = os.environ.get("FLEXURAL_COVER_COMPARE_STEM", "flexural_cover_cases").strip()


@dataclass(frozen=True)
class CalibrationConfig:
    # Safe default: use the native direct mask-derived cover in the solver.
    # Set enabled=True, or FLEXURAL_COVER_CALIBRATE=1, only for the optional
    # scale-sensitivity ablation where the floe diameter statistic is matched
    # to the target flexural wavelength.
    enabled: bool = os.environ.get("FLEXURAL_COVER_CALIBRATE", "0").lower() in {"1", "true", "yes", "on"}
    target_lambda_flex_m: float = 0.6041194293172856
    reference_stat: str = "area_weighted_mean"
    min_area_px: int = 4
    match_tol_rel: float = 0.05
    max_scale_iters: int = 10
    preserve_mean_cover: bool = True
    meters_per_pixel: float | None = None
    meters_per_pixel_fallback: float = 1.0 / 29.0


@dataclass(frozen=True)
class SolverConfig:
    # Domain / backend
    full_domain: bool = True
    full_domain_y0: float | None = None
    block_builder: str = "analytic"

    # Symmetry-aware shortcut for cover fields.  The full-domain solver is still
    # required for asymmetric covers, but symmetric covers can be solved with the
    # cheaper half-y mirrored deep solver used by run_cases.py.
    #
    # Options:
    #   auto  : use the half-y solver only when the selected full-y cover is
    #           symmetric and the full grid contains an exact y=0 row.
    #   never : always use the domain requested by full_domain.
    #   force : require the selected cover to be eligible for the half-y solver;
    #           raise an error otherwise.
    symmetric_solver_mode: str = os.environ.get("FLEXURAL_COVER_SYMMETRIC_SOLVER", "auto").strip().lower()
    symmetry_atol: float = float(os.environ.get("FLEXURAL_COVER_SYMMETRY_ATOL", "1e-10"))
    symmetry_rtol: float = float(os.environ.get("FLEXURAL_COVER_SYMMETRY_RTOL", "1e-8"))

    # Physical regime. Run tags are generated from these values by
    # waveslab.steady.names.flexural_cover_run_tag.
    Fr: float = 0.7
    aleph: float = 0.5
    mu: float = 0.0
    tauf: float = 0.0
    epsilon: float = 1.
    Lx: float = 1.0
    Ly: float = 1.0
    N: int = 240
    M: int = 120
    dx: float = 0.2
    dy: float = 0.4
    x0: float = -24.0
    upstream_bc: str = "centered"
    pad_mode: str = "zeros"
    use_radiation: bool = False

    # Cover -> rigidity mapping.
    rigidity_min: float = 0.08
    rigidity_max_scale: float = 1.0  # upper bound = rigidity_max_scale * aleph

    # Solver controls.
    skip_if_exists: bool = True
    f_tol_x64: float = 1e-10
    f_tol_x32: float = 1e-8
    maxiter: int = 50
    line_search: str = "wolfe"


COVER_CFG = CoverBuildConfig()
COVER_SELECT_CFG = CoverSelectionConfig()
RUN_PLAN_CFG = CoverRunPlanConfig()
CROSS_SECTION_CFG = CrossSectionPlotConfig()
CAL_CFG = CalibrationConfig()
SOLVER_CFG = SolverConfig()


def _calibration_options() -> CalibrationOptions:
    """Convert runner config into the unified cover-core calibration options."""
    return CalibrationOptions(
        enabled=bool(CAL_CFG.enabled),
        target_lambda_flex_m=float(CAL_CFG.target_lambda_flex_m),
        reference_stat=CAL_CFG.reference_stat,  # type: ignore[arg-type]
        min_area_px=int(CAL_CFG.min_area_px),
        match_tol_rel=float(CAL_CFG.match_tol_rel),
        max_scale_iters=int(CAL_CFG.max_scale_iters),
        preserve_mean_cover=bool(CAL_CFG.preserve_mean_cover),
        meters_per_pixel=CAL_CFG.meters_per_pixel,
        meters_per_pixel_fallback=float(CAL_CFG.meters_per_pixel_fallback),
    )


def _has_sam_pipeline_inputs(path: Path) -> bool:
    return (
        (path / "data" / "pipeline_arrays.npz").exists()
        and (path / "data" / "floe_catalog.csv").exists()
    )


def resolve_sam_pipeline_dir() -> Path:
    """Find the SAM single-image pipeline output directory to read."""
    candidates: list[Path] = []

    for env_name in ("SAM_PIPELINE_DIR", "FLEXURAL_COVER_PIPELINE_DIR"):
        env_dir = os.environ.get(env_name)
        if env_dir:
            candidates.append(Path(env_dir).expanduser())

    candidates.extend(
        [
            REPO_ROOT / "pythonwaves" / "segment_anything" / "new_seaice" / "results" / "sam_single_image_vscode",
            REPO_ROOT / "pythonwaves" / "segment_anything" / "new_seaice" / "results" / "sam_single_seaice",
            HERE.parent.parent / "segment_anything" / "new_seaice" / "results" / "sam_single_image_vscode",
            HERE.parent.parent / "segment_anything" / "new_seaice" / "results" / "sam_single_seaice",
        ]
    )

    seen: set[Path] = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if _has_sam_pipeline_inputs(cand):
            return cand

    checked = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find a SAM pipeline output directory containing "
        "data/pipeline_arrays.npz and data/floe_catalog.csv. Checked:\n"
        f"  - {checked}\n\n"
        "Set SAM_PIPELINE_DIR=/path/to/results/sam_single_image_vscode if needed."
    )


def resolve_workflow_output_root(sam_pipeline_dir: Path) -> Path:
    """Directory where cover models, solver runs, and figures are written."""
    env_dir = os.environ.get("FLEXURAL_COVER_OUTPUT_ROOT")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    return (
        REPO_ROOT
        / "pythonwaves"
        / "FlexuralCovers"
        / "new_seaice"
        / "results"
        / sam_pipeline_dir.name
    )


def resolve_pipeline_dir() -> Path:
    """Backward-compatible alias for older callers."""
    return resolve_sam_pipeline_dir()


def fmt_hms_ms(seconds: float) -> str:
    s = float(seconds)
    h = int(s // 3600)
    s -= 3600 * h
    m = int(s // 60)
    s -= 60 * m
    return f"{h:d}:{m:02d}:{s:06.3f}"


def _make_grid(cfg: SolverConfig):
    if cfg.full_domain:
        return make_full_grid(cfg.N, cfg.M, cfg.dx, cfg.dy, x0=float(cfg.x0), y0=cfg.full_domain_y0)
    return make_half_grid(cfg.N, cfg.M, cfg.dx, cfg.dy, x0=float(cfg.x0))


def _normalise_symmetric_solver_mode(raw: str) -> str:
    """Normalize the domain shortcut mode while accepting common aliases."""
    mode = str(raw).strip().lower()
    aliases = {
        "0": "never",
        "false": "never",
        "off": "never",
        "no": "never",
        "full": "never",
        "full_domain": "never",
        "1": "auto",
        "true": "auto",
        "on": "auto",
        "yes": "auto",
        "half": "force",
        "half_domain": "force",
        "symmetric": "force",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"auto", "never", "force"}:
        raise ValueError(
            "SolverConfig.symmetric_solver_mode must be one of "
            "'auto', 'never', or 'force' "
            f"(got {raw!r})."
        )
    return mode


def _default_full_domain_y0(M: int, dy: float) -> float:
    """Match waveslab.steady.full_domain.make_full_grid(y0=None)."""
    return -0.5 * float(dy) * (int(M) - 1)


def _full_y_grid_has_exact_centerline(cfg: SolverConfig, M: int) -> tuple[bool, float, float]:
    """Return whether the full-y grid has a row exactly at y=0.

    The half-domain mirrored deep solver in run_cases.py expects y = 0, dy,
    2dy, ... . Therefore an exact extraction from a full-domain grid requires
    odd M and a centre row at y=0. For even M the full grid is centred about
    y=0 but has two rows straddling the centreline; using the half solver would
    silently shift/interpolate the cover, so auto mode keeps the full solver.
    """
    M = int(M)
    y0 = float(cfg.full_domain_y0) if cfg.full_domain_y0 is not None else _default_full_domain_y0(M, cfg.dy)
    if M % 2 == 0:
        return False, y0, y0 + 0.5 * float(cfg.dy) * (M - 1)
    mid = M // 2
    y_mid = y0 + float(cfg.dy) * mid
    tol = max(1e-12, 1e-10 * max(1.0, abs(float(cfg.dy))))
    return bool(abs(y_mid) <= tol), y0, y_mid


def _cover_symmetry_report(F: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    """Check mirror symmetry across y=0 by comparing rows with row reversal."""
    arr = np.asarray(F, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D cover array, got shape {arr.shape!r}.")
    reflected = arr[::-1, :]
    diff = np.abs(arr - reflected)
    finite = np.isfinite(diff)
    max_abs = float(np.max(diff[finite])) if np.any(finite) else float("nan")
    scale = float(max(np.nanmax(np.abs(arr)) if arr.size else 0.0, 1.0))
    return {
        "is_symmetric": bool(np.allclose(arr, reflected, atol=float(atol), rtol=float(rtol), equal_nan=True)),
        "max_abs_y_mirror_difference": max_abs,
        "relative_to_cover_scale": float(max_abs / scale) if np.isfinite(max_abs) else float("nan"),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _select_effective_solver_domain(
    F: np.ndarray,
    cfg: SolverConfig,
    *,
    cover_case_label: str,
) -> tuple[SolverConfig, np.ndarray, dict[str, Any]]:
    """Choose the full-y or half-y solver and return the matching cover array.

    The selected cover NPZs are built on the solver-scale grid requested by
    SolverConfig. If a full-y cover is mirror-symmetric and the full grid has
    an exact y=0 row, we can extract the non-negative-y half and run the same
    mirrored deep system used by run_cases.py. Asymmetric covers, or even-M
    full grids without a y=0 row, stay on the full-domain path.
    """
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2:
        raise ValueError(f"Expected F_low to be a 2D array, got shape {F_arr.shape!r}.")

    requested_shape = (int(cfg.M), int(cfg.N))
    if F_arr.shape != requested_shape:
        raise ValueError(
            "Selected solver cover shape does not match SolverConfig: "
            f"F_low.shape={F_arr.shape}, expected {requested_shape}. "
            "Check COVER_CFG.N/M versus SOLVER_CFG.N/M."
        )

    mode = _normalise_symmetric_solver_mode(cfg.symmetric_solver_mode)
    decision: dict[str, Any] = {
        "cover_case_label": cover_case_label,
        "symmetric_solver_mode": mode,
        "requested_full_domain": bool(cfg.full_domain),
        "input_cover_shape": list(F_arr.shape),
        "used_symmetric_half_domain": False,
        "reason": "using requested solver domain",
    }

    if not cfg.full_domain:
        decision.update(
            {
                "reason": "SolverConfig.full_domain is False, so the half-y mirrored solver was requested directly",
                "solver_cover_shape": list(F_arr.shape),
                "effective_M": int(cfg.M),
                "effective_full_domain": False,
            }
        )
        return cfg, F_arr, decision

    sym = _cover_symmetry_report(F_arr, atol=cfg.symmetry_atol, rtol=cfg.symmetry_rtol)
    has_centerline, y0, y_mid = _full_y_grid_has_exact_centerline(cfg, F_arr.shape[0])
    decision.update(
        {
            "cover_y_symmetric": sym["is_symmetric"],
            "symmetry_report": sym,
            "full_grid_has_exact_y0_row": bool(has_centerline),
            "full_grid_y0": float(y0),
            "full_grid_center_y": float(y_mid),
        }
    )

    if mode == "never":
        decision.update(
            {
                "reason": "symmetric_solver_mode='never'",
                "solver_cover_shape": list(F_arr.shape),
                "effective_M": int(cfg.M),
                "effective_full_domain": True,
            }
        )
        return cfg, F_arr, decision

    if not sym["is_symmetric"]:
        msg = (
            "selected cover is not symmetric under y-row reversal "
            f"(max |F-F_reflected|={sym['max_abs_y_mirror_difference']:.3e})"
        )
        if mode == "force":
            raise ValueError(
                "Cannot use the half-y symmetric solver because the " + msg + "."
            )
        decision.update(
            {
                "reason": msg,
                "solver_cover_shape": list(F_arr.shape),
                "effective_M": int(cfg.M),
                "effective_full_domain": True,
            }
        )
        return cfg, F_arr, decision

    if not has_centerline:
        msg = (
            "selected cover is symmetric, but the full-y grid has no exact y=0 row; "
            "use an odd full-domain M = 2*M_half - 1 for an exact half-domain extraction"
        )
        if mode == "force":
            raise ValueError("Cannot use the half-y symmetric solver: " + msg + ".")
        decision.update(
            {
                "reason": msg,
                "solver_cover_shape": list(F_arr.shape),
                "effective_M": int(cfg.M),
                "effective_full_domain": True,
            }
        )
        return cfg, F_arr, decision

    mid = F_arr.shape[0] // 2
    F_half = np.asarray(F_arr[mid:, :], dtype=float)
    half_cfg = replace(cfg, full_domain=False, M=int(F_half.shape[0]), full_domain_y0=None)
    decision.update(
        {
            "reason": "selected cover is y-symmetric and the full grid has an exact y=0 row",
            "used_symmetric_half_domain": True,
            "solver_cover_shape": list(F_half.shape),
            "source_full_M": int(cfg.M),
            "effective_M": int(half_cfg.M),
            "effective_full_domain": False,
            "extracted_rows": [int(mid), int(F_arr.shape[0] - 1)],
        }
    )
    return half_cfg, F_half, decision


def _run_wave_case(
    F: np.ndarray,
    run_dir: Path,
    cfg: SolverConfig,
    *,
    domain_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the steady flexural case for a solver-scale cover field F."""
    run_dir.mkdir(parents=True, exist_ok=True)

    F_arr = np.asarray(F, dtype=float)
    expected_shape = (int(cfg.M), int(cfg.N))
    if F_arr.shape != expected_shape:
        raise ValueError(
            f"Solver cover shape {F_arr.shape} does not match effective solver grid "
            f"{expected_shape}."
        )

    bounds = (float(cfg.rigidity_min), float(cfg.rigidity_max_scale) * float(cfg.aleph))
    rigidity = RigidityField.from_cover_array(F_arr, bounds=bounds)
    aleph_full = np.asarray(rigidity.aleph, dtype=np.float32)
    aleph_half = 0.5 * (aleph_full[:, 1:] + aleph_full[:, :-1])

    jfloat = jnp.float64 if _USE_X64 else jnp.float32
    policy = BiharmonicPolicy(upstream_bc=cfg.upstream_bc, mode=cfg.pad_mode)

    if cfg.full_domain:
        grid = make_full_grid(cfg.N, cfg.M, cfg.dx, cfg.dy, x0=float(cfg.x0), y0=cfg.full_domain_y0)
        pressure_fn = pressure_full
        residual_fn = jax_residual_deep_full_domain
        initial_guess_fn = initial_guess_full
        reshaping2Unknowns_fn = reshaping2Unknowns_full
        allVals_fn = allVals_full
        build_blocks_fn = lambda: build_blocks_infinite_depth_full_domain(
            N=int(cfg.N),
            M=int(cfg.M),
            dx=float(cfg.dx),
            dy=float(cfg.dy),
            grid=grid,
            Fr=float(cfg.Fr),
            mu=float(cfg.mu),
            use_radiation=bool(cfg.use_radiation),
            policy=policy,
            method=cfg.block_builder,
        )
        flexural_fn = flexural_contribution_full_domain
        domain_mode = "full_y"
    else:
        grid = make_half_grid(cfg.N, cfg.M, cfg.dx, cfg.dy, x0=float(cfg.x0))
        pressure_fn = pressure_half
        residual_fn = jax_residual_deep_half
        initial_guess_fn = initial_guess_half
        reshaping2Unknowns_fn = reshaping2Unknowns_half
        allVals_fn = allVals_half
        build_blocks_fn = lambda: build_blocks_infinite_depth_half(
            N=int(cfg.N),
            M=int(cfg.M),
            dx=float(cfg.dx),
            dy=float(cfg.dy),
            grid=grid,
            Fr=float(cfg.Fr),
            mu=float(cfg.mu),
            use_radiation=bool(cfg.use_radiation),
        )
        flexural_fn = flexural_contribution_half
        domain_mode = "half_y_symmetric"

    xm = jnp.asarray(grid.xm, dtype=jfloat)
    y1d = jnp.asarray(np.ravel(np.asarray(grid.y)), dtype=jfloat)
    pm = pressure_fn(xm, y1d, eps=float(cfg.epsilon), Lx=float(cfg.Lx), Ly=float(cfg.Ly))

    params = dict(
        M=int(cfg.M),
        N=int(cfg.N),
        dx=float(cfg.dx),
        dy=float(cfg.dy),
        Fr=float(cfg.Fr),
        aleph=jnp.asarray(aleph_half, dtype=jfloat),
        tauf=float(cfg.tauf),
        mu=float(cfg.mu),
        x=jnp.asarray(grid.x, dtype=jfloat),
        xm=xm,
        y=y1d,
        pm=jnp.asarray(pm, dtype=jfloat),
        use_radiation=bool(cfg.use_radiation),
        policy=policy,
    )

    def residual_np(u_flat: np.ndarray) -> np.ndarray:
        u_j = jnp.asarray(u_flat, dtype=jfloat)
        r = residual_fn(u_j, **params)
        return np.asarray(jax.device_get(r), dtype=np.float64 if _USE_X64 else np.float32)

    u0 = np.ravel(np.asarray(initial_guess_fn(M=int(cfg.M), N=int(cfg.N), x0=float(cfg.x0))))
    _ = residual_np(u0)

    print(f"[domain] {domain_mode}")
    print(
        f"[grid] x in [{float(np.ravel(grid.x)[0]):.6g}, {float(np.ravel(grid.x)[-1]):.6g}], "
        f"y in [{float(np.ravel(grid.y)[0]):.6g}, {float(np.ravel(grid.y)[-1]):.6g}]"
    )
    if cfg.full_domain:
        print(f"[blocks] {cfg.block_builder}")

    solver_dtype = np.float64 if _USE_X64 else np.float32
    initial_norm2 = float(np.linalg.norm(residual_np(u0)))
    print("[check] ||F(u0)||2 =", initial_norm2)

    # Match waveslab.steady.driver._run_case_2:
    #   base blocks + flexural Jacobian in B -> LDU(2x2) preconditioner
    #   -> NewtonKrylovSolver(method="lgmres", preconditioner=M_prec).
    # Here aleph_half may be spatially varying because it comes from the cover.
    t_prec0 = time.time()
    A, B0, C, D = build_blocks_fn()
    Bflex = flexural_fn(
        N=int(cfg.N),
        M=int(cfg.M),
        dx=float(cfg.dx),
        dy=float(cfg.dy),
        aleph=np.asarray(aleph_half, dtype=solver_dtype),
        policy=policy,
        tauf=float(cfg.tauf),
    )
    B = np.asarray(B0, dtype=solver_dtype) + np.asarray(Bflex, dtype=solver_dtype)
    M_prec = ldu2_prec_operator(A, B, C, D)
    t_prec1 = time.time()
    print("[timer] preconditioner:", f"{t_prec1 - t_prec0:.3f}s")

    solver = NewtonKrylovSolver(method="lgmres", verbose=True)
    t0 = time.time()
    u_sol = solver.solve(
        residual_np,
        np.asarray(u0, dtype=solver_dtype),
        preconditioner=M_prec,
        f_tol=float(cfg.f_tol_x64 if _USE_X64 else cfg.f_tol_x32),
        maxiter=int(cfg.maxiter),
        line_search=cfg.line_search,
        callback=lambda x, f: print(f"[nk] ||F||_inf={np.linalg.norm(f, ord=np.inf):.3e}"),
    )
    t1 = time.time()
    final_norm2 = float(np.linalg.norm(residual_np(u_sol)))
    print("[timer] solve:", f"{t1 - t0:.3f}s")
    print("[done]  ||F(u*)||2 =", final_norm2)

    u_sol_j = jnp.asarray(u_sol, dtype=jfloat)
    _, _, zeta1, zetax = reshaping2Unknowns_fn(u_sol_j, int(cfg.M), int(cfg.N))
    Z = np.asarray(
        jax.device_get(allVals_fn(zeta1, zetax, float(cfg.dx), int(cfg.M), int(cfg.N))),
        dtype=np.float64 if _USE_X64 else np.float32,
    )

    np.save(run_dir / "zeta.npy", Z)
    np.save(run_dir / "cover_F.npy", np.asarray(F_arr, dtype=np.float32))
    np.save(run_dir / "aleph_full.npy", aleph_full)
    np.save(run_dir / "aleph_half.npy", aleph_half)
    np.savetxt(run_dir / "zeta.csv", Z.reshape(-1), fmt="%.16e")

    meta = {
        "Fr": cfg.Fr,
        "aleph": cfg.aleph,
        "mu": cfg.mu,
        "tauf": cfg.tauf,
        "epsilon": cfg.epsilon,
        "Lx": cfg.Lx,
        "Ly": cfg.Ly,
        "N": cfg.N,
        "M": cfg.M,
        "dx": cfg.dx,
        "dy": cfg.dy,
        "x0": cfg.x0,
        "full_domain": bool(cfg.full_domain),
        "full_domain_y0": cfg.full_domain_y0,
        "block_builder": cfg.block_builder if cfg.full_domain else "half_domain_analytic",
        "domain_mode": domain_mode,
        "y_min": float(np.ravel(grid.y)[0]),
        "y_max": float(np.ravel(grid.y)[-1]),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "wall_time_s": float(t1 - t0),
        "wall_time_hms": fmt_hms_ms(t1 - t0),
        "preconditioner_time_s": float(t_prec1 - t_prec0),
        "solve_time_s": float(t1 - t0),
        "initial_residual_norm2": float(initial_norm2),
        "final_residual_norm2": float(final_norm2),
        "solver_method": "lgmres",
        "preconditioner": "ldu2_prec_operator",
        "cover_mean": float(np.mean(F_arr)),
        "cover_min": float(np.min(F_arr)),
        "cover_max": float(np.max(F_arr)),
        "rigidity_bounds": [float(bounds[0]), float(bounds[1])],
        "regime_lambda": float(cfg.aleph * cfg.Fr**3),
        "lambda_star": float(27.0 / 256.0),
        "symmetric_domain_decision": domain_decision or {},
    }
    (run_dir / "meta.json").write_text(json.dumps(json_ready(meta), indent=2), encoding="utf-8")

    return {
        "run_dir": run_dir,
        "grid_x": np.asarray(grid.x, dtype=float),
        "grid_y": np.asarray(grid.y).reshape(-1),
        "Z": Z,
        "F": np.asarray(F_arr, dtype=float),
        "meta": meta,
    }


def _load_existing_run(run_dir: Path, cfg: SolverConfig) -> dict[str, Any]:
    Z = np.load(run_dir / "zeta.npy")
    F = np.load(run_dir / "cover_F.npy")
    grid = _make_grid(cfg)
    return {
        "run_dir": run_dir,
        "grid_x": np.asarray(grid.x, dtype=float),
        "grid_y": np.asarray(grid.y).reshape(-1),
        "Z": Z,
        "F": F,
        "meta": json.loads((run_dir / "meta.json").read_text(encoding="utf-8")),
    }



def _public_case_summary(case_summary: dict[str, Any]) -> dict[str, Any]:
    """Drop in-memory arrays before writing the batch JSON summary."""
    return {k: v for k, v in case_summary.items() if not k.startswith("_")}


def _cross_section_plot_data(case_summary: dict[str, Any]) -> dict[str, Any] | None:
    """Return the in-memory plot payload for one completed case, if available."""
    data = case_summary.get("_plot_data")
    return data if isinstance(data, dict) else None


def _comparison_plot_payload(
    *,
    label: str,
    run_tag: str,
    run_dir: Path,
    run_info: dict[str, Any],
    native_domain_mode: str,
) -> dict[str, Any]:
    """Build cross-section payload, mirroring half-y results when useful.

    Solver output from the half-domain path is stored only on y >= 0.  For
    cross-case comparison against full-domain asymmetric cases, it is more
    useful to expand that result to the corresponding full-y array here while
    still keeping the actual saved solver output compact.
    """
    X1d = np.asarray(run_info["grid_x"], dtype=float)
    Y1d = np.asarray(run_info["grid_y"], dtype=float).reshape(-1)
    Z = np.asarray(run_info["Z"], dtype=float)
    domain_mode = native_domain_mode

    if native_domain_mode == "mirror_y0" and Y1d.size and abs(float(Y1d[0])) <= 1e-12:
        Y1d = np.concatenate([-Y1d[:0:-1], Y1d])
        Z = np.vstack([Z[:0:-1, :][::-1, :], Z])
        domain_mode = "full"

    return {
        "label": label,
        "selected_cover_case": label,
        "run_tag": run_tag,
        "run_dir": run_dir,
        "X1d": X1d,
        "Y1d": Y1d,
        "Z": Z,
        "domain_mode": domain_mode,
        "native_domain_mode": native_domain_mode,
    }


def _save_multi_case_cross_sections(
    *,
    workflow_root: Path,
    case_summaries: list[dict[str, Any]],
) -> dict[str, Path]:
    """Save x/y cross-section comparison plots when at least two cases completed."""
    if not CROSS_SECTION_CFG.enabled:
        print("[cross-sections] disabled")
        return {}

    plot_cases = [
        data
        for item in case_summaries
        if (data := _cross_section_plot_data(item)) is not None
    ]
    if len(plot_cases) < 2:
        print("[cross-sections] skipped: fewer than two completed cases")
        return {}

    domain_modes = {str(data.get("domain_mode", "")) for data in plot_cases}
    domain_modes.discard("")
    if len(domain_modes) > 1:
        print(
            "[cross-sections] skipped: completed cases used mixed domain modes "
            f"{sorted(domain_modes)}"
        )
        return {}

    plot_domain_mode = next(iter(domain_modes), "full" if SOLVER_CFG.full_domain else "mirror_y0")
    out_dir = workflow_root / "runs" / "comparison_plots"
    paths = save_publication_cross_section_comparison(
        cases=plot_cases,
        out_dir=out_dir,
        stem=CROSS_SECTION_CFG.out_stem,
        domain_mode=plot_domain_mode,
        slice_kind=CROSS_SECTION_CFG.slice_kind,
    )
    return paths


def _run_selected_cover_case(
    *,
    requested_cover_case: str,
    workflow_root: Path,
    cover_dir: Path,
    direct_npz: Path,
    ellipse_npz: Path,
    direct_cover: dict[str, Any],
    ellipse_cover: dict[str, Any],
    sam_pipeline_dir: Path,
    cover_meta: dict[str, Any],
    cover_N: int,
    cover_M: int,
) -> dict[str, Any]:
    """Select one cover, run/reload the solver, save plots, and write a final summary."""
    calibrated_npz = cover_dir / "cover_direct_lambda_flex_matched.npz"

    # 3. Select the cover field used by the solver.  The cover-core package
    # owns the case normalization, homogeneous baseline construction, and
    # optional lambda_flex calibration so runner logic stays focused on solving.
    selected_cover = select_cover_for_case(
        cover_case=requested_cover_case,
        direct_npz=direct_npz,
        ellipse_npz=ellipse_npz,
        cover_dir=cover_dir,
        direct_cover=direct_cover,
        ellipse_cover=ellipse_cover,
        sam_pipeline_dir=sam_pipeline_dir,
        homogeneous_value=COVER_SELECT_CFG.homogeneous_value,
        calibration=_calibration_options(),
    )
    solver_cover = selected_cover.cover
    solver_cover_npz = selected_cover.npz_path
    cover_case_label = selected_cover.label
    cover_source = selected_cover.source
    scale_summary = selected_cover.scale_summary
    scale_summary_json = selected_cover.scale_summary_json
    print(f"[cover] requested case = {requested_cover_case}")
    print(f"[cover] selected case  = {cover_case_label}")
    print(f"[cover] selected npz   = {solver_cover_npz}")

    # 4. Decide whether this particular cover can use the cheaper symmetric
    # half-domain system. This is case-specific: homogeneous covers may use it,
    # while direct/ellipse covers remain full-domain unless they are actually
    # mirror-symmetric about y=0.
    F_low_native = np.asarray(solver_cover["F_low"], dtype=float)
    effective_solver_cfg, F_low, domain_decision = _select_effective_solver_domain(
        F_low_native,
        SOLVER_CFG,
        cover_case_label=cover_case_label,
    )
    print(f"[domain] decision      = {domain_decision['reason']}")
    if domain_decision.get("used_symmetric_half_domain"):
        print(
            "[domain] using half-y mirrored solver: "
            f"full M={SOLVER_CFG.M} -> half M={effective_solver_cfg.M}"
        )

    run_tag = flexural_cover_run_tag(
        N=effective_solver_cfg.N,
        M=effective_solver_cfg.M,
        dx=effective_solver_cfg.dx,
        dy=effective_solver_cfg.dy,
        x0=effective_solver_cfg.x0,
        Fr=effective_solver_cfg.Fr,
        aleph=effective_solver_cfg.aleph,
        epsilon=effective_solver_cfg.epsilon,
        mu=effective_solver_cfg.mu,
        use_radiation=effective_solver_cfg.use_radiation,
        full_domain=effective_solver_cfg.full_domain,
        cover_case_label=cover_case_label,
        rigidity_tag="F1",
        cover_source=cover_source,
    )
    run_dir = workflow_root / "runs" / run_tag

    # 5. Run or reload solver.
    if (
        effective_solver_cfg.skip_if_exists
        and (run_dir / "zeta.npy").exists()
        and (run_dir / "cover_F.npy").exists()
        and (run_dir / "meta.json").exists()
    ):
        print(f"[skip] existing run found: {run_dir}")
        run_info = _load_existing_run(run_dir, effective_solver_cfg)
    else:
        run_info = _run_wave_case(
            F_low,
            run_dir,
            effective_solver_cfg,
            domain_decision=domain_decision,
        )

    # 6. Save one publication-style surface-cover figure.
    plot_domain_mode = "full" if effective_solver_cfg.full_domain else "mirror_y0"
    surface_path = flexural_cover_surface_path(
        run_dir,
        run_tag=run_tag,
        domain_mode=plot_domain_mode,
    )
    save_publication_surface_with_cover(
        out_png=surface_path,
        X1d=run_info["grid_x"],
        Y1d=run_info["grid_y"],
        Z=run_info["Z"],
        cover=run_info["F"],
        domain_mode=plot_domain_mode,
    )

    final_summary = {
        "sam_pipeline_dir": sam_pipeline_dir,
        "workflow_output_root": workflow_root,
        "cover_build_summary_json": cover_meta.get("summary_json", cover_dir / "cover_build_summary.json"),
        "requested_cover_case": requested_cover_case,
        "selected_cover_case": cover_case_label,
        "selected_cover_npz": solver_cover_npz,
        "calibration_enabled": bool(cover_case_label.endswith("lambda_flex_matched")),
        "scale_summary_json": scale_summary_json,
        "scale_summary": scale_summary,
        "run_dir": run_dir,
        "run_tag": run_tag,
        "surface_plot": surface_path,
        "publication_surface": surface_path,
        "cover_npzs": {
            "direct": direct_npz,
            "ellipse": ellipse_npz,
            "direct_lambda_flex_matched": calibrated_npz if calibrated_npz.exists() else None,
            "ellipse_lambda_flex_matched": cover_dir / "cover_ellipse_lambda_flex_matched.npz"
            if (cover_dir / "cover_ellipse_lambda_flex_matched.npz").exists()
            else None,
            "selected": solver_cover_npz,
        },
        "solver_outputs": {
            "zeta": run_dir / "zeta.npy",
            "cover_F": run_dir / "cover_F.npy",
            "meta": run_dir / "meta.json",
        },
        "achieved_reference_diameter_m": scale_summary.get("achieved_reference_diameter_m"),
        "target_lambda_flex_m": CAL_CFG.target_lambda_flex_m if cover_case_label.endswith("lambda_flex_matched") else None,
        "Fr": effective_solver_cfg.Fr,
        "aleph": effective_solver_cfg.aleph,
        "regime_lambda": float(effective_solver_cfg.aleph * effective_solver_cfg.Fr**3),
        "lambda_star": float(27.0 / 256.0),
        "full_domain": bool(effective_solver_cfg.full_domain),
        "requested_full_domain": bool(SOLVER_CFG.full_domain),
        "block_builder": effective_solver_cfg.block_builder if effective_solver_cfg.full_domain else "half_domain_analytic",
        "symmetric_domain_decision": domain_decision,
        "cover_config": asdict(COVER_CFG),
        "cover_selection_config": asdict(COVER_SELECT_CFG),
        "run_plan_config": asdict(RUN_PLAN_CFG),
        "effective_cover_grid": {"N": cover_N, "M": cover_M},
        "effective_solver_grid": {"N": effective_solver_cfg.N, "M": effective_solver_cfg.M},
        "calibration_config": asdict(CAL_CFG),
        "solver_config": asdict(effective_solver_cfg),
        "base_solver_config": asdict(SOLVER_CFG),
    }
    final_summary_path = run_dir / "final_summary.json"
    final_summary_path.write_text(json.dumps(json_ready(final_summary), indent=2), encoding="utf-8")

    print("\n--- flexural-cover case summary ---")
    print(f"requested cover case  = {requested_cover_case}")
    print(f"selected cover case   = {cover_case_label}")
    print(f"selected cover npz    = {solver_cover_npz}")
    if cover_case_label.endswith("lambda_flex_matched"):
        print(f"target lambda_flex [m] = {CAL_CFG.target_lambda_flex_m:.6f}")
        achieved = scale_summary.get("achieved_reference_diameter_m")
        if achieved is not None:
            print(f"achieved D_ref [m]     = {float(achieved):.6f}")
        print(f"best scale             = {float(scale_summary.get('best_scale', 1.0)):.6f}")
    else:
        print("calibration            = skipped")
    if cover_case_label.startswith("homogeneous"):
        print(f"homogeneous F          = {float(scale_summary['homogeneous_value']):.6f}")
    print(f"Fr                     = {effective_solver_cfg.Fr:.6f}")
    print(f"aleph                  = {effective_solver_cfg.aleph:.6f}")
    print(f"lambda                 = {effective_solver_cfg.aleph * effective_solver_cfg.Fr**3:.6f}")
    print(f"domain mode            = {plot_domain_mode}")
    print(f"[saved] {surface_path}")
    print(f"[saved] {final_summary_path}")

    return {
        "requested_cover_case": requested_cover_case,
        "selected_cover_case": cover_case_label,
        "run_tag": run_tag,
        "run_dir": run_dir,
        "selected_cover_npz": solver_cover_npz,
        "surface_plot": surface_path,
        "final_summary": final_summary_path,
        "calibration_enabled": bool(cover_case_label.endswith("lambda_flex_matched")),
        "domain_mode": plot_domain_mode,
        "used_symmetric_half_domain": bool(domain_decision.get("used_symmetric_half_domain", False)),
        "effective_solver_M": int(effective_solver_cfg.M),
        "requested_solver_M": int(SOLVER_CFG.M),
        "symmetric_domain_decision": domain_decision,
        "cover_mean": float(np.mean(np.asarray(F_low, dtype=float))),
        "cover_min": float(np.min(np.asarray(F_low, dtype=float))),
        "cover_max": float(np.max(np.asarray(F_low, dtype=float))),
        "_plot_data": _comparison_plot_payload(
            label=cover_case_label,
            run_tag=run_tag,
            run_dir=run_dir,
            run_info=run_info,
            native_domain_mode=plot_domain_mode,
        ),
    }


def main() -> None:
    sam_pipeline_dir = resolve_sam_pipeline_dir()
    workflow_root = resolve_workflow_output_root(sam_pipeline_dir)
    cover_dir = workflow_root / "cover_models"

    direct_npz = cover_dir / "cover_direct_from_crop.npz"
    ellipse_npz = cover_dir / "cover_from_ellipse_crop.npz"

    print("[repo_root]", REPO_ROOT)
    print("[sam_pipeline_dir]", sam_pipeline_dir)
    print("[workflow_output_root]", workflow_root)
    print("[backend]", jax.default_backend(), "| devices:", jax.devices())

    # 1-2. Read SAM outputs and build direct/ellipse solver covers once.
    cover_N = int(COVER_CFG.N if COVER_CFG.N is not None else SOLVER_CFG.N)
    cover_M = int(COVER_CFG.M if COVER_CFG.M is not None else SOLVER_CFG.M)
    cover_meta = build_cover_npzs_from_sam_pipeline(
        pipeline_dir=sam_pipeline_dir,
        out_dir=cover_dir,
        direct_source=COVER_CFG.direct_source,
        crop_x0=COVER_CFG.crop_x0,
        crop_y0=COVER_CFG.crop_y0,
        crop_w=COVER_CFG.crop_w,
        crop_h=COVER_CFG.crop_h,
        smooth_width_px=COVER_CFG.smooth_width_px,
        N=cover_N,
        M=cover_M,
        write_full_binary_exports=COVER_CFG.write_full_binary_exports,
    )

    direct_cover = load_cover_npz(direct_npz)
    ellipse_cover = load_cover_npz(ellipse_npz)

    requested_cases = cover_cases_to_run(
        run_homogeneous=RUN_PLAN_CFG.run_homogeneous_constant,
        run_direct=RUN_PLAN_CFG.run_direct,
        run_ellipse=RUN_PLAN_CFG.run_ellipse,
        use_single_case=RUN_PLAN_CFG.use_single_case,
        single_case=COVER_SELECT_CFG.case,
        calibration_enabled=CAL_CFG.enabled,
    )
    print("[run_plan] use_single_case =", RUN_PLAN_CFG.use_single_case)
    print("[run_plan] requested cover cases =", ", ".join(requested_cases))
    if CAL_CFG.enabled:
        print("[run_plan] calibration enabled: direct/ellipse cases will use lambda_flex_matched covers")
    else:
        print("[run_plan] calibration disabled: direct/ellipse cases use native solver-scale covers")

    case_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, requested_cover_case in enumerate(requested_cases, start=1):
        print("\n" + "=" * 78)
        print(f"[case {index}/{len(requested_cases)}] {requested_cover_case}")
        print("=" * 78)
        try:
            case_summary = _run_selected_cover_case(
                requested_cover_case=requested_cover_case,
                workflow_root=workflow_root,
                cover_dir=cover_dir,
                direct_npz=direct_npz,
                ellipse_npz=ellipse_npz,
                direct_cover=direct_cover,
                ellipse_cover=ellipse_cover,
                sam_pipeline_dir=sam_pipeline_dir,
                cover_meta=cover_meta,
                cover_N=cover_N,
                cover_M=cover_M,
            )
            case_summaries.append(case_summary)
        except Exception as exc:
            failure = {
                "requested_cover_case": requested_cover_case,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"[error] {requested_cover_case}: {type(exc).__name__}: {exc}")
            if not RUN_PLAN_CFG.continue_on_case_error:
                raise

    cross_section_paths: dict[str, Path] = {}
    if len(case_summaries) > 1:
        cross_section_paths = _save_multi_case_cross_sections(
            workflow_root=workflow_root,
            case_summaries=case_summaries,
        )

    completed_cases_for_json = [_public_case_summary(item) for item in case_summaries]

    batch_summary = {
        "sam_pipeline_dir": sam_pipeline_dir,
        "workflow_output_root": workflow_root,
        "cover_dir": cover_dir,
        "requested_cases": requested_cases,
        "completed_cases": completed_cases_for_json,
        "failures": failures,
        "cross_section_comparison_plots": cross_section_paths,
        "cross_section_plot_config": asdict(CROSS_SECTION_CFG),
        "cover_npzs": {
            "direct": direct_npz,
            "ellipse": ellipse_npz,
        },
        "cover_build_summary_json": cover_meta.get("summary_json", cover_dir / "cover_build_summary.json"),
        "run_plan_config": asdict(RUN_PLAN_CFG),
        "cover_selection_config": asdict(COVER_SELECT_CFG),
        "calibration_config": asdict(CAL_CFG),
        "solver_config": asdict(SOLVER_CFG),
        "effective_cover_grid": {"N": cover_N, "M": cover_M},
    }
    batch_summary_path = workflow_root / "runs" / "flexural_cover_multi_case_summary.json"
    batch_summary_path.parent.mkdir(parents=True, exist_ok=True)
    batch_summary_path.write_text(json.dumps(json_ready(batch_summary), indent=2), encoding="utf-8")

    print("\n=== flexural-cover multi-case workflow summary ===")
    print(f"completed cases = {len(case_summaries)} / {len(requested_cases)}")
    for item in case_summaries:
        print(
            f"  - {item['selected_cover_case']}: "
            f"run_dir={item['run_dir']} | surface={item['surface_plot']}"
        )
    if cross_section_paths:
        print("cross-section comparisons:")
        for key, path in cross_section_paths.items():
            print(f"  - {key}: {path}")
    if failures:
        print("failures:")
        for item in failures:
            print(f"  - {item['requested_cover_case']}: {item['error_type']}: {item['error']}")
    print(f"[saved] {direct_npz}")
    print(f"[saved] {ellipse_npz}")
    print(f"[saved] {batch_summary_path}")


if __name__ == "__main__":
    main()