# my-packages/pypure/pywave/waves_helpers/jax_biharmonic.py
from __future__ import annotations

from dataclasses import dataclass
# import os
from functools import partial, lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from .regimes import classify_infinite_depth_regime, classify_finite_depth_regime

UpstreamBC = Literal["onesided", "centered"]
PadMode = Literal["extrapolate", "zeros"]
DepthKind = Literal["infinite", "finite"]


def _upstream_bc_from_regime_string(regime: str) -> UpstreamBC:
    """Map a human-readable regime string to an upstream stencil choice."""
    if ("U > c0" in regime) or ("c_min < U < c0" in regime) or ("c_min < U" in regime):
        return "onesided"
    if ("U < c_min" in regime):
        return "centered"
    raise ValueError(f"Unrecognized regime string: {regime!r}")


def _mode_from_upstream_bc(upstream_bc: UpstreamBC) -> PadMode:
    # Default: onesided -> extrapolate, centered -> zeros
    return "extrapolate" if upstream_bc == "onesided" else "zeros"


# def _get_regime_classifiers():
#     """Lazy import (avoids hard deps / import-time side effects)."""
#     try:
#         # Preferred: within the same `waves_helpers` package.
#         from .regimes import classify_infinite_depth_regime, classify_finite_depth_regime  # type: ignore
#     except Exception:
#         # Fallback: older namespace used in some repos.
#         from pywave.waves_helpers.classify_regimes import (  # type: ignore
#             classify_infinite_depth_regime,
#             classify_finite_depth_regime,
#         )
#     return classify_infinite_depth_regime, classify_finite_depth_regime


@lru_cache(maxsize=None)
def _classify_regime_cached(depth: DepthKind, F: float, aleph: float) -> str:
    # classify_infinite_depth_regime, classify_finite_depth_regime = _get_regime_classifiers()

    if depth == "infinite":
        r = classify_infinite_depth_regime(F=float(F), aleph=float(aleph))
    elif depth == "finite":
        r = classify_finite_depth_regime(F=float(F), aleph=float(aleph))
    else:
        raise ValueError("depth must be 'infinite' or 'finite'")

    regime = r.get("regime", "") if isinstance(r, dict) else str(r)
    return str(regime)


def make_biharmonic_policy_from_params(
    *,
    depth: DepthKind,
    F: float,
    aleph: float,
    upstream_bc: UpstreamBC | None = None,
    mode: PadMode | None = None,
    announce: bool = True,
) -> BiharmonicPolicy:
    """Infer a `BiharmonicPolicy` from (depth, F, aleph) regime classification.

    Overrides:
      - If `upstream_bc` is provided, it overrides the inferred upstream stencil.
      - If `mode` is provided, it overrides the inferred padding mode.
      - If only `upstream_bc` is provided, `mode` is chosen via `_mode_from_upstream_bc`.

    This helper prints a one-line summary when `announce=True`.
    """
    regime = _classify_regime_cached(depth, float(F), float(aleph))

    inferred_upstream = _upstream_bc_from_regime_string(regime)
    inferred_mode = _mode_from_upstream_bc(inferred_upstream)

    chosen_upstream = inferred_upstream if upstream_bc is None else upstream_bc
    if mode is None:
        chosen_mode = inferred_mode if upstream_bc is None else _mode_from_upstream_bc(chosen_upstream)
    else:
        chosen_mode = mode

    policy = BiharmonicPolicy(upstream_bc=chosen_upstream, mode=chosen_mode)

    if announce:
        print(
            "[biharmonic-policy] depth=%s F=%.6g aleph=%.6g | regime=%s | upstream_bc=%s | mode=%s"
            % (depth, float(F), float(aleph), regime, policy.upstream_bc, policy.mode)
        )

    return policy


def resolve_biharmonic_policy(
    *,
    policy: BiharmonicPolicy | None = None,
    upstream_bc: UpstreamBC | None = None,
    mode: PadMode | None = None,
    depth: DepthKind | None = None,
    F: float | None = None,
    aleph: float | None = None,
    announce: bool = False,
) -> BiharmonicPolicy:
    """Resolve a `BiharmonicPolicy` from either explicit flags or (depth, F, aleph).

    Priority:
      1) explicit `policy` (optionally overridden by `upstream_bc` / `mode`)
      2) explicit `upstream_bc` (+ optional `mode`)
      3) inferred from (depth, F, aleph)

    If only `upstream_bc` is given, a consistent default `mode` is chosen.
    If only `mode` is given, we raise (insufficient info).
    """
    if policy is not None:
        chosen_upstream = policy.upstream_bc if upstream_bc is None else upstream_bc
        chosen_mode = policy.mode if mode is None else mode
        resolved = BiharmonicPolicy(upstream_bc=chosen_upstream, mode=chosen_mode)
        if announce:
            print("[biharmonic-policy] explicit | upstream_bc=%s | mode=%s" % (resolved.upstream_bc, resolved.mode))
        return resolved

    if upstream_bc is not None:
        resolved_mode = _mode_from_upstream_bc(upstream_bc) if mode is None else mode
        resolved = BiharmonicPolicy(upstream_bc=upstream_bc, mode=resolved_mode)
        if announce:
            print("[biharmonic-policy] explicit flags | upstream_bc=%s | mode=%s" % (resolved.upstream_bc, resolved.mode))
        return resolved

    if mode is not None:
        raise ValueError("Cannot infer upstream_bc from mode alone. Provide upstream_bc or (depth, F, aleph).")

    if (depth is None) or (F is None) or (aleph is None):
        raise ValueError(
            "Policy not provided. Provide either `policy`, or `upstream_bc` (and optionally `mode`), "
            "or infer from `depth`, `F`, and `aleph`."
        )

    return make_biharmonic_policy_from_params(depth=depth, F=float(F), aleph=float(aleph), announce=announce)


@dataclass(frozen=True)
class BiharmonicPolicy:
    """Policy for the biharmonic/viscoelastic stencil kernels.
    """

    upstream_bc: UpstreamBC
    mode: PadMode

    def __post_init__(self) -> None:
        if self.upstream_bc not in ("onesided", "centered"):
            raise ValueError(f"upstream_bc must be 'onesided' or 'centered', got {self.upstream_bc!r}")
        if self.mode not in ("extrapolate", "zeros"):
            raise ValueError(f"mode must be 'extrapolate' or 'zeros', got {self.mode!r}")

    # --- Flags used by the JAX stencil kernels ---
    @property
    def upstream_stencil(self) -> bool:
        return self.upstream_bc == "onesided"

    @property
    def pad_mode(self) -> bool:
        return self.mode == "extrapolate"

    @property
    def left_pad_zeros(self) -> bool:
        # left-pad zeros for centered, no pad for onesided.
        return self.upstream_bc == "centered"

    @property
    def reflect(self) -> bool:
        return True # always mirror at y=0

def flexural_contribution(
    N: int,
    M: int,
    dx: float,
    dy: float,
    *,
    aleph: float,
    tauf: float,
    policy: BiharmonicPolicy | None = None,
    depth: DepthKind | None = None,
    F: float | None = None,
    upstream_bc: UpstreamBC | None = None,
    mode: PadMode | None = None,
    announce_policy: bool = False,
    chunk_cols: int = 16,
    device: str = "gpu",
    dtype: np.dtype = np.float32,
    jax_dtype=jnp.float32,
) -> np.ndarray:
    """Dense flexural Jacobian via chunked JVPs (streams to host)."""
    policy = resolve_biharmonic_policy(
        policy=policy,
        upstream_bc=upstream_bc,
        mode=mode,
        depth=depth,
        F=F,
        aleph=aleph,
        announce=announce_policy,
    )
    neq = M * (N + 1)
    dev = jax.devices(device)[0]
    with jax.default_device(dev):
        u0 = jnp.zeros((neq,), dtype=jax_dtype)
        J_host = jflex_dense_jacobian(
            u0,
            N, M, dx, dy,
            aleph=aleph,
            policy=policy,
            tauf=tauf,
            chunk_cols=chunk_cols,
            return_numpy=True,
        )
    return np.asarray(J_host, dtype=dtype)


@partial(jax.jit, static_argnames=("M", "N"))
def _reconstruct_zeta_from_slice(u_slice: jnp.ndarray, M: int, N: int, dx: float) -> jnp.ndarray:
    U = u_slice.reshape((M, N + 1))
    zeta1 = U[:, 0]
    gx = U[:, 1:]
    inc = 0.5 * dx * (gx[:, 1:] + gx[:, :-1])
    csum = jnp.cumsum(inc, axis=1)
    csum = jnp.concatenate([jnp.zeros((M, 1), gx.dtype), csum], axis=1)
    return zeta1[:, None] + csum


@partial(jax.jit, static_argnames=("N", "M", "policy", "tauf"))
def ice_sheet(
    u_slice: jnp.ndarray,
    N: int,
    M: int,
    dx: float,
    dy: float,
    aleph: float,
    policy: BiharmonicPolicy,
    *,
    tauf: float,
) -> jnp.ndarray:
    """Flexural (biharmonic + optional difflaplace) residual on the half-mesh."""
    zeta = _reconstruct_zeta_from_slice(u_slice, M, N, dx)
    dlaplace_over2 = sliced_biharmonic_operator_policy(zeta, dx, dy, policy)
    if tauf == 0.0:
        PFlexHalf = aleph * dlaplace_over2
    else:
        difflaplace = sliced_biharmonic_difflaplace_policy(zeta, dx, dy, policy)
        PFlexHalf = aleph * (dlaplace_over2 + tauf * difflaplace)

    zeros_col = jnp.zeros((M, 1), dtype=PFlexHalf.dtype)
    E1 = jnp.concatenate([zeros_col, zeros_col, PFlexHalf], axis=1)
    return E1.reshape(M * (N + 1),)


def jflex_dense_jacobian(
    u: jnp.ndarray,
    N: int,
    M: int,
    dx: float,
    dy: float,
    aleph: float,
    policy: BiharmonicPolicy,
    *,
    tauf: float,
    chunk_cols: int = 16,
    return_numpy: bool = False,
):
    """Exact Jacobian of the flexural residual wrt [zeta1, zeta_x].
    Computes columns in chunks via JVPs to avoid OOM.
    """
    neq = M * (N + 1)
    u_slice = u[:neq]
    f = lambda us: ice_sheet(us, N, M, dx, dy, aleph, policy, tauf=tauf)

    def _jvp_cols(us: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
        def one_col(i):
            v = jax.nn.one_hot(i, neq, dtype=us.dtype)
            _, jv = jax.jvp(f, (us,), (v,))
            return jv
        return jax.vmap(one_col)(idx)

    jvp_cols = jax.jit(_jvp_cols)
    nblocks = (neq + chunk_cols - 1) // chunk_cols

    if return_numpy:
        J_host = np.empty((neq, neq), dtype=np.asarray(u).dtype)
        for b in range(nblocks):
            start = b * chunk_cols
            idx = start + jnp.arange(chunk_cols, dtype=jnp.int32)
            valid = idx < neq
            idx_safe = jnp.where(valid, idx, jnp.int32(0))
            block = jvp_cols(u_slice, idx_safe)  # (chunk, neq)
            block_T = block.T                    # (neq, chunk)
            cols = int(min(chunk_cols, neq - start))
            J_host[:, start:start + cols] = np.asarray(jax.device_get(block_T[:, :cols]))
        return J_host

    blocks = []
    for b in range(nblocks):
        start = b * chunk_cols
        idx = start + jnp.arange(chunk_cols, dtype=jnp.int32)
        valid = idx < neq
        idx_safe = jnp.where(valid, idx, jnp.int32(0))
        block = jvp_cols(u_slice, idx_safe)
        blocks.append(block.T)

    return jnp.concatenate(blocks, axis=1)[:, :neq]

#--------------biharmonic operator construction -----------------------
def _ghost_right_cols(u: jnp.ndarray, pad_mode: bool) -> jnp.ndarray:
    m, n = u.shape

    def _extrap(_):
        g1x = 2.0 * u[:, n - 1] - u[:, n - 2]
        g2x = 2.0 * g1x - u[:, n - 1]
        return jnp.stack([g1x, g2x], axis=1)

    def _zeros(_):
        return jnp.zeros((m, 2), dtype=u.dtype)

    return lax.cond(pad_mode, _extrap, _zeros, operand=None)


def _ghost_bottom_rows(u_right: jnp.ndarray, pad_mode: bool) -> jnp.ndarray:
    m, _ = u_right.shape

    def _extrap(_):
        g1y = 2.0 * u_right[m - 1, :] - u_right[m - 2, :]
        g2y = 2.0 * g1y - u_right[m - 1, :]
        return jnp.stack([g1y, g2y], axis=0)

    def _zeros(_):
        return jnp.zeros((2, u_right.shape[1]), dtype=u_right.dtype)

    return lax.cond(pad_mode, _extrap, _zeros, operand=None)


@partial(jax.jit, static_argnames=("pad_mode", "reflect", "left_pad_zeros"))
def padding_scheme(
    u: jnp.ndarray,
    *,
    pad_mode: bool,
    reflect: bool,
    left_pad_zeros: bool,
) -> jnp.ndarray:
    u = jnp.asarray(u)

    right_cols = _ghost_right_cols(u, pad_mode)
    u_r = jnp.concatenate([u, right_cols], axis=1)

    bottom_rows = _ghost_bottom_rows(u_r, pad_mode)
    u_rb = jnp.concatenate([u_r, bottom_rows], axis=0)

    def _reflect(_):
        top_rows = u_rb[jnp.array([1, 0]), :]
        return jnp.concatenate([top_rows, u_rb], axis=0)

    def _zeros_top(_):
        top_rows = jnp.zeros((2, u_rb.shape[1]), dtype=u.dtype)
        return jnp.concatenate([top_rows, u_rb], axis=0)

    u_t = lax.cond(reflect, _reflect, _zeros_top, operand=None)

    if left_pad_zeros:
        left_cols = jnp.zeros((u_t.shape[0], 2), dtype=u.dtype)
        return jnp.concatenate([left_cols, u_t], axis=1)

    return u_t


def precompute_powers(dx: float, dy: float):
    dx2 = dx * dx
    dy2 = dy * dy
    return dx2 * dx2, dy2 * dy2, dx2 * dy2


def _interior_stencil_gather(
    u_pad: jnp.ndarray,
    j0: int,
    m: int,
    ii: jnp.ndarray,
    dx4: float,
    dx2dy2: float,
    dy4: float,
    *,
    shift: int = 0,
) -> jnp.ndarray:
    s = shift
    rows = j0 + jnp.arange(m)
    rp1 = rows + 1
    rm1 = rows - 1
    rp2 = rows + 2
    rm2 = rows - 2

    def take_rc(R, C):
        return jnp.take(jnp.take(u_pad, R, axis=0), C, axis=1)

    c_2 = take_rc(rows, ii + 2 + s)
    c_1 = take_rc(rows, ii + 1 + s)
    c0 = take_rc(rows, ii + 0 + s)
    cm1 = take_rc(rows, ii - 1 + s)
    cm2 = take_rc(rows, ii - 2 + s)
    u_xxxx = (c_2 - 4.0 * c_1 + 6.0 * c0 - 4.0 * cm1 + cm2) / dx4

    tp1_p1 = take_rc(rp1, ii + 1 + s)
    tp1_m1 = take_rc(rp1, ii - 1 + s)
    tm1_p1 = take_rc(rm1, ii + 1 + s)
    tm1_m1 = take_rc(rm1, ii - 1 + s)

    t0_p1 = take_rc(rows, ii + 1 + s)
    t0_0 = take_rc(rows, ii + 0 + s)
    t0_m1 = take_rc(rows, ii - 1 + s)

    u_xxyy = (
        tp1_p1
        + tp1_m1
        + tm1_p1
        + tm1_m1
        - 2.0 * t0_p1
        - 2.0 * jnp.take(u_pad, rp1, axis=0)[:, ii + 0 + s]
        - 2.0 * t0_m1
        - 2.0 * jnp.take(u_pad, rm1, axis=0)[:, ii + 0 + s]
        + 4.0 * t0_0
    ) / dx2dy2

    u_yyyy = (
        jnp.take(u_pad, rp2, axis=0)[:, ii + 0 + s]
        - 4.0 * jnp.take(u_pad, rp1, axis=0)[:, ii + 0 + s]
        + 6.0 * t0_0
        - 4.0 * jnp.take(u_pad, rm1, axis=0)[:, ii + 0 + s]
        + jnp.take(u_pad, rm2, axis=0)[:, ii + 0 + s]
    ) / dy4

    return u_xxxx + 2.0 * u_xxyy + u_yyyy


def _upstream_onesided(
    u_pad: jnp.ndarray,
    j0: int,
    base_col: int,
    m: int,
    dx4: float,
    dx2dy2: float,
    dy4: float,
) -> jnp.ndarray:
    rows = j0 + jnp.arange(m)
    rp1 = rows + 1
    rm1 = rows - 1
    rp2 = rows + 2
    rm2 = rows - 2

    i0 = base_col
    i1 = i0 + 1
    i2 = i0 + 2
    i3 = i0 + 3
    i4 = i0 + 4

    def take_rc(R, c):
        return jnp.take(jnp.take(u_pad, R, axis=0), jnp.array([c]), axis=1)[:, 0]

    x_term = (
        take_rc(rows, i4)
        - 4.0 * take_rc(rows, i3)
        + 6.0 * take_rc(rows, i2)
        - 4.0 * take_rc(rows, i1)
        + take_rc(rows, i0)
    ) / dx4

    y0 = (
        take_rc(rp2, i0)
        - 4.0 * take_rc(rp1, i0)
        + 6.0 * take_rc(rows, i0)
        - 4.0 * take_rc(rm1, i0)
        + take_rc(rm2, i0)
    ) / dy4
    y1 = (
        take_rc(rp2, i1)
        - 4.0 * take_rc(rp1, i1)
        + 6.0 * take_rc(rows, i1)
        - 4.0 * take_rc(rm1, i1)
        + take_rc(rm2, i1)
    ) / dy4

    mix = (
        jnp.take(u_pad, rp1, axis=0)[:, i2]
        + jnp.take(u_pad, rp1, axis=0)[:, i0]
        + jnp.take(u_pad, rm1, axis=0)[:, i2]
        + jnp.take(u_pad, rm1, axis=0)[:, i0]
        - 2.0 * jnp.take(u_pad, rows, axis=0)[:, i2]
        - 2.0 * jnp.take(u_pad, rp1, axis=0)[:, i1]
        - 2.0 * jnp.take(u_pad, rm1, axis=0)[:, i1]
        - 2.0 * jnp.take(u_pad, rows, axis=0)[:, i0]
        + 4.0 * jnp.take(u_pad, rows, axis=0)[:, i1]
    ) / dx2dy2
    col0 = 2.0 * (x_term + 2.0 * mix) + y0 + y1

    y2 = (
        take_rc(rp2, i2)
        - 4.0 * take_rc(rp1, i2)
        + 6.0 * take_rc(rows, i2)
        - 4.0 * take_rc(rm1, i2)
        + take_rc(rm2, i2)
    ) / dy4
    mix_s = (
        jnp.take(u_pad, rp1, axis=0)[:, i3]
        + jnp.take(u_pad, rp1, axis=0)[:, i1]
        + jnp.take(u_pad, rm1, axis=0)[:, i3]
        + jnp.take(u_pad, rm1, axis=0)[:, i1]
        - 2.0 * jnp.take(u_pad, rows, axis=0)[:, i3]
        - 2.0 * jnp.take(u_pad, rp1, axis=0)[:, i2]
        - 2.0 * jnp.take(u_pad, rm1, axis=0)[:, i2]
        - 2.0 * jnp.take(u_pad, rows, axis=0)[:, i1]
        + 4.0 * jnp.take(u_pad, rows, axis=0)[:, i2]
    ) / dx2dy2
    col1 = (x_term + 2.0 * mix + y1) + (x_term + 2.0 * mix_s + y2)

    return jnp.stack([col0, col1], axis=1)


def _upstream_onesided_difflaplace(
    u_pad: jnp.ndarray,
    j0: int,
    base_col: int,
    m: int,
    dx: float,
    dx2dy2: float,
    dy4: float,
) -> jnp.ndarray:
    rows = j0 + jnp.arange(m)
    rp1 = rows + 1
    rm1 = rows - 1
    rp2 = rows + 2
    rm2 = rows - 2

    i0 = base_col
    i1 = i0 + 1
    i2 = i0 + 2
    i3 = i0 + 3

    def take_rc(R, c):
        return jnp.take(jnp.take(u_pad, R, axis=0), jnp.array([c]), axis=1)[:, 0]

    y0 = (
        take_rc(rp2, i0)
        - 4.0 * take_rc(rp1, i0)
        + 6.0 * take_rc(rows, i0)
        - 4.0 * take_rc(rm1, i0)
        + take_rc(rm2, i0)
    ) / dy4
    y1 = (
        take_rc(rp2, i1)
        - 4.0 * take_rc(rp1, i1)
        + 6.0 * take_rc(rows, i1)
        - 4.0 * take_rc(rm1, i1)
        + take_rc(rm2, i1)
    ) / dy4
    y2 = (
        take_rc(rp2, i2)
        - 4.0 * take_rc(rp1, i2)
        + 6.0 * take_rc(rows, i2)
        - 4.0 * take_rc(rm1, i2)
        + take_rc(rm2, i2)
    ) / dy4

    mix = (
        jnp.take(u_pad, rp1, axis=0)[:, i2]
        + jnp.take(u_pad, rp1, axis=0)[:, i0]
        + jnp.take(u_pad, rm1, axis=0)[:, i2]
        + jnp.take(u_pad, rm1, axis=0)[:, i0]
        - 2.0 * jnp.take(u_pad, rows, axis=0)[:, i2]
        - 2.0 * jnp.take(u_pad, rp1, axis=0)[:, i1]
        - 2.0 * jnp.take(u_pad, rm1, axis=0)[:, i1]
        - 2.0 * jnp.take(u_pad, rows, axis=0)[:, i0]
        + 4.0 * jnp.take(u_pad, rows, axis=0)[:, i1]
    ) / dx2dy2

    mix_s = (
        jnp.take(u_pad, rp1, axis=0)[:, i3]
        + jnp.take(u_pad, rp1, axis=0)[:, i1]
        + jnp.take(u_pad, rm1, axis=0)[:, i3]
        + jnp.take(u_pad, rm1, axis=0)[:, i1]
        - 2.0 * jnp.take(u_pad, rows, axis=0)[:, i3]
        - 2.0 * jnp.take(u_pad, rp1, axis=0)[:, i2]
        - 2.0 * jnp.take(u_pad, rm1, axis=0)[:, i2]
        - 2.0 * jnp.take(u_pad, rows, axis=0)[:, i1]
        + 4.0 * jnp.take(u_pad, rows, axis=0)[:, i2]
    ) / dx2dy2

    col0 = (y1 - y0) / dx
    col1 = (2.0 * (mix_s - mix) + (y2 - y1)) / dx
    return jnp.stack([col0, col1], axis=1)


@partial(jax.jit, static_argnames=("upstream_stencil", "pad_mode", "reflect", "left_pad_zeros"))
def _sliced_biharmonic_operator_flags(
    u: jnp.ndarray,
    dx: float,
    dy: float,
    *,
    upstream_stencil: bool,
    pad_mode: bool,
    reflect: bool,
    left_pad_zeros: bool,
) -> jnp.ndarray:
    u = jnp.asarray(u)
    m, n = u.shape
    assert (m >= 5) and (n >= 5), "Need at least a 5x5 grid for a ±2 stencil."

    u_pad = padding_scheme(
        u,
        pad_mode=pad_mode,
        reflect=reflect,
        left_pad_zeros=left_pad_zeros,
    )
    dx4, dy4, dx2dy2 = precompute_powers(dx, dy)

    j0 = 2
    i0 = 2 if left_pad_zeros else 0
    out = jnp.zeros((m, n - 1), dtype=u.dtype)

    ii_interior = (i0 + 2) + jnp.arange(n - 3)
    interior = (
        _interior_stencil_gather(u_pad, j0, m, ii_interior, dx4, dx2dy2, dy4, shift=0)
        + _interior_stencil_gather(u_pad, j0, m, ii_interior, dx4, dx2dy2, dy4, shift=1)
    )
    out = out.at[:, 2:].set(interior)

    def _do_onesided(_):
        return _upstream_onesided(u_pad, j0, i0, m, dx4, dx2dy2, dy4)

    def _do_centered(_):
        ii_up = (i0 + 0) + jnp.arange(2)
        return (
            _interior_stencil_gather(u_pad, j0, m, ii_up, dx4, dx2dy2, dy4, shift=0)
            + _interior_stencil_gather(u_pad, j0, m, ii_up, dx4, dx2dy2, dy4, shift=1)
        )

    out = out.at[:, :2].set(lax.cond(upstream_stencil, _do_onesided, _do_centered, operand=None))
    return 0.5 * out


@partial(jax.jit, static_argnames=("upstream_stencil", "pad_mode", "reflect", "left_pad_zeros"))
def _sliced_biharmonic_difflaplace_flags(
    u: jnp.ndarray,
    dx: float,
    dy: float,
    *,
    upstream_stencil: bool,
    pad_mode: bool,
    reflect: bool,
    left_pad_zeros: bool,
) -> jnp.ndarray:
    u = jnp.asarray(u)
    m, n = u.shape
    assert (m >= 5) and (n >= 5), "Need at least a 5×5 grid for a ±2 stencil."

    u_pad = padding_scheme(
        u,
        pad_mode=pad_mode,
        reflect=reflect,
        left_pad_zeros=left_pad_zeros,
    )
    dx4, dy4, dx2dy2 = precompute_powers(dx, dy)

    j0 = 2
    i0 = 2 if left_pad_zeros else 0
    out = jnp.zeros((m, n - 1), dtype=u.dtype)

    ii_interior = (i0 + 2) + jnp.arange(n - 3)
    L = _interior_stencil_gather(u_pad, j0, m, ii_interior, dx4, dx2dy2, dy4, shift=0)
    R = _interior_stencil_gather(u_pad, j0, m, ii_interior, dx4, dx2dy2, dy4, shift=1)
    out = out.at[:, 2:].set((R - L) / dx)

    def _do_onesided(_):
        return _upstream_onesided_difflaplace(u_pad, j0, i0, m, dx, dx2dy2, dy4)

    def _do_centered(_):
        ii_up = (i0 + 0) + jnp.arange(2)
        L0 = _interior_stencil_gather(u_pad, j0, m, ii_up, dx4, dx2dy2, dy4, shift=0)
        R0 = _interior_stencil_gather(u_pad, j0, m, ii_up, dx4, dx2dy2, dy4, shift=1)
        return (R0 - L0) / dx

    out = out.at[:, :2].set(lax.cond(upstream_stencil, _do_onesided, _do_centered, operand=None))
    return out


def sliced_biharmonic_operator_policy(
    u: jnp.ndarray,
    dx: float,
    dy: float,
    policy: BiharmonicPolicy,
    *,
    reflect: bool = True,
) -> jnp.ndarray:
    return _sliced_biharmonic_operator_flags(
        u,
        dx, dy,
        upstream_stencil=policy.upstream_stencil,
        pad_mode=policy.pad_mode,
        reflect=bool(reflect),
        left_pad_zeros=policy.left_pad_zeros,
    )



def sliced_biharmonic_difflaplace_policy(
    u: jnp.ndarray,
    dx: float,
    dy: float,
    policy: BiharmonicPolicy,
    *,
    reflect: bool = True,
) -> jnp.ndarray:
    return _sliced_biharmonic_difflaplace_flags(
        u,
        dx, dy,
        upstream_stencil=policy.upstream_stencil,
        pad_mode=policy.pad_mode,
        reflect=bool(reflect),
        left_pad_zeros=policy.left_pad_zeros,
    )