from __future__ import annotations

"""
Full-domain steady helpers for the infinite-depth/deep case.

This module is intended to sit beside ``run_cases.py`` in
``pywave/waves_helpers``.  It keeps the existing half-domain implementation
untouched and provides the pieces that must change when the symmetry condition
about y=0 is removed.

Main differences from the original ``run_case_deep`` path:

1. The grid is built on a full truncated y-domain, typically y in [-Y, Y], not
   only y >= 0.
2. The free-surface BIE kernels use only the physical source point distance
   Tn = y - y_*; the image contribution Tp = y + y_* is not included.
3. The desingularization and singular subtraction contain only the physical
   rectangular integral, not the mirrored-image part.
4. The flexural/biharmonic stencil no longer reflects across y=0.  Instead both
   y-boundaries are treated as artificial truncated-domain boundaries using the
   same zero/extrapolate policy already used at the downstream boundary.
5. Plotting does not mirror the result about y=0.

For production-size runs, the old analytic block preconditioner can still be
used as a practical fallback, but it is only an approximation to the full-domain
Jacobian.  ``build_blocks_infinite_depth_full_domain_autodiff`` is included as a
small-grid consistency/reference builder while the analytic block builder is
ported.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import time
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix, eye as sparse_eye, kron as sparse_kron

from jax import config as jax_config

USE_X64 = os.environ.get("VISCICE_USE_X64", "1") == "1"
jax_config.update("jax_enable_x64", USE_X64)

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

_FLOAT = np.float64 if USE_X64 else np.float32
_JFLOAT = jnp.float64 if USE_X64 else jnp.float32

from .biharmonic import (
    BiharmonicPolicy,
    UpstreamBC,
    PadMode,
    make_biharmonic_policy_from_params,
    resolve_biharmonic_policy,
    _ghost_right_cols,
    _ghost_bottom_rows,
    _interior_stencil_gather,
    _upstream_onesided,
    _upstream_onesided_difflaplace,
    precompute_powers,
)
from .preconditioners import ldu2_prec_operator
from .newton import NewtonKrylovSolver
from . import plotting as steady_case_plots
from .names import out_paths, zeta_case_tag


EPS = 1e-7
System = Literal["deep"]
BlockBuilder = Literal["analytic", "autodiff"]


# -----------------------------------------------------------------------------
# Grid / run-case utilities
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid2D:
    x: np.ndarray      # (N,)
    xm: np.ndarray     # (N-1,)
    y: np.ndarray      # (M, 1), full physical y-domain

    @property
    def x0(self) -> float:
        return float(self.x[0])

    @property
    def y0(self) -> float:
        return float(np.ravel(self.y)[0])

    @property
    def yN(self) -> float:
        return float(np.ravel(self.y)[-1])


def make_full_grid(
    N: int,
    M: int,
    dx: float,
    dy: float,
    *,
    x0: float | None = None,
    y0: float | None = None,
) -> Grid2D:
    """Build a full truncated grid.

    ``M`` is the total number of y rows.  To lift an old half-domain run with
    ``M_half`` rows and preserve the same positive-y extent, use
    ``M = 2*M_half - 1``.  With the default ``y0=None``, odd ``M`` places a row
    exactly at y=0 and gives a symmetric grid.
    """
    N = int(N)
    M = int(M)
    if x0 is None:
        x0 = -(float(dx) * N) / 2.0
    if y0 is None:
        y0 = -0.5 * float(dy) * (M - 1)

    x = float(x0) + float(dx) * np.arange(N)
    y = (float(y0) + float(dy) * np.arange(M))[:, None]
    xm = 0.5 * (x[1:] + x[:-1])
    return Grid2D(np.asarray(x, _FLOAT), np.asarray(xm, _FLOAT), np.asarray(y, _FLOAT))


def initial_guess(M: int, N: int, x0: float) -> np.ndarray:
    """Same unknown ordering as the original 2-field solver: [phi; zeta]."""
    dof = int(M) * (int(N) + 1)
    col = np.vstack(([float(x0)], np.ones((int(N), 1), dtype=_FLOAT))).astype(_FLOAT, copy=False)
    phi0 = np.tile(col, (int(M), 1)).astype(_FLOAT, copy=False)
    zeta0 = np.zeros((dof, 1), dtype=_FLOAT)
    return np.vstack((phi0, zeta0)).astype(_FLOAT, copy=False)


def pressure(x: jnp.ndarray, y: jnp.ndarray, *, eps: float, Lx: float, Ly: float) -> jnp.ndarray:
    """Compact smooth pressure evaluated on x-midpoints and full-y rows."""
    x = jnp.ravel(x)
    y = jnp.ravel(y)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    cond = (jnp.abs(X) < Lx) & (jnp.abs(Y) < Ly)
    val = jnp.exp((Lx * Lx) / (X * X - Lx * Lx) + (Ly * Ly) / (Y * Y - Ly * Ly))
    return eps * jnp.where(cond, val, 0.0)


def _fmt(dt: float) -> str:
    return f"{float(dt):.3f}s"


def _residual_vec_from_f(f_jax):
    def residual_vec(u_np: np.ndarray) -> np.ndarray:
        r = f_jax(jnp.asarray(u_np, dtype=_JFLOAT).reshape(-1))
        return np.asarray(jax.device_get(r), dtype=_FLOAT).reshape(-1)

    return residual_vec


def _plot_and_save_full_domain(grid: Grid2D, Z: np.ndarray, csv_path: Path, fig_dir: Path) -> Path:
    paths = steady_case_plots.save_case_outputs(
        X1d=np.asarray(grid.x).reshape(-1),
        Y1d=np.asarray(grid.y).reshape(-1),
        Z=Z,
        csv_path=csv_path,
        figs_dir=fig_dir,
        stem=Path(csv_path).stem,
        slice_kind="absmax",
        mirror_y0=False,
        save_csv=True,
    )
    return paths["surface"]


# -----------------------------------------------------------------------------
# Basic reconstruction / residual helpers copied from error_all.py
# -----------------------------------------------------------------------------
def midpoints(a: jnp.ndarray) -> jnp.ndarray:
    a = jnp.asarray(a)
    if a.ndim != 2:
        raise ValueError("midpoints expects a 2D array (m, n).")
    return 0.5 * (a[:, 1:] + a[:, :-1])


def allVals(u1, ux, dx, m, n):
    u1 = jnp.asarray(u1).reshape(m)
    ux = jnp.asarray(ux).reshape(m, n)

    def step(u_i, i):
        u_ip1 = u_i + 0.5 * dx * (ux[:, i] + ux[:, i + 1])
        return u_ip1, u_ip1

    _, cols = lax.scan(step, u1, jnp.arange(n - 1))
    return jnp.concatenate([u1[:, None], jnp.swapaxes(cols, 0, 1)], axis=1)


def reshaping2Unknowns(unk, m, n):
    unk = jnp.ravel(unk)
    nblk = int(m) * (int(n) + 1)
    phi = unk[:nblk].reshape(m, n + 1)
    zeta = unk[nblk:].reshape(m, n + 1)
    return phi[:, 0], phi[:, 1:], zeta[:, 0], zeta[:, 1:]


def grad_y(a: jnp.ndarray, dy: float, *, stencil: str = "centered") -> jnp.ndarray:
    """Centered y-derivative on a full interval with one-sided edge values."""
    a = jnp.asarray(a)

    def _centered(u: jnp.ndarray) -> jnp.ndarray:
        ap = jnp.pad(u, ((1, 1), (0, 0)), mode="edge")
        g = (ap[2:] - ap[:-2]) / (2.0 * dy)
        return g.at[0].set((u[1] - u[0]) / dy).at[-1].set((u[-1] - u[-2]) / dy)

    if stencil == "centered":
        return _centered(a)

    m = a.shape[0]
    if m < 5:
        return _centered(a)

    inv = 1.0 / (24.0 * dy)
    g = jnp.zeros_like(a)
    g = g.at[1].set((2.0 * a[1] - 16.0 * a[0] + 16.0 * a[2] - 2.0 * a[3]) * inv)
    g = g.at[2:-2].set((2.0 * a[:-4] - 16.0 * a[1:-3] + 16.0 * a[3:-1] - 2.0 * a[4:]) * inv)
    g = g.at[-2].set((-2.0 * a[-5] + 12.0 * a[-4] - 36.0 * a[-3] + 20.0 * a[-2] + 6.0 * a[-1]) * inv)
    return g.at[-1].set(2.0 * g[-2] - g[-3])


def grad_x(Fx, dx):
    Fx = jnp.asarray(Fx)
    return (Fx[:, 1] - Fx[:, 0]) / dx


def _apply_bc2(*, x0: float, dx: float, use_radiation: bool, zeta, zetaX, phi, phiX):
    if not use_radiation:
        return zeta[:, 0], zetaX[:, 0], phi[:, 0] - x0, phiX[:, 0] - 1.0
    nu_rad = 0.05
    zetaxx1, phixx1 = grad_x(zetaX, dx), grad_x(phiX, dx)
    F5 = x0 * zetaX[:, 0] + nu_rad * zeta[:, 0]
    F6 = x0 * zetaxx1 + nu_rad * zetaX[:, 0]
    F3 = x0 * (phiX[:, 0] - 1.0) + nu_rad * (phi[:, 0] - x0)
    F4 = x0 * phixx1 + nu_rad * (phiX[:, 0] - 1.0)
    return F5, F6, F3, F4


# -----------------------------------------------------------------------------
# Full-domain biharmonic helpers: no reflection at y=0
# -----------------------------------------------------------------------------
def _ghost_top_rows_full_domain(u_right: jnp.ndarray, pad_mode: bool) -> jnp.ndarray:
    """Two ghost rows below the first physical y-row.

    For extrapolation, row order is [second ghost, first ghost] so that the
    physical first row starts at padded index 2.
    """

    def _extrap(_):
        g1y = 2.0 * u_right[0, :] - u_right[1, :]
        g2y = 2.0 * g1y - u_right[0, :]
        return jnp.stack([g2y, g1y], axis=0)

    def _zeros(_):
        return jnp.zeros((2, u_right.shape[1]), dtype=u_right.dtype)

    return lax.cond(pad_mode, _extrap, _zeros, operand=None)


@partial(jax.jit, static_argnames=("pad_mode", "left_pad_zeros"))
def padding_scheme_full_domain(
    u: jnp.ndarray,
    *,
    pad_mode: bool,
    left_pad_zeros: bool,
) -> jnp.ndarray:
    """Pad a full y-domain field without reflecting across y=0."""
    u = jnp.asarray(u)

    right_cols = _ghost_right_cols(u, pad_mode)
    u_r = jnp.concatenate([u, right_cols], axis=1)

    top_rows = _ghost_top_rows_full_domain(u_r, pad_mode)
    bottom_rows = _ghost_bottom_rows(u_r, pad_mode)
    u_trb = jnp.concatenate([top_rows, u_r, bottom_rows], axis=0)

    if left_pad_zeros:
        left_cols = jnp.zeros((u_trb.shape[0], 2), dtype=u.dtype)
        return jnp.concatenate([left_cols, u_trb], axis=1)
    return u_trb


@partial(jax.jit, static_argnames=("upstream_stencil", "pad_mode", "left_pad_zeros"))
def _sliced_biharmonic_operator_full_domain_flags(
    u: jnp.ndarray,
    dx: float,
    dy: float,
    *,
    upstream_stencil: bool,
    pad_mode: bool,
    left_pad_zeros: bool,
) -> jnp.ndarray:
    u = jnp.asarray(u)
    m, n = u.shape
    assert (m >= 5) and (n >= 5), "Need at least a 5x5 grid for a ±2 stencil."

    u_pad = padding_scheme_full_domain(u, pad_mode=pad_mode, left_pad_zeros=left_pad_zeros)
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


@partial(jax.jit, static_argnames=("upstream_stencil", "pad_mode", "left_pad_zeros"))
def _sliced_biharmonic_difflaplace_full_domain_flags(
    u: jnp.ndarray,
    dx: float,
    dy: float,
    *,
    upstream_stencil: bool,
    pad_mode: bool,
    left_pad_zeros: bool,
) -> jnp.ndarray:
    u = jnp.asarray(u)
    m, n = u.shape
    assert (m >= 5) and (n >= 5), "Need at least a 5x5 grid for a ±2 stencil."

    u_pad = padding_scheme_full_domain(u, pad_mode=pad_mode, left_pad_zeros=left_pad_zeros)
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


def sliced_biharmonic_operator_full_domain(
    u: jnp.ndarray,
    dx: float,
    dy: float,
    policy: BiharmonicPolicy,
) -> jnp.ndarray:
    return _sliced_biharmonic_operator_full_domain_flags(
        u,
        dx,
        dy,
        upstream_stencil=policy.upstream_stencil,
        pad_mode=policy.pad_mode,
        left_pad_zeros=policy.left_pad_zeros,
    )


def sliced_biharmonic_difflaplace_full_domain(
    u: jnp.ndarray,
    dx: float,
    dy: float,
    policy: BiharmonicPolicy,
) -> jnp.ndarray:
    return _sliced_biharmonic_difflaplace_full_domain_flags(
        u,
        dx,
        dy,
        upstream_stencil=policy.upstream_stencil,
        pad_mode=policy.pad_mode,
        left_pad_zeros=policy.left_pad_zeros,
    )


@partial(jax.jit, static_argnames=("N", "M"))
def _reconstruct_zeta_from_slice(u_slice: jnp.ndarray, M: int, N: int, dx: float) -> jnp.ndarray:
    U = u_slice.reshape((M, N + 1))
    zeta1 = U[:, 0]
    gx = U[:, 1:]
    inc = 0.5 * dx * (gx[:, 1:] + gx[:, :-1])
    csum = jnp.cumsum(inc, axis=1)
    csum = jnp.concatenate([jnp.zeros((M, 1), gx.dtype), csum], axis=1)
    return zeta1[:, None] + csum


@partial(jax.jit, static_argnames=("N", "M", "policy", "tauf"))
def ice_sheet_full_domain(
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
    """Flexural residual using full-domain y-padding rather than y=0 reflection."""
    zeta = _reconstruct_zeta_from_slice(u_slice, M, N, dx)
    dlaplace_over2 = sliced_biharmonic_operator_full_domain(zeta, dx, dy, policy)
    if tauf == 0.0:
        PFlexHalf = aleph * dlaplace_over2
    else:
        difflaplace = sliced_biharmonic_difflaplace_full_domain(zeta, dx, dy, policy)
        PFlexHalf = aleph * (dlaplace_over2 + tauf * difflaplace)

    zeros_col = jnp.zeros((M, 1), dtype=PFlexHalf.dtype)
    E1 = jnp.concatenate([zeros_col, zeros_col, PFlexHalf], axis=1)
    return E1.reshape(M * (N + 1),)


def jflex_dense_jacobian_full_domain(
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
    """Dense flexural Jacobian for the full-domain biharmonic residual."""
    neq = int(M) * (int(N) + 1)
    u_slice = u[:neq]
    f = lambda us: ice_sheet_full_domain(us, N, M, dx, dy, aleph, policy, tauf=tauf)

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
            block = jvp_cols(u_slice, idx_safe)
            block_T = block.T
            cols = int(min(chunk_cols, neq - start))
            J_host[:, start : start + cols] = np.asarray(jax.device_get(block_T[:, :cols]))
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


def _default_device_name(preferred: str = "gpu") -> str:
    try:
        _ = jax.devices(preferred)[0]
        return preferred
    except Exception:
        return "cpu"


def flexural_contribution_full_domain(
    N: int,
    M: int,
    dx: float,
    dy: float,
    *,
    aleph: float,
    tauf: float,
    policy: BiharmonicPolicy | None = None,
    depth: Literal["infinite", "finite"] | None = None,
    F: float | None = None,
    upstream_bc: UpstreamBC | None = None,
    mode: PadMode | None = None,
    announce_policy: bool = False,
    chunk_cols: int = 16,
    device: str = "gpu",
    dtype: np.dtype = _FLOAT,
    jax_dtype=_JFLOAT,
) -> np.ndarray:
    """Dense flexural Jacobian with no y=0 mirror symmetry."""
    policy = resolve_biharmonic_policy(
        policy=policy,
        upstream_bc=upstream_bc,
        mode=mode,
        depth=depth,
        F=F,
        aleph=aleph,
        announce=announce_policy,
    )
    neq = int(M) * (int(N) + 1)
    dev = jax.devices(_default_device_name(device))[0]
    with jax.default_device(dev):
        u0 = jnp.zeros((neq,), dtype=jax_dtype)
        J_host = jflex_dense_jacobian_full_domain(
            u0,
            int(N),
            int(M),
            float(dx),
            float(dy),
            aleph=jnp.asarray(aleph, dtype=jax_dtype),
            policy=policy,
            tauf=float(tauf),
            chunk_cols=int(chunk_cols),
            return_numpy=True,
        )
    return np.asarray(J_host, dtype=dtype)


def dynamic_cond_full_domain(
    dx: float,
    dy: float,
    Fr: float,
    aleph: float,
    mu: float,
    xm: jnp.ndarray,
    *,
    zeta: jnp.ndarray,
    zetam: jnp.ndarray,
    zetaxm: jnp.ndarray,
    zetaym: jnp.ndarray,
    phim: jnp.ndarray,
    phixm: jnp.ndarray,
    phiym: jnp.ndarray,
    pm: jnp.ndarray | None = None,
    tauf: float,
    policy: BiharmonicPolicy,
) -> jnp.ndarray:
    """Dynamic condition using the full-domain flexural stencil."""
    pm = 0.0 if pm is None else pm
    dlaplace = sliced_biharmonic_operator_full_domain(zeta, dx, dy, policy)
    difflaplace = lax.cond(
        jnp.not_equal(tauf, 0.0),
        lambda _: sliced_biharmonic_difflaplace_full_domain(zeta, dx, dy, policy),
        lambda _: jnp.zeros_like(dlaplace),
        operand=None,
    )
    flex_term = aleph * (dlaplace + tauf * difflaplace)
    den = 1.0 + zetaxm**2 + zetaym**2
    kin_num = (
        (1.0 + zetaym**2) * (phixm**2)
        + (1.0 + zetaxm**2) * (phiym**2)
        - 2.0 * zetaxm * zetaym * phixm * phiym
    )
    kin = 0.5 * kin_num / den
    return kin - 0.5 + zetam / (Fr**2) + pm + flex_term + mu * (phim - xm[None, :])


# -----------------------------------------------------------------------------
# Full-domain BIE / ERROR kernel helpers: no y-image terms
# -----------------------------------------------------------------------------
def trapz2d(F: jnp.ndarray, dx_x: jnp.ndarray, dy_y: jnp.ndarray) -> jnp.ndarray:
    F = jnp.asarray(F)
    Ix = jnp.sum(0.5 * (F[:, :-1] + F[:, 1:]) * dx_x[None, :], axis=1)
    return jnp.sum(0.5 * (Ix[:-1] + Ix[1:]) * dy_y, axis=0)


def safe_sqrt(x):
    return jnp.sqrt(jnp.clip(x, EPS, jnp.inf))


def _safe_log(x):
    return jnp.log(jnp.clip(x, EPS, jnp.inf))


def safe_log(x):
    return _safe_log(x)


def kernels_full_domain(S, Tn, zeta, zetax, zetay, zeta_star):
    """Free-surface kernels over the physical full y-domain only."""
    r2n = S * S + Tn * Tn + (zeta - zeta_star) ** 2
    r3n = r2n * safe_sqrt(r2n)
    return (
        ((zeta - zeta_star) - S * zetax - Tn * zetay) / r3n,
        1.0 / safe_sqrt(r2n),
    )


# Compatibility names matching error_all.py signatures; Tp is ignored on purpose.
def kernels(S, Tn, Tp, zeta, zetax, zetay, zeta_star):
    del Tp
    return kernels_full_domain(S, Tn, zeta, zetax, zetay, zeta_star)


def desingularize_jax(sN, tN, s1, t1, tNp, t1p, A, B, C):
    """Rectangular singular integral without the mirrored-image contribution."""
    del tNp, t1p

    def I2pp1(s, t, Bb):
        return (t / jnp.sqrt(A)) * _safe_log(
            2.0 * A * s + Bb * t + 2.0 * safe_sqrt(A * (A * s * s + Bb * s * t + C * t * t))
        )

    def I2pp2(s, t, Bb):
        return (s / jnp.sqrt(C)) * _safe_log(
            2.0 * C * t + Bb * s + 2.0 * safe_sqrt(C * (A * s * s + Bb * s * t + C * t * t))
        )

    out = I2pp2(sN, tN, B) - I2pp2(sN, t1, B) - I2pp2(s1, tN, B) + I2pp2(s1, t1, B)
    out += (jnp.abs(t1) > EPS) * (-I2pp1(sN, t1, B) + I2pp1(s1, t1, B))
    out += (jnp.abs(tN) > EPS) * (-I2pp1(s1, tN, B) + I2pp1(sN, tN, B))
    return out


def S2_term(A, B, C, S, Tn, Tp):
    """Singular subtraction term without the y-image term."""
    del Tp
    return 1.0 / safe_sqrt(A * S * S + B * S * Tn + C * Tn * Tn)


def _desing_eta_x_k2(*, S, Tn, Tp, x0, xN, y0, yN, x_star, y_star, eta_x, eta_x_star, eta_y_star, K2, dx_x, dy_y):
    A = 1.0 + eta_x_star**2
    B = 2.0 * eta_x_star * eta_y_star
    C = 1.0 + eta_y_star**2
    I_reg = trapz2d(eta_x * K2 - eta_x_star * S2_term(A, B, C, S, Tn, Tp), dx_x, dy_y)
    s1, sN = x0 - x_star, xN - x_star
    t1, tN = y0 - y_star, yN - y_star
    return I_reg + eta_x_star * desingularize_jax(sN, tN, s1, t1, 0.0, 0.0, A, B, C)


_BIE_VMAP_OVER_K = True
_BIE_STREAM_OVER_L = True
_BIE_GUARD_MAX_EVALS = 20_000


def _map_bie(fn, My: int, Neval: int, *, vmap_over_k: bool | None = None, stream_over_l: bool | None = None):
    n = int(My) * int(Neval)
    vmap_k = (_BIE_VMAP_OVER_K and (n < _BIE_GUARD_MAX_EVALS)) if vmap_over_k is None else bool(vmap_over_k)
    stream_l = _BIE_STREAM_OVER_L if stream_over_l is None else bool(stream_over_l)
    l_idx, k_idx = jnp.arange(My), jnp.arange(Neval)

    def row(l):
        return jax.vmap(lambda k: fn(l, k))(k_idx) if vmap_k else lax.map(lambda k: fn(l, k), k_idx)

    return lax.map(row, l_idx) if stream_l else jax.vmap(row)(l_idx)


def _deep_bie_lk(
    l: int,
    k: int,
    x: jnp.ndarray,
    xm: jnp.ndarray,
    y1d: jnp.ndarray,
    *,
    dx_x,
    dy_y,
    x0,
    xN,
    y0,
    yN,
    phi,
    phim,
    zeta,
    zetax,
    zetay,
    zetam,
    zetaxm,
    zetaym,
):
    zeta_star = zetam[l, k]
    phi_star = phim[l, k]
    x_star = xm[k]
    y_star = y1d[l]
    zetax_star = zetaxm[l, k]
    zetay_star = zetaym[l, k]
    S = (x - x_star)[None, :]
    Tn = (y1d - y_star)[:, None]
    Tp = Tn  # retained only to keep the desingularization call signature stable
    K1, K2 = kernels(S, Tn, Tp, zeta, zetax, zetay, zeta_star)
    I1 = trapz2d(((phi - phi_star) - S) * K1, dx_x, dy_y)
    I2 = _desing_eta_x_k2(
        S=S,
        Tn=Tn,
        Tp=Tp,
        x0=x0,
        xN=xN,
        y0=y0,
        yN=yN,
        x_star=x_star,
        y_star=y_star,
        eta_x=zetax,
        eta_x_star=zetax_star,
        eta_y_star=zetay_star,
        K2=K2,
        dx_x=dx_x,
        dy_y=dy_y,
    )
    return -2.0 * jnp.pi * (phi_star - x_star) + I1 + I2


@partial(jax.jit, static_argnames=("M", "N", "use_radiation", "policy"))
def jax_residual_deep_full_domain(
    u,
    M,
    N,
    dx,
    dy,
    Fr,
    aleph,
    tauf,
    mu,
    x,
    xm,
    y,
    pm,
    *,
    use_radiation: bool,
    policy: BiharmonicPolicy,
):
    """Deep-water steady residual on the full truncated y-domain."""
    phi1, phix, zeta1, zetax = reshaping2Unknowns(u, M, N)
    phi = allVals(phi1, phix, dx, M, N)
    zeta = allVals(zeta1, zetax, dx, M, N)
    phiy = grad_y(phi, dy, stencil="centered")
    zetay = grad_y(zeta, dy, stencil="centered")

    y1d, x1d = jnp.ravel(y), jnp.ravel(x)
    dx_x, dy_y = x1d[1:] - x1d[:-1], y1d[1:] - y1d[:-1]
    x0, xN = x1d[0], x1d[-1]
    y0, yN = y1d[0], y1d[-1]

    zetam, zetaxm, zetaym = midpoints(zeta), midpoints(zetax), midpoints(zetay)
    phim, phixm, phiym = midpoints(phi), midpoints(phix), midpoints(phiy)
    dyn = dynamic_cond_full_domain(
        dx=dx,
        dy=dy,
        Fr=Fr,
        aleph=aleph,
        mu=mu,
        xm=xm,
        zeta=zeta,
        zetam=zetam,
        zetaxm=zetaxm,
        zetaym=zetaym,
        phim=phim,
        phixm=phixm,
        phiym=phiym,
        pm=pm,
        tauf=tauf,
        policy=policy,
    )
    My, Neval = int(y1d.shape[0]), int(xm.shape[0])
    bie = _map_bie(
        lambda l, k: _deep_bie_lk(
            l,
            k,
            x1d,
            xm,
            y1d,
            dx_x=dx_x,
            dy_y=dy_y,
            x0=x0,
            xN=xN,
            y0=y0,
            yN=yN,
            phi=phi,
            phim=phim,
            zeta=zeta,
            zetax=zetax,
            zetay=zetay,
            zetam=zetam,
            zetaxm=zetaxm,
            zetaym=zetaym,
        ),
        My,
        Neval,
    ).reshape(My, Neval)

    bc_zeta, bc_zetax, bc_phi, bc_phix = _apply_bc2(
        x0=x0,
        dx=dx,
        use_radiation=use_radiation,
        zeta=zeta,
        zetaX=zetax,
        phi=phi,
        phiX=phix,
    )
    E1 = jnp.hstack((bc_phi[:, None], bc_phix[:, None], dyn)).reshape(M * (N + 1), 1)
    E2 = jnp.hstack((bc_zeta[:, None], bc_zetax[:, None], bie)).reshape(M * (N + 1), 1)
    return jnp.vstack((E1, E2))[:, 0]


# -----------------------------------------------------------------------------
# Preconditioner block helpers
# -----------------------------------------------------------------------------

# ---- analytic full-domain infinite-depth block builder ------------------------
def analytic_I0_integral_full_domain(x: np.ndarray, y: np.ndarray, xm: np.ndarray) -> np.ndarray:
    """Analytic rectangular integral of R0^{-1} over the physical full y-domain.

    This is the full-domain counterpart of ``analytic_I0_integral`` in
    ``build_all_blocks.py``.  The symmetric half-domain implementation appends an
    extra image contribution involving y + y_*.  That term is deliberately absent
    here because the integration grid already contains both positive and negative
    physical y values.
    """
    x_1d = np.asarray(x, dtype=_FLOAT).reshape(-1)
    x_mid_1d = np.asarray(xm, dtype=_FLOAT).reshape(-1)
    y_col = np.asarray(y, dtype=_FLOAT).reshape(-1, 1)

    s1 = x_1d[0] - x_mid_1d
    sN = x_1d[-1] - x_mid_1d
    t1 = y_col[0:1] - y_col
    tM = y_col[-1:] - y_col
    log2 = np.log(np.asarray(2.0, dtype=_FLOAT)).astype(_FLOAT)

    def F1(s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return t * (np.log(s + np.sqrt(s * s + t * t)) + log2)

    def F2(s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return s * (np.log(t + np.sqrt(s * s + t * t)) + log2)

    def evl(F, t: np.ndarray) -> np.ndarray:
        return F(sN, t) - F(s1, t)

    I = evl(F2, tM) - evl(F2, t1)

    m1 = (t1[:, 0] != 0.0)
    if m1.any():
        I[m1, :] -= evl(F1, t1[m1, :])

    mM = (tM[:, 0] != 0.0)
    if mM.any():
        I[mM, :] += evl(F1, tM[mM, :])

    return np.asarray(I, dtype=_FLOAT)


def _bc_identity(N: int) -> np.ndarray:
    E = np.zeros((2, int(N) + 1), dtype=_FLOAT)
    E[0, 0] = 1.0
    E[1, 1] = 1.0
    return E


def _bc_zero(N: int) -> np.ndarray:
    return np.zeros((2, int(N) + 1), dtype=_FLOAT)


def _bc_radiation(N: int, dx: float, x0: float, n: float = 0.05) -> np.ndarray:
    N = int(N)
    dx = float(dx)
    x0 = float(x0)
    E = np.zeros((2, N + 1), dtype=_FLOAT)
    inv = 1.0 / dx
    E[0, 0] = float(n)
    E[0, 1] = x0
    E[1, 1] = float(n) - x0 * inv
    if N >= 2:
        E[1, 2] = x0 * inv
    return E


def _trapz_dx_op(N: int, dx: float) -> np.ndarray:
    N = int(N)
    dx = float(dx)
    T = np.tri(N - 1, dtype=_FLOAT)
    T = T + 2.0 * np.tril(T, -1) + np.tril(T, -2)
    v = np.ones((N - 1, 1), dtype=_FLOAT)
    v[1:, 0] = 2.0
    return np.hstack((np.ones((N - 1, 1), dtype=_FLOAT), (dx / 4.0) * np.hstack((v, T))))


def _halfmesh_avg_op(N: int) -> np.ndarray:
    N = int(N)
    d = (np.eye(N - 1, N, 0, dtype=_FLOAT) + np.eye(N - 1, N, 1, dtype=_FLOAT)) / 2.0
    return np.hstack((np.zeros((N - 1, 1), dtype=_FLOAT), d))


def _repeat_blockdiag(block: np.ndarray, M: int, sparse: bool):
    block = np.asarray(block, dtype=_FLOAT)
    M = int(M)
    if sparse:
        return sparse_kron(sparse_eye(M, format="csr", dtype=_FLOAT), csr_matrix(block), format="csr")
    return np.kron(np.eye(M, dtype=_FLOAT), block)


def _xstar_blockdiag(N: int, M: int, sparse: bool, dx: float | None = None, x0: float | None = None, use_radiation: bool = False):
    if sparse:
        E = _bc_radiation(N, float(dx), float(x0)) if use_radiation else _bc_identity(N)
        return _repeat_blockdiag(np.vstack((E, _halfmesh_avg_op(N))), M, True)
    return _repeat_blockdiag(np.vstack((_bc_identity(N), _halfmesh_avg_op(N))), M, False)


def _star_blockdiag(N: int, M: int, dx: float, weights, sparse: bool):
    base = _trapz_dx_op(N, dx)
    if np.isscalar(weights):
        interior = base * float(weights)
    else:
        interior = base * np.asarray(weights, dtype=_FLOAT).reshape(-1)[:, None]
    return _repeat_blockdiag(np.vstack((_bc_zero(N), interior)), M, sparse)


def _dense_grids_full_domain(M: int, N: int, x, y, xm):
    M = int(M)
    N = int(N)
    x = np.asarray(x, dtype=_FLOAT).reshape(-1)
    y = np.asarray(y, dtype=_FLOAT).reshape(-1)
    xm = np.asarray(xm, dtype=_FLOAT).reshape(-1)
    x_diff2 = (x[None, :] - xm[:, None]) ** 2
    wy = np.ones((M,), dtype=_FLOAT)
    wx = np.ones((N,), dtype=_FLOAT)
    if M >= 2:
        wy[0] *= 0.5
        wy[-1] *= 0.5
    if N >= 2:
        wx[0] *= 0.5
        wx[-1] *= 0.5
    return x, y, xm, x_diff2, wy[:, None] * wx[None, :]


def _dense_kernel_Rinv_full_domain(x_diff2: np.ndarray, w2: np.ndarray, y: np.ndarray, j: int, a2: float, scale: float) -> np.ndarray:
    dy2 = (y - y[j]) ** 2
    inv_r = 1.0 / np.sqrt(x_diff2[None, :, :] + dy2[:, None, None] + float(a2))
    return inv_r * (w2[:, None, :] * float(scale))


def _dense_bc_rows_all(M: int, N: int, width: int, dx: float, x0: float, use_radiation: bool) -> np.ndarray:
    E = _bc_radiation(N, dx, x0) if use_radiation else _bc_identity(N)
    nz0 = np.flatnonzero(E[0])
    nz1 = np.flatnonzero(E[1])
    bc = np.zeros((int(M), 2, int(width)), dtype=_FLOAT)
    for j in range(int(M)):
        base = j * (int(N) + 1)
        bc[j, 0, base + nz0] = E[0, nz0]
        bc[j, 1, base + nz1] = E[1, nz1]
    return bc


def _apply_I0_correction(K0: np.ndarray, I0_row: np.ndarray, j: int, N: int) -> None:
    k = np.arange(int(N) - 1)
    corr = -(K0.sum(axis=(0, 2)) + I0_row) / 2.0
    K0[j, k, k] += corr
    K0[j, k, k + 1] += corr


def _infinite_depth_D_full_domain(M: int, N: int, dx: float, dy: float, *, x0: float, use_radiation: bool, x, y, xm) -> np.ndarray:
    """Analytic D block for infinite depth on the full physical y-domain."""
    M = int(M)
    N = int(N)
    dx = float(dx)
    dy = float(dy)
    width = (N + 1) * M
    x1, y1, xm1, x_diff2, w2 = _dense_grids_full_domain(M, N, x, y, xm)
    I0 = analytic_I0_integral_full_domain(x1, y1, xm1)
    out = np.zeros((M * (N + 1), width), dtype=_FLOAT)
    bc_all = _dense_bc_rows_all(M, N, width, dx, float(x0), bool(use_radiation))
    scale0 = -dx * dy
    for j in range(M):
        base = j * (N + 1)
        out[base : base + 2, :] = bc_all[j]
        K0 = _dense_kernel_Rinv_full_domain(x_diff2, w2, y1, j, 0.0, scale0)
        _apply_I0_correction(K0, I0[j, :], j, N)
        B = np.concatenate([np.zeros((M, N - 1, 1), dtype=_FLOAT), K0], axis=2)
        out[base + 2 : base + 2 + (N - 1), :] = B.transpose(1, 0, 2).reshape(N - 1, width)
    return out


def build_blocks_infinite_depth_full_domain_analytic(
    *,
    N: int,
    M: int,
    dx: float,
    dy: float,
    grid: Grid2D,
    Fr: float,
    mu: float,
    use_radiation: bool,
) -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]:
    """Analytic block preconditioner for the full-domain infinite-depth case."""
    N = int(N)
    M = int(M)
    dx = float(dx)
    dy = float(dy)
    A = _xstar_blockdiag(N, M, True, dx=dx, x0=float(grid.x0), use_radiation=bool(use_radiation))
    A = A + _star_blockdiag(N, M, dx, float(mu), True)
    B = _star_blockdiag(N, M, dx, 1.0 / (float(Fr) ** 2), True).toarray()
    C = _star_blockdiag(N, M, dx, 2.0 * np.pi, True)
    D = _infinite_depth_D_full_domain(
        M,
        N,
        dx,
        dy,
        x0=float(grid.x0),
        use_radiation=bool(use_radiation),
        x=grid.x,
        y=grid.y,
        xm=grid.xm,
    )
    return csr_matrix(A), B, csr_matrix(C), D

def build_blocks_infinite_depth_full_domain_autodiff(
    *,
    N: int,
    M: int,
    dx: float,
    dy: float,
    grid: Grid2D,
    Fr: float,
    mu: float,
    use_radiation: bool,
    policy: BiharmonicPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference full-domain block builder by differentiating the residual.

    This is meant for small-grid verification and for deriving/testing the
    analytic block formulas.  It omits flexural terms because the caller adds
    ``flexural_contribution_full_domain`` to the B block, matching ``run_cases``.
    """
    nblk = int(M) * (int(N) + 1)
    u0 = jnp.asarray(initial_guess(int(M), int(N), grid.x0), dtype=_JFLOAT).reshape(-1)
    x = jnp.asarray(grid.x, dtype=_JFLOAT)
    xm = jnp.asarray(grid.xm, dtype=_JFLOAT)
    y = jnp.asarray(np.ravel(grid.y), dtype=_JFLOAT)
    pm0 = jnp.zeros((int(M), int(N) - 1), dtype=_JFLOAT)

    def f_base(v):
        return jax_residual_deep_full_domain(
            v,
            int(M),
            int(N),
            float(dx),
            float(dy),
            float(Fr),
            0.0,      # no flexural term in base blocks
            0.0,
            float(mu),
            x,
            xm,
            y,
            pm0,
            use_radiation=bool(use_radiation),
            policy=policy,
        )

    J = np.asarray(jax.device_get(jax.jacfwd(f_base)(u0)), dtype=_FLOAT)
    A = J[:nblk, :nblk]
    B = J[:nblk, nblk:]
    C = J[nblk:, :nblk]
    D = J[nblk:, nblk:]
    return A, B, C, D


def build_blocks_infinite_depth_full_domain(
    *,
    N: int,
    M: int,
    dx: float,
    dy: float,
    grid: Grid2D,
    Fr: float,
    mu: float,
    use_radiation: bool,
    policy: BiharmonicPolicy,
    method: BlockBuilder = "analytic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return preconditioner blocks for the full-domain deep case.

    ``method='analytic'`` is the production path. It ports the infinite-depth
    analytic block formulas while removing the y-image terms used by the
    symmetric half-domain implementation. ``method='autodiff'`` is slower but
    useful for small-grid consistency checks.
    """
    if method == "analytic":
        return build_blocks_infinite_depth_full_domain_analytic(
            N=N,
            M=M,
            dx=dx,
            dy=dy,
            grid=grid,
            Fr=Fr,
            mu=mu,
            use_radiation=use_radiation,
        )
    if method == "autodiff":
        return build_blocks_infinite_depth_full_domain_autodiff(
            N=N,
            M=M,
            dx=dx,
            dy=dy,
            grid=grid,
            Fr=Fr,
            mu=mu,
            use_radiation=use_radiation,
            policy=policy,
        )
    raise ValueError("method must be 'analytic' or 'autodiff'")


def _run_case_2_full_domain(
    *,
    tag: str,
    grid: Grid2D,
    fig_dir: Path,
    csv_path: Path,
    f_jax,
    u0: np.ndarray,
    build_blocks,
    N: int,
    M: int,
    dx: float,
    dy: float,
    aleph: float,
    tauf: float,
    policy: BiharmonicPolicy,
    verbose: bool = True,
) -> tuple[np.ndarray, Path, Path]:
    residual_vec = _residual_vec_from_f(f_jax)
    _ = jax.block_until_ready(f_jax(jnp.asarray(u0, dtype=_JFLOAT).reshape(-1)))
    print("[check] ||F(u0)||2 =", float(np.linalg.norm(residual_vec(u0))))

    t0 = time.time()
    A, B_base, C, D = build_blocks()
    B_flex = flexural_contribution_full_domain(
        int(N),
        int(M),
        float(dx),
        float(dy),
        aleph=float(aleph),
        tauf=float(tauf),
        policy=policy,
    )
    B = np.asarray(B_base) + np.asarray(B_flex)
    M_prec = ldu2_prec_operator(A, B, C, D)
    print("[timer] preconditioner:", _fmt(time.time() - t0))

    t1 = time.time()
    u_star = NewtonKrylovSolver(method="lgmres", verbose=bool(verbose)).solve(
        residual_vec, np.asarray(u0, dtype=_FLOAT), preconditioner=M_prec
    )
    print("[timer] solve:", _fmt(time.time() - t1))
    print("[done]  ||F(u*)||2 =", float(np.linalg.norm(residual_vec(u_star))))

    _, _, zeta1, zeta_x = reshaping2Unknowns(u_star, int(M), int(N))
    Z = np.asarray(jax.device_get(allVals(zeta1, zeta_x, float(dx), int(M), int(N))))
    fig_path = _plot_and_save_full_domain(grid, Z, csv_path, fig_dir)
    print("[save]", csv_path)
    print("[save]", fig_path)
    return Z, csv_path, fig_path


# -----------------------------------------------------------------------------
# Full-domain version of run_case_deep
# -----------------------------------------------------------------------------
def run_case_deep_full_domain(
    *,
    out_base: str | Path | None = None,
    upstream_bc: UpstreamBC | None = None,
    mode: PadMode | None = None,
    announce_policy: bool = True,
    N: int = 160,
    M: int = 161,
    dx: float = 0.3,
    dy: float = 0.6,
    x0: float | None = None,
    y0: float | None = None,
    Fr: float = 2.0,
    aleph: float = 0.5,
    mu: float = 0.0,
    epsilon: float = 0.1,
    Lx: float = 1.0,
    Ly: float = 1.0,
    tauf: float = 0.0,
    use_radiation: bool = False,
    block_builder: BlockBuilder = "analytic",
    verbose: bool = True,
) -> tuple[np.ndarray, Path, Path]:
    """Run the deep steady case on a full y-domain.

    ``M`` is the total number of y rows.  The default ``M=161`` corresponds to
    the full-domain lift of the old ``M=81`` half-domain grid.
    """
    system: System = "deep"
    grid = make_full_grid(N, M, dx, dy, x0=x0, y0=y0)
    policy = make_biharmonic_policy_from_params(
        depth="infinite",
        F=float(Fr),
        aleph=float(aleph),
        upstream_bc=upstream_bc,
        mode=mode,
        announce=bool(announce_policy),
    )
    upstream_bc = policy.upstream_bc
    mode = policy.mode

    results_dir, fig_dir = out_paths(system=system, out_base=out_base)
    tag = zeta_case_tag(
        system=system,
        M=M,
        N=N,
        dx=dx,
        dy=dy,
        x0=grid.x0,
        Fr=Fr,
        aleph=aleph,
        epsilon=epsilon,
        mu=mu,
        tauf=tauf,
        use_radiation=use_radiation,
        upstream_bc=upstream_bc,
        mode=mode,
    )
    tag = f"full_y__{tag}"
    csv_path = results_dir / f"zeta_{tag}.csv"

    print("[backend]", jax.default_backend(), "| devices:", jax.devices())
    print("[case]", tag)
    print("[grid] y in [%.6g, %.6g] with M=%d" % (grid.y0, grid.yN, int(M)))
    print("[out_base]", results_dir.parents[2])
    print("[blocks]", block_builder)

    xm = jnp.asarray(grid.xm, dtype=_JFLOAT)
    y1d = jnp.asarray(np.ravel(np.asarray(grid.y)), dtype=_JFLOAT)
    pm = pressure(xm, y1d, eps=float(epsilon), Lx=float(Lx), Ly=float(Ly))
    params = dict(
        M=int(M),
        N=int(N),
        dx=float(dx),
        dy=float(dy),
        Fr=float(Fr),
        aleph=float(aleph),
        tauf=float(tauf),
        mu=float(mu),
        x=jnp.asarray(grid.x, dtype=_JFLOAT),
        xm=xm,
        y=y1d,
        pm=jnp.asarray(pm, dtype=_JFLOAT),
        use_radiation=bool(use_radiation),
        policy=policy,
    )
    f_jax = lambda u_flat: jax_residual_deep_full_domain(u_flat, **params)
    u0 = initial_guess(M=int(M), N=int(N), x0=grid.x0)

    def build_blocks():
        return build_blocks_infinite_depth_full_domain(
            N=int(N),
            M=int(M),
            dx=float(dx),
            dy=float(dy),
            grid=grid,
            Fr=float(Fr),
            mu=float(mu),
            use_radiation=bool(use_radiation),
            policy=policy,
            method=block_builder,
        )

    return _run_case_2_full_domain(
        tag=tag,
        grid=grid,
        fig_dir=fig_dir,
        csv_path=csv_path,
        f_jax=f_jax,
        u0=u0,
        build_blocks=build_blocks,
        N=N,
        M=M,
        dx=dx,
        dy=dy,
        aleph=aleph,
        tauf=tauf,
        policy=policy,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Small smoke-test-sized example.  Use larger production values from your
    # runner once this import path is working locally.
    run_case_deep_full_domain(N=40, M=41, dx=0.3, dy=0.6, block_builder="analytic")