from __future__ import annotations

from functools import partial
import os
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import lax
# my-packages/pypure/pywave/waves_helpers/error_all.py

from .biharmonic import BiharmonicPolicy, sliced_biharmonic_operator_policy, sliced_biharmonic_difflaplace_policy

# EPS = 1e-12
EPS = 1e-7

__all__ = [
    "EPS",
    "allVals",
    "reshaping2Unknowns",
    "reshaping3Unknowns",
    "safe_sqrt",
    "safe_log",
    "S2_term",
    "desingularize_jax",
    "jax_residual_flatbed",
    "jax_residual_deep",
    "bathymetry_bie_residual",
    "jax_residual_bathy",
]


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
    nblk = m * (n + 1)
    phi = unk[:nblk].reshape(m, n + 1)
    zeta = unk[nblk:].reshape(m, n + 1)
    return phi[:, 0], phi[:, 1:], zeta[:, 0], zeta[:, 1:]


def reshaping3Unknowns(u, M, N):
    u = jnp.ravel(u)
    nblk = int(M) * (int(N) + 1)
    zet = u[0 * nblk : 1 * nblk].reshape(M, N + 1)
    phi = u[1 * nblk : 2 * nblk].reshape(M, N + 1)
    psi = u[2 * nblk : 3 * nblk].reshape(M, N + 1)
    return zet[:, 0], zet[:, 1:], phi[:, 0], phi[:, 1:], psi[:, 0], psi[:, 1:]


def grad_y(a: jnp.ndarray, dy: float, *, stencil: str = "centered") -> jnp.ndarray:
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


def _apply_bc3(*, x0: float, dx: float, use_radiation: bool, zeta, zetax, phi, phix, psi, psix):
    if not use_radiation:
        return zeta[:, 0], zetax[:, 0], phi[:, 0] - x0, phix[:, 0] - 1.0, psi[:, 0] - x0, psix[:, 0] - 1.0
    nu_rad = 0.05
    zetaxx1, phixx1, psixx1 = grad_x(zetax, dx), grad_x(phix, dx), grad_x(psix, dx)
    return (
        x0 * zetax[:, 0] + nu_rad * zeta[:, 0],
        x0 * zetaxx1 + nu_rad * zetax[:, 0],
        x0 * (phix[:, 0] - 1.0) + nu_rad * (phi[:, 0] - x0),
        x0 * phixx1 + nu_rad * (phix[:, 0] - 1.0),
        x0 * (psix[:, 0] - 1.0) + nu_rad * (psi[:, 0] - x0),
        x0 * psixx1 + nu_rad * (psix[:, 0] - 1.0),
    )


def dynamic_cond(
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
    pm: Optional[jnp.ndarray] = None,
    tauf: float,
    policy: BiharmonicPolicy,
) -> jnp.ndarray:
    pm = 0.0 if pm is None else pm
    dlaplace = sliced_biharmonic_operator_policy(zeta, dx, dy, policy)
    difflaplace = lax.cond(
        jnp.not_equal(tauf, 0.0),
        lambda _: sliced_biharmonic_difflaplace_policy(zeta, dx, dy, policy),
        lambda _: jnp.zeros_like(dlaplace),
        operand=None,
    )
    flex_term = aleph * (dlaplace + tauf * difflaplace)
    den = 1.0 + zetaxm**2 + zetaym**2
    kin_num = (1.0 + zetaym**2) * (phixm**2) + (1.0 + zetaxm**2) * (phiym**2) - 2.0 * zetaxm * zetaym * phixm * phiym
    kin = 0.5 * kin_num / den
    return kin - 0.5 + zetam / (Fr**2) + pm + flex_term + mu * (phim - xm[None, :])


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


def kernels(S, Tn, Tp, zeta, zetax, zetay, zeta_star):
    r2n = S * S + Tn * Tn + (zeta - zeta_star) ** 2
    r2p = S * S + Tp * Tp + (zeta - zeta_star) ** 2
    r3n, r3p = r2n * safe_sqrt(r2n), r2p * safe_sqrt(r2p)
    return (
        ((zeta - zeta_star) - S * zetax - Tn * zetay) / r3n + ((zeta - zeta_star) - S * zetax - Tp * zetay) / r3p,
        1.0 / safe_sqrt(r2n) + 1.0 / safe_sqrt(r2p),
    )


def desingularize_jax(sN, tN, s1, t1, tNp, t1p, A, B, C):
    def I2pp1(s, t, Bb):
        return (t / jnp.sqrt(A)) * _safe_log(2.0 * A * s + Bb * t + 2.0 * safe_sqrt(A * (A * s * s + Bb * s * t + C * t * t)))

    def I2pp2(s, t, Bb):
        return (s / jnp.sqrt(C)) * _safe_log(2.0 * C * t + Bb * s + 2.0 * safe_sqrt(C * (A * s * s + Bb * s * t + C * t * t)))

    out = I2pp2(sN, tN, B) - I2pp2(sN, t1, B) - I2pp2(s1, tN, B) + I2pp2(s1, t1, B)
    out += (jnp.abs(t1) > EPS) * (-I2pp1(sN, t1, B) + I2pp1(s1, t1, B))
    out += (jnp.abs(tN) > EPS) * (-I2pp1(s1, tN, B) + I2pp1(sN, tN, B))
    Bp = -B
    out += (
        I2pp1(sN, tNp, Bp)
        - I2pp1(s1, tNp, Bp)
        + I2pp2(sN, tNp, Bp)
        - I2pp2(sN, t1p, Bp)
        - I2pp2(s1, tNp, Bp)
        + I2pp2(s1, t1p, Bp)
        + (jnp.abs(t1p) > EPS) * (-I2pp1(sN, t1p, Bp) + I2pp1(s1, t1p, Bp))
    )
    return out


def S2_term(A, B, C, S, Tn, Tp):
    return 1.0 / safe_sqrt(A * S * S + B * S * Tn + C * Tn * Tn) + 1.0 / safe_sqrt(A * S * S - B * S * Tp + C * Tp * Tp)


def _desing_eta_x_k2(*, S, Tn, Tp, x0, xN, y0, yN, x_star, y_star, eta_x, eta_x_star, eta_y_star, K2, dx_x, dy_y):
    A = 1.0 + eta_x_star**2
    B = 2.0 * eta_x_star * eta_y_star
    C = 1.0 + eta_y_star**2
    I_reg = trapz2d(eta_x * K2 - eta_x_star * S2_term(A, B, C, S, Tn, Tp), dx_x, dy_y)
    s1, sN = x0 - x_star, xN - x_star
    t1, tN = y0 - y_star, yN - y_star
    t1p, tNp = y0 + y_star, yN + y_star
    return I_reg + eta_x_star * desingularize_jax(sN, tN, s1, t1, tNp, t1p, A, B, C)


_BIE_TILE_X_INTERVALS = 32
_BIE_VMAP_OVER_K = True
_BIE_STREAM_OVER_L = True
_BIE_GUARD_MAX_EVALS = 20_000


def _map_bie(fn, My: int, Neval: int, *, vmap_over_k: Optional[bool] = None, stream_over_l: Optional[bool] = None):
    n = int(My) * int(Neval)
    vmap_k = (_BIE_VMAP_OVER_K and (n < _BIE_GUARD_MAX_EVALS)) if vmap_over_k is None else bool(vmap_over_k)
    stream_l = _BIE_STREAM_OVER_L if stream_over_l is None else bool(stream_over_l)
    l_idx, k_idx = jnp.arange(My), jnp.arange(Neval)

    def row(l):
        return jax.vmap(lambda k: fn(l, k))(k_idx) if vmap_k else lax.map(lambda k: fn(l, k), k_idx)

    return lax.map(row, l_idx) if stream_l else jax.vmap(row)(l_idx)


def _k3k4_flatbed(S, Tn, Tp, *, zeta, zetaX, zetaY, zeta_star: float, H: float):
    dz = zeta + zeta_star + 2.0 * H
    r2n = S * S + Tn * Tn + dz * dz
    r2p = S * S + Tp * Tp + dz * dz
    r3n, r3p = r2n * safe_sqrt(r2n), r2p * safe_sqrt(r2p)
    K3 = (dz - S * zetaX - Tn * zetaY) / r3n + (dz - S * zetaX - Tp * zetaY) / r3p
    K4 = 1.0 / safe_sqrt(r2n) + 1.0 / safe_sqrt(r2p)
    return K3, K4


def _flatbed_bie_lk(l: int, k: int, *, x, xm, y, phi, phim, zeta, zetax, zetay, zetam, zetaxm, zetaym, H: float):
    xstar, ystar = xm[k], y[l]
    phistar, zetastar = phim[l, k], zetam[l, k]
    zetaxstar, zetaystar = zetaxm[l, k], zetaym[l, k]
    A = 1.0 + zetaxstar**2
    B = 2.0 * zetaxstar * zetaystar
    C = 1.0 + zetaystar**2
    Tn = (y - ystar)[:, None]
    Tp = (y + ystar)[:, None]
    dy_y = y[1:] - y[:-1]
    dtype = jnp.result_type(phi, zeta, x)
    I1 = jnp.zeros((), dtype=dtype)
    I2_reg = jnp.zeros((), dtype=dtype)
    I3 = jnp.zeros((), dtype=dtype)
    I4 = jnp.zeros((), dtype=dtype)
    nx = int(x.shape[0])
    tile = int(_BIE_TILE_X_INTERVALS)
    for i0 in range(0, nx - 1, tile):
        i1 = min(i0 + tile, nx - 1)
        xs = x[i0 : i1 + 1]
        dx_x = xs[1:] - xs[:-1]
        S = (xs - xstar)[None, :]
        phi_t = phi[:, i0 : i1 + 1]
        zeta_t = zeta[:, i0 : i1 + 1]
        zetax_t = zetax[:, i0 : i1 + 1]
        zetay_t = zetay[:, i0 : i1 + 1]
        K1, K2 = kernels(S, Tn, Tp, zeta_t, zetax_t, zetay_t, zetastar)
        I1 = I1 + trapz2d(((phi_t - phistar) - S) * K1, dx_x, dy_y)
        I2_reg = I2_reg + trapz2d(zetax_t * K2 - zetaxstar * S2_term(A, B, C, S, Tn, Tp), dx_x, dy_y)
        K3, K4 = _k3k4_flatbed(S, Tn, Tp, zeta=zeta_t, zetaX=zetax_t, zetaY=zetay_t, zeta_star=zetastar, H=H)
        I3 = I3 + trapz2d((phi_t - xs[None, :]) * K3, dx_x, dy_y)
        I4 = I4 + trapz2d(zetax_t * K4, dx_x, dy_y)
    sN, s1 = x[-1] - xstar, x[0] - xstar
    tN, t1 = y[-1] - ystar, y[0] - ystar
    tNp, t1p = y[-1] + ystar, y[0] + ystar
    I2_sing = zetaxstar * desingularize_jax(sN, tN, s1, t1, tNp, t1p, A, B, C)
    return -2.0 * jnp.pi * (phistar - xstar) + I1 + (I2_reg + I2_sing) + I3 + I4


def _deep_bie_lk(l: int, k: int, x, xm, y1d, *, dx_x, dy_y, x0, xN, y0, yN, phi, phim, zeta, zetax, zetay, zetam, zetaxm, zetaym):
    zeta_star = zetam[l, k]
    phi_star = phim[l, k]
    x_star = xm[k]
    y_star = y1d[l]
    zetax_star = zetaxm[l, k]
    zetay_star = zetaym[l, k]
    S = (x - x_star)[None, :]
    Tn = (y1d - y_star)[:, None]
    Tp = (y1d + y_star)[:, None]
    K1, K2 = kernels(S, Tn, Tp, zeta, zetax, zetay, zeta_star)
    I1 = trapz2d(((phi - phi_star) - S) * K1, dx_x, dy_y)
    I2 = _desing_eta_x_k2(S=S, Tn=Tn, Tp=Tp, x0=x0, xN=xN, y0=y0, yN=yN, x_star=x_star, y_star=y_star, eta_x=zetax, eta_x_star=zetax_star, eta_y_star=zetay_star, K2=K2, dx_x=dx_x, dy_y=dy_y)
    return -2.0 * jnp.pi * (phi_star - x_star) + I1 + I2


def _surface_bie_lk(l: int, k: int, x, xm, y, *, dx_x, dy_y, x0, xN, y0, yN, phi, psi, zeta, zetax, zetay, beta, betax, betay):
    x_star, y_star = xm[k], y[l]
    phi_star = 0.5 * (phi[l, k] + phi[l, k + 1])
    zeta_star = 0.5 * (zeta[l, k] + zeta[l, k + 1])
    zetax_star = 0.5 * (zetax[l, k] + zetax[l, k + 1])
    zetay_star = 0.5 * (zetay[l, k] + zetay[l, k + 1])
    # betax_star = 0.5 * (betax[l, k] + betax[l, k + 1])
    # betay_star = 0.5 * (betay[l, k] + betay[l, k + 1])
    S = (x - x_star)[None, :]
    Tn = (y - y_star)[:, None]
    Tp = (y + y_star)[:, None]
    K1_zz, K2_zz = kernels(S, Tn, Tp, zeta, zetax, zetay, zeta_star)
    I1 = trapz2d(((phi - phi_star) - S) * K1_zz, dx_x, dy_y)
    I2 = _desing_eta_x_k2(S=S, Tn=Tn, Tp=Tp, x0=x0, xN=xN, y0=y0, yN=yN, x_star=x_star, y_star=y_star, eta_x=zetax, eta_x_star=zetax_star, eta_y_star=zetay_star, K2=K2_zz, dx_x=dx_x, dy_y=dy_y)
    K1_bz, K2_bz = kernels(S, Tn, Tp, beta, betax, betay, zeta_star)
    I3 = trapz2d(((psi - phi_star) - S) * K1_bz, dx_x, dy_y)
    I4 = trapz2d(betax * K2_bz, dx_x, dy_y)
    return I1 + I2 - I3 - I4


def _bottom_bie_lk(l: int, k: int, x, xm, y, *, dx_x, dy_y, x0, xN, y0, yN, phi, psi, zeta, zetax, zetay, beta, betax, betay):
    x_star, y_star = xm[k], y[l]
    psi_star = 0.5 * (psi[l, k] + psi[l, k + 1])
    beta_star = 0.5 * (beta[l, k] + beta[l, k + 1])
    betax_star = 0.5 * (betax[l, k] + betax[l, k + 1])
    betay_star = 0.5 * (betay[l, k] + betay[l, k + 1])
    S = (x - x_star)[None, :]
    Tn = (y - y_star)[:, None]
    Tp = (y + y_star)[:, None]
    K1_bb, K2_bb = kernels(S, Tn, Tp, beta, betax, betay, beta_star)
    L1 = trapz2d(((psi - psi_star) - S) * K1_bb, dx_x, dy_y)
    L2 = _desing_eta_x_k2(S=S, Tn=Tn, Tp=Tp, x0=x0, xN=xN, y0=y0, yN=yN, x_star=x_star, y_star=y_star, eta_x=betax, eta_x_star=betax_star, eta_y_star=betay_star, K2=K2_bb, dx_x=dx_x, dy_y=dy_y)
    K1_zb, K2_zb = kernels(S, Tn, Tp, zeta, zetax, zetay, beta_star)
    L3 = trapz2d(((phi - psi_star) - S) * K1_zb, dx_x, dy_y)
    L4 = trapz2d(zetax * K2_zb, dx_x, dy_y)
    return -L1 - L2 + L3 + L4


def bathymetry_bie_residual(*, phi, psi, zeta, zetax, zetay, beta, betax, betay, x, xm, y):
    x1d = jnp.ravel(x)
    y1d = jnp.ravel(y)
    dx_x = x1d[1:] - x1d[:-1]
    dy_y = y1d[1:] - y1d[:-1]
    x0, xN = x1d[0], x1d[-1]
    y0, yN = y1d[0], y1d[-1]
    My, Neval = int(y1d.shape[0]), int(xm.shape[0])
    surf = _map_bie(lambda l, k: _surface_bie_lk(l, k, x1d, xm, y1d, dx_x=dx_x, dy_y=dy_y, x0=x0, xN=xN, y0=y0, yN=yN, phi=phi, psi=psi, zeta=zeta, zetax=zetax, zetay=zetay, beta=beta, betax=betax, betay=betay), My, Neval)
    bott = _map_bie(lambda l, k: _bottom_bie_lk(l, k, x1d, xm, y1d, dx_x=dx_x, dy_y=dy_y, x0=x0, xN=xN, y0=y0, yN=yN, phi=phi, psi=psi, zeta=zeta, zetax=zetax, zetay=zetay, beta=beta, betax=betax, betay=betay), My, Neval)
    return surf.reshape(My, Neval), bott.reshape(My, Neval)


@partial(jax.jit, static_argnames=("M", "N", "use_radiation", "policy"))
def jax_residual_deep(u, M, N, dx, dy, Fr, aleph, tauf, mu, x, xm, y, pm, *, use_radiation: bool, policy: BiharmonicPolicy):
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
    dyn = dynamic_cond(dx=dx, dy=dy, Fr=Fr, aleph=aleph, mu=mu, xm=xm, zeta=zeta, zetam=zetam, zetaxm=zetaxm, zetaym=zetaym, phim=phim, phixm=phixm, phiym=phiym, pm=pm, tauf=tauf, policy=policy)
    My, Neval = int(y1d.shape[0]), int(xm.shape[0])
    bie = _map_bie(lambda l, k: _deep_bie_lk(l, k, x1d, xm, y1d, dx_x=dx_x, dy_y=dy_y, x0=x0, xN=xN, y0=y0, yN=yN, phi=phi, phim=phim, zeta=zeta, zetax=zetax, zetay=zetay, zetam=zetam, zetaxm=zetaxm, zetaym=zetaym), My, Neval).reshape(My, Neval)
    bc_zeta, bc_zetax, bc_phi, bc_phix = _apply_bc2(x0=x0, dx=dx, use_radiation=use_radiation, zeta=zeta, zetaX=zetax, phi=phi, phiX=phix)
    E1 = jnp.hstack((bc_phi[:, None], bc_phix[:, None], dyn)).reshape(M * (N + 1), 1)
    E2 = jnp.hstack((bc_zeta[:, None], bc_zetax[:, None], bie)).reshape(M * (N + 1), 1)
    return jnp.vstack((E1, E2))[:, 0]


def jax_residual_flatbed(unk, M, N, dx, dy, x, y, xm, pm, Fr, H, aleph, mu, tauf, *, policy: BiharmonicPolicy, use_radiation: bool):
    m, n = int(M), int(N)
    phi1, phix, zeta1, zetax = reshaping2Unknowns(unk, m, n)
    phi, zeta = allVals(phi1, phix, dx, m, n), allVals(zeta1, zetax, dx, m, n)
    phiy, zetay = grad_y(phi, dy, stencil="centered"), grad_y(zeta, dy, stencil="centered")
    zetam, zetaxm, zetaym = midpoints(zeta), midpoints(zetax), midpoints(zetay)
    phim, phixm, phiym = midpoints(phi), midpoints(phix), midpoints(phiy)
    dyn = dynamic_cond(dx=dx, dy=dy, Fr=Fr, aleph=aleph, mu=mu, xm=xm, zeta=zeta, zetam=zetam, zetaxm=zetaxm, zetaym=zetaym, phim=phim, phixm=phixm, phiym=phiym, pm=pm, tauf=tauf, policy=policy)
    y1d = jnp.ravel(y)
    bie = _map_bie(lambda l, k: _flatbed_bie_lk(l, k, x=x, xm=xm, y=y1d, phi=phi, phim=phim, zeta=zeta, zetax=zetax, zetay=zetay, zetam=zetam, zetaxm=zetaxm, zetaym=zetaym, H=H), m, n - 1, vmap_over_k=False).reshape(m, n - 1)
    bc_zeta, bc_zetax, bc_phi, bc_phix = _apply_bc2(x0=x[0], dx=dx, use_radiation=use_radiation, zeta=zeta, zetaX=zetax, phi=phi, phiX=phix)
    E1 = jnp.hstack((bc_phi[:, None], bc_phix[:, None], dyn)).reshape(m * (n + 1), 1)
    E2 = jnp.hstack((bc_zeta[:, None], bc_zetax[:, None], bie)).reshape(m * (n + 1), 1)
    return jnp.vstack((E1, E2))[:, 0]


def jax_residual_bathy(u, M, N, dx, dy, Fr, aleph, mu, x, xm, y, *, use_radiation: bool, policy: BiharmonicPolicy, beta, betax, betay):
    zeta1, zetax, phi1, phix, psi1, psix = reshaping3Unknowns(u, M, N)
    zeta, phi, psi = allVals(zeta1, zetax, dx, M, N), allVals(phi1, phix, dx, M, N), allVals(psi1, psix, dx, M, N)
    zetay, phiy = grad_y(zeta, dy), grad_y(phi, dy)
    zetam, zetaxm, zetaym = midpoints(zeta), midpoints(zetax), midpoints(zetay)
    phim, phixm, phiym = midpoints(phi), midpoints(phix), midpoints(phiy)
    dyn = dynamic_cond(dx=dx, dy=dy, Fr=Fr, aleph=aleph, mu=mu, xm=xm, zeta=zeta, zetam=zetam, zetaxm=zetaxm, zetaym=zetaym, phim=phim, phixm=phixm, phiym=phiym, pm=None, tauf=0.0, policy=policy)
    surf, bott = bathymetry_bie_residual(phi=phi, psi=psi, zeta=zeta, zetax=zetax, zetay=zetay, beta=beta, betax=betax, betay=betay, x=x, xm=xm, y=y)
    bc_zeta, bc_zetax, bc_phi, bc_phix, bc_psi, bc_psix = _apply_bc3(x0=float(jnp.ravel(x)[0]), dx=dx, use_radiation=use_radiation, zeta=zeta, zetax=zetax, phi=phi, phix=phix, psi=psi, psix=psix)
    E1 = jnp.hstack((bc_zeta[:, None], bc_zetax[:, None], dyn)).reshape(int(M) * (int(N) + 1), 1)
    E2 = jnp.hstack((bc_phi[:, None], bc_phix[:, None], surf)).reshape(int(M) * (int(N) + 1), 1)
    E3 = jnp.hstack((bc_psi[:, None], bc_psix[:, None], bott)).reshape(int(M) * (int(N) + 1), 1)
    return jnp.vstack((E1, E2, E3))[:, 0]