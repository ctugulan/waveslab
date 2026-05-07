from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import time
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

USE_X64 = os.environ.get("VISCICE_USE_X64", "1") == "1"  # set to "0" for float32

from jax import config as jax_config

jax_config.update("jax_enable_x64", USE_X64)

import jax
import jax.numpy as jnp

_FLOAT = np.float64 if USE_X64 else np.float32
_JFLOAT = jnp.float64 if USE_X64 else jnp.float32

# -----------------------------------------------------------------------------
# Driver for steady cases
# waveslab.steady.driver
# -----------------------------------------------------------------------------
# ice helpers
from .biharmonic import (
    BiharmonicPolicy,
    UpstreamBC,
    PadMode,
    flexural_contribution,
    make_biharmonic_policy_from_params,
)
# equations helpers
from .residuals import (
    jax_residual_flatbed,
    jax_residual_deep,
    allVals,
    reshaping2Unknowns,
    jax_residual_bathy,
    reshaping3Unknowns,
)
# preconditioner construction helpers
from .blocks import (
    build_jacobian_blocks_flatbed,
    build_blocks_infinite_depth,
    build_jacobian_blocks_bathy,
)
# preconditioner application helpers
from .preconditioners import ldu2_prec_operator, ldu3_prec_operator
# numerical solver helper
from .newton import NewtonKrylovSolver
# output helpers
from .names import out_paths, zeta_case_tag
from .plotting import save_case_outputs

# Simulation setup
# -----------------------------------------------------------------------------
System = Literal["flatbed", "deep", "bathy"]

@dataclass(frozen=True)
class Grid2D:
    x: np.ndarray  # (N,)
    xm: np.ndarray  # (N-1,)
    y: np.ndarray  # (M,1)

    @property
    def x0(self) -> float:
        return float(self.x[0])


def make_grid(N: int, M: int, dx: float, dy: float, x0: float | None = None) -> Grid2D:
    if x0 is None:
        x0 = -(float(dx) * int(N)) / 2.0
    x = float(x0) + float(dx) * np.arange(int(N))
    y = (float(dy) * np.arange(int(M)))[:, None]
    xm = 0.5 * (x[1:] + x[:-1])
    return Grid2D(np.asarray(x, _FLOAT), np.asarray(xm, _FLOAT), np.asarray(y, _FLOAT))


def initial_guess(M: int, N: int, x0: float) -> np.ndarray:
    dof = int(M) * (int(N) + 1)
    col = np.vstack(([float(x0)], np.ones((int(N), 1), dtype=_FLOAT))).astype(_FLOAT, copy=False)
    return np.vstack((np.tile(col, (int(M), 1)), np.zeros((dof, 1), dtype=_FLOAT))).astype(_FLOAT, copy=False)


def initial_guess3(M: int, N: int, x0: float) -> np.ndarray:
    dof = int(M) * (int(N) + 1)
    col = np.vstack(([float(x0)], np.ones((int(N), 1), dtype=_FLOAT))).astype(_FLOAT, copy=False)
    blk = np.tile(col, (int(M), 1)).astype(_FLOAT, copy=False)
    return np.vstack((np.zeros((dof, 1), dtype=_FLOAT), blk, blk)).astype(_FLOAT, copy=False)

# Forcing helpers
def pressure(x: jnp.ndarray, y: jnp.ndarray, *, eps: float, Lx: float, Ly: float) -> jnp.ndarray:
    x = jnp.ravel(x)
    y = jnp.ravel(y)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    cond = (jnp.abs(X) < Lx) & (jnp.abs(Y) < Ly)
    val = jnp.exp((Lx * Lx) / (X * X - Lx * Lx) + (Ly * Ly) / (Y * Y - Ly * Ly))
    return eps * jnp.where(cond, val, 0.0)


def topography(
    x: jnp.ndarray, y: jnp.ndarray, eps: float, *, delta: float = 0.5, bx: float = 0.0, by: float = 0.0
):
    bump = eps * jnp.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2.0 * delta**2))
    return -1.0 + bump, -((x - bx) / (delta**2)) * bump, -((y - by) / (delta**2)) * bump

# -----------------------------------------------------------------------------
# small utilities
# def fmt_hms_ms(seconds: float) -> str:
#     h = int(seconds // 3600)
#     m = int((seconds % 3600) // 60)
#     s = int(seconds % 60)
#     ms = int((seconds - int(seconds)) * 1000)
#     return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _fmt(dt: float) -> str:
    return f"{float(dt):.3f}s"


def _residual_vec_from_f(f_jax):
    def residual_vec(u_np: np.ndarray) -> np.ndarray:
        r = f_jax(jnp.asarray(u_np, dtype=_JFLOAT).reshape(-1))
        return np.asarray(jax.device_get(r), dtype=_FLOAT).reshape(-1)

    return residual_vec


def _plot_and_save(grid: Grid2D, Z: np.ndarray, csv_path: Path, fig_dir: Path) -> Path:
    paths = save_case_outputs(
        X1d=np.asarray(grid.x).reshape(-1),
        Y1d=np.asarray(grid.y).reshape(-1),
        Z=Z,
        csv_path=csv_path,
        figs_dir=fig_dir,
        stem=Path(csv_path).stem,
        slice_kind="absmax",
        mirror_y0=True,
        save_csv=True,
    )
    return paths["surface"]


# -----------------------------------------------------------------------------
# Case runners
# -----------------------------------------------------------------------------

def _run_case_2(
    *,
    # tag: str,
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
    B_flex = flexural_contribution(int(N), int(M), float(dx), float(dy), aleph=float(aleph), tauf=float(tauf), policy=policy)
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
    fig_path = _plot_and_save(grid, Z, csv_path, fig_dir)
    return Z, csv_path, fig_path


def run_case_flatbed(
    *,
    out_base: str | Path | None = None,
    upstream_bc: UpstreamBC | None = None,
    mode: PadMode | None = None,
    announce_policy: bool = True,
    N: int = 160,
    M: int = 81,
    dx: float = 0.3,
    dy: float = 0.6,
    x0: float | None = None,
    H: float = 1.0,
    Fr: float = 2.0,
    aleph: float = 0.1,
    tauf: float = 0.1,
    mu: float = 0.1,
    epsilon: float = 1.0,
    Lx: float = 1.0,
    Ly: float = 1.0,
    # use_radiation: bool = False,
) -> tuple[np.ndarray, Path, Path]:
    system: System = "flatbed"
    grid = make_grid(N, M, dx, dy, x0=x0)
    policy = make_biharmonic_policy_from_params(
        depth="finite",
        F=float(Fr),
        aleph=float(aleph),
        upstream_bc=upstream_bc,
        mode=mode,
        announce=bool(announce_policy),
    )
    upstream_bc = policy.upstream_bc
    mode = policy.mode

    fancy_bc = False if float(mu) > 0.0 else True
    results_dir, fig_dir = out_paths(system=system, out_base=out_base)

    tag = zeta_case_tag(
        system=system,
        M=M, N=N, dx=dx, dy=dy, x0=grid.x0,
        Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu, tauf=tauf,
        use_radiation=fancy_bc,        
        upstream_bc=upstream_bc,
        mode=mode,
    )
    csv_path  = results_dir / f"zeta_{tag}.csv"
    
    print("[backend]", jax.default_backend(), "| devices:", jax.devices())
    print("[case]", tag)
    print("[out_base]", results_dir.parents[2])

    xm = jnp.asarray(grid.xm, dtype=_JFLOAT)
    y1d = jnp.asarray(np.ravel(np.asarray(grid.y)), dtype=_JFLOAT)
    pm = pressure(xm, y1d, eps=float(epsilon), Lx=float(Lx), Ly=float(Ly))
    params = dict(
        M=int(M),
        N=int(N),
        dx=float(dx),
        dy=float(dy),
        H=float(H),
        Fr=float(Fr),
        aleph=float(aleph),
        tauf=float(tauf),
        mu=float(mu),
        x=jnp.asarray(grid.x, dtype=_JFLOAT),
        xm=xm,
        y=y1d,
        pm=jnp.asarray(pm, dtype=_JFLOAT),
        policy=policy,
        use_radiation=bool(fancy_bc),
    )
    f_jax = lambda u_flat: jax_residual_flatbed(u_flat, **params)
    u0 = initial_guess(M=int(M), N=int(N), x0=grid.x0)

    def build_blocks():
        return build_jacobian_blocks_flatbed(
            N=int(N),
            M=int(M),
            dx=float(dx),
            dy=float(dy),
            grid=grid,
            H=float(H),
            Fr=float(Fr),
            mu=float(mu),
            use_radiation=bool(fancy_bc),
        )

    return _run_case_2(
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
    )


def run_case_deep(
    *,
    out_base: str | Path | None = None,
    upstream_bc: UpstreamBC | None = None,
    mode: PadMode | None = None,
    announce_policy: bool = True,
    N: int = 160,
    M: int = 81,
    dx: float = 0.3,
    dy: float = 0.6,
    x0: float | None = None,
    Fr: float = 2.0,
    aleph: float = 0.5,
    mu: float = 0.0,
    epsilon: float = 0.1,
    Lx: float = 1.0,
    Ly: float = 1.0,
    tauf: float = 0.0,
    use_radiation: bool = False,
) -> tuple[np.ndarray, Path, Path]:
    system: System = "deep"
    grid = make_grid(N, M, dx, dy, x0=x0)
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
        M=M, N=N, dx=dx, dy=dy, x0=grid.x0,
        Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu, tauf=tauf,
        use_radiation= use_radiation,
        upstream_bc=upstream_bc,
        mode=mode,
    )
    csv_path  = results_dir / f"zeta_{tag}.csv"

    print("[backend]", jax.default_backend(), "| devices:", jax.devices())
    print("[case]", tag)
    print("[out_base]", results_dir.parents[2])

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
    f_jax = lambda u_flat: jax_residual_deep(u_flat, **params)
    u0 = initial_guess(M=int(M), N=int(N), x0=grid.x0)

    def build_blocks():
        return build_blocks_infinite_depth(
            N=int(N),
            M=int(M),
            dx=float(dx),
            dy=float(dy),
            grid=grid,
            Fr=float(Fr),
            mu=float(mu),
            use_radiation=bool(use_radiation),
        )

    return _run_case_2(
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
    )


def run_case_bathy(
    *,
    out_base: str | Path | None = None,
    upstream_bc: UpstreamBC | None = None,
    mode: PadMode | None = None,
    announce_policy: bool = True,
    N: int = 160,
    M: int = 80,
    dx: float = 0.2,
    dy: float = 0.2,
    x0: float | None = None,
    Fr: float = 0.3,
    aleph: float = 0.5,
    mu: float = 0.0,
    epsilon: float = 0.1,
    delta: float = 0.5,
    bx: float = 0.0,
    by: float = 0.0,
    tauf: float = 0.0,
) -> tuple[np.ndarray, Path, Path]:
    system: System = "bathy"
    grid = make_grid(N, M, dx, dy, x0=x0)
    policy = make_biharmonic_policy_from_params(
        depth="finite",
        F=float(Fr),
        aleph=float(aleph),
        upstream_bc=upstream_bc,
        mode=mode,
        announce=bool(announce_policy),
    )
    upstream_bc = policy.upstream_bc
    mode = policy.mode

    fancy_bc = False if float(mu) > 0.0 else True
    
    results_dir, fig_dir = out_paths(system=system, out_base=out_base)

    tag = zeta_case_tag(
        system=system,
        M=M, N=N, dx=dx, dy=dy, x0=grid.x0,
        Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu, tauf=tauf,
        use_radiation=fancy_bc,  
        upstream_bc=upstream_bc,
        mode=mode,
    )
    csv_path  = results_dir / f"zeta_{tag}.csv"

    print("[backend]", jax.default_backend(), "| devices:", jax.devices())
    print("[case]", tag)
    print("[out_base]", results_dir.parents[2])

    x1d = jnp.asarray(grid.x)
    xm = jnp.asarray(grid.xm)  
    y1d = jnp.asarray(np.ravel(np.asarray(grid.y)), dtype=_JFLOAT)
    X, Y = x1d[None, :], y1d[:, None]
    beta, betax, betay = topography(X, Y, float(epsilon), delta=float(delta), bx=float(bx), by=float(by))

    params = dict(
        M=int(M),
        N=int(N),
        dx=float(dx),
        dy=float(dy),
        Fr=float(Fr),
        aleph=float(aleph),
        mu=float(mu),
        x=x1d,
        xm=xm,
        y=y1d,
        policy=policy,
        use_radiation=bool(fancy_bc),
        beta=beta,
        betax=betax,
        betay=betay,
    )
    f_jax = lambda u_flat: jax_residual_bathy(u_flat, **params)
    residual_vec = _residual_vec_from_f(f_jax)

    u0 = initial_guess3(M=int(M), N=int(N), x0=grid.x0)
    _ = jax.block_until_ready(f_jax(jnp.asarray(u0, dtype=_JFLOAT).reshape(-1)))
    print("[check] ||F(u0)||2 =", float(np.linalg.norm(residual_vec(u0))))

    t0 = time.time()
    blocks = build_jacobian_blocks_bathy(
        N=int(N),
        M=int(M),
        dx=float(dx),
        dy=float(dy),
        grid=grid,
        Fr=float(Fr),
        mu=float(mu),
        n_rad=0.05,
        fancy_bc=bool(fancy_bc),
    )
    J11, J12, J21, J22, J23, J31 = (
        blocks["J11"],
        blocks["J12"],
        blocks["J21"],
        blocks["J22"],
        blocks["J23"],
        blocks["J31"],
    )
    print("[timer] analytic blocks:", _fmt(time.time() - t0))

    t1 = time.time()
    PFlex = flexural_contribution(int(N), int(M), float(dx), float(dy), aleph=float(aleph), tauf=float(tauf), policy=policy)
    PFlex = np.asarray(PFlex)
    PFlex[np.abs(PFlex) < 1e-14] = 0.0
    J11 = J11 + csr_matrix(PFlex)
    print("[timer] P_flex (+ add):", _fmt(time.time() - t1))

    t2 = time.time()
    Pinv = ldu3_prec_operator(J11, J12, J21, J22, J23, J31, J23, J22, memlog=False)
    print("[timer] invert 3x3 prec:", _fmt(time.time() - t2))

    t3 = time.time()
    u_star = NewtonKrylovSolver(method="lgmres", verbose=True).solve(residual_vec, np.asarray(u0), preconditioner=Pinv)
    print("[timer] solve:", _fmt(time.time() - t3))
    print("[done]  ||F(u*)||2 =", float(np.linalg.norm(residual_vec(u_star))))

    zeta1, zetax, _, _, _, _ = reshaping3Unknowns(jnp.asarray(u_star), int(M), int(N))
    Z = np.asarray(jax.device_get(allVals(zeta1, zetax, float(dx), int(M), int(N))))
    fig_path = _plot_and_save(grid, Z, csv_path, fig_dir)
    print("[save]", csv_path)
    return Z, csv_path, fig_path


def main() -> None:
    # python -m waveslab.steady.driver
    # run_case_deep(epsilon=1.0)
    # run_case_flatbed(epsilon=0.1)
    run_case_bathy(epsilon=0.1)


if __name__ == "__main__":
    main()