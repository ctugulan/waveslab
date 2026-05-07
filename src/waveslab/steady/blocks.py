from __future__ import annotations
# my-packages/pypure/pywave/waves_helpers/build_all_blocks.py
import numpy as np
from scipy.sparse import block_diag, csr_matrix, identity, kron

_FLOAT = np.float32

try:
    _trapz = np.trapezoid
except AttributeError:  
    _trapz = np.trapz

# --------
def analytic_I0_integral(x: np.ndarray, y: np.ndarray, xm: np.ndarray) -> np.ndarray:
    """Analytic integral I0 used to desingularize the R0^{-1} kernel."""
    x_1d = np.asarray(x, dtype=_FLOAT).reshape(-1)
    x_mid_1d = np.asarray(xm, dtype=_FLOAT).reshape(-1)
    y_col = np.asarray(y, dtype=_FLOAT).reshape(-1, 1)

    s1, sN = x_1d[0] - x_mid_1d, x_1d[-1] - x_mid_1d
    t1, tM = y_col[0:1] - y_col, y_col[-1:] - y_col
    log2 = np.log(2.0)

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

    tp, tm = tM + 2.0 * y_col, t1 + 2.0 * y_col
    I += evl(F1, tp) + evl(F2, tp) - evl(F2, tm)

    m0 = (tm[:, 0] != 0.0)
    if m0.any():
        I[m0, :] -= evl(F1, tm[m0, :])

    return I

def _bc_identity(N):
    E = np.zeros((2, int(N) + 1), dtype=_FLOAT); E[0, 0] = 1.0; E[1, 1] = 1.0; return E
    
def _bc_zero(N): return np.zeros((2, int(N) + 1), dtype=_FLOAT)

def _bc_radiation(N, dx, x0, n=0.05):
    N = int(N); dx = float(dx); x0 = float(x0)
    E = np.zeros((2, N + 1), dtype=_FLOAT); inv = 1.0 / dx
    E[0, 0] = float(n); E[0, 1] = x0
    E[1, 1] = float(n) - x0 * inv
    if N >= 2: E[1, 2] = x0 * inv
    return E

def _trapz_dx_op(N, dx):
    N = int(N); dx = float(dx)
    T = np.tri(N - 1, dtype=_FLOAT); T = T + 2.0 * np.tril(T, -1) + np.tril(T, -2)
    v = np.ones((N - 1, 1), dtype=_FLOAT); v[1:, 0] = 2.0
    return np.hstack((np.ones((N - 1, 1), dtype=_FLOAT), (dx / 4.0) * np.hstack((v, T))))

def _halfmesh_avg_op(N):
    N = int(N)
    d = (np.eye(N - 1, N, 0, dtype=_FLOAT) + np.eye(N - 1, N, 1, dtype=_FLOAT)) / 2.0
    return np.hstack((np.zeros((N - 1, 1), dtype=_FLOAT), d))

def _repeat_blockdiag(block, M, sparse):
    block = np.asarray(block, dtype=_FLOAT); M = int(M)
    return np.kron(np.eye(M, dtype=_FLOAT), block) if not sparse else kron(identity(M, format="csr", dtype=_FLOAT), csr_matrix(block), format="csr")

def _xstar_blockdiag(N, M, sparse, dx=None, x0=None, use_radiation=False):
    if sparse:
        E = _bc_radiation(N, float(dx), float(x0)) if use_radiation else _bc_identity(N)
        return _repeat_blockdiag(np.vstack((E, _halfmesh_avg_op(N))), M, True)
    return _repeat_blockdiag(np.vstack((_bc_identity(N), _halfmesh_avg_op(N))), M, False)

def _star_blockdiag(N, M, dx, weights, sparse):
    base = _trapz_dx_op(N, dx)
    if np.isscalar(weights): interior = base * float(weights)
    else: interior = base * np.asarray(weights, dtype=_FLOAT).reshape(-1)[:, None]
    return _repeat_blockdiag(np.vstack((_bc_zero(N), interior)), M, sparse)

# ---- dense kernels ----
def _dense_grids(M, N, x, y, xm):
    M = int(M); N = int(N)
    x = np.asarray(x, dtype=_FLOAT).reshape(-1); y = np.asarray(y, dtype=_FLOAT).reshape(-1); xm = np.asarray(xm, dtype=_FLOAT).reshape(-1)
    x_diff2 = (x[None, :] - xm[:, None]) ** 2
    wy = np.ones((M,), dtype=_FLOAT); wx = np.ones((N,), dtype=_FLOAT)
    if M >= 2: wy[0] *= 0.5; wy[-1] *= 0.5
    if N >= 2: wx[0] *= 0.5; wx[-1] *= 0.5
    return x, y, xm, x_diff2, wy[:, None] * wx[None, :]

def _dense_kernel_Rinv(x_diff2, w2, y, j, a2, scale):
    dy_neg2 = (y - y[j]) ** 2; dy_pos2 = (y + y[j]) ** 2
    inv_rneg = 1.0 / np.sqrt(x_diff2[None, :, :] + dy_neg2[:, None, None] + a2)
    inv_rpos = 1.0 / np.sqrt(x_diff2[None, :, :] + dy_pos2[:, None, None] + a2)
    return (inv_rneg + inv_rpos) * (w2[:, None, :] * scale)

def _dense_bc_rows_all(M, N, width, dx, x0, use_radiation):
    E = _bc_radiation(N, dx, x0) if use_radiation else _bc_identity(N)
    nz0, nz1 = np.flatnonzero(E[0]), np.flatnonzero(E[1])
    bc = np.zeros((int(M), 2, int(width)), dtype=_FLOAT)
    for j in range(int(M)):
        base = j * (int(N) + 1)
        bc[j, 0, base + nz0] = E[0, nz0]; bc[j, 1, base + nz1] = E[1, nz1]
    return bc

def _apply_I0_correction(K0, I0_row, j, N):
    k = np.arange(int(N) - 1)
    corr = -(K0.sum(axis=(0, 2)) + I0_row) / 2.0
    K0[j, k, k] += corr; K0[j, k, k + 1] += corr

# ---- finite-depth flatbed ----
def _flatbed_E2_zeta_block(M, N, dx, dy, *, x0, use_radiation, x, y, xm, H):
    M = int(M); N = int(N); dx = float(dx); dy = float(dy); width = (N + 1) * M
    x1, y1, xm1, x_diff2, w2 = _dense_grids(M, N, x, y, xm)
    I0 = analytic_I0_integral(x1, y1, xm1)
    out = np.zeros((M * (N + 1), width), dtype=_FLOAT)
    bc_all = _dense_bc_rows_all(M, N, width, dx, float(x0), bool(use_radiation))
    a2 = (2.0 * float(H)) ** 2; scale0, scaleH = -dx * dy, dx * dy
    for j in range(M):
        base = j * (N + 1); out[base:base + 2, :] = bc_all[j]
        K0 = _dense_kernel_Rinv(x_diff2, w2, y1, j, 0.0, scale0); _apply_I0_correction(K0, I0[j, :], j, N)
        KH = _dense_kernel_Rinv(x_diff2, w2, y1, j, a2, scaleH)
        K56 = (-K0) + KH
        B = np.concatenate([np.zeros((M, N - 1, 1), dtype=_FLOAT), K56], axis=2)
        out[base + 2: base + 2 + (N - 1), :] = B.transpose(1, 0, 2).reshape(N - 1, width)
    return out

def _flatbed_E2_phi_block(M, N, dx, dy, *, x, y, xm, H):
    M = int(M); N = int(N); dx = float(dx); dy = float(dy); width = (N + 1) * M
    _, y1, _, x_diff2, w2 = _dense_grids(M, N, x, y, xm)
    a = 2.0 * float(H); a2 = a * a
    out = np.zeros((M * (N + 1), width), dtype=_FLOAT)
    for j in range(M):
        base = j * (N + 1)
        dy_n2 = (y1 - y1[j]) ** 2; dy_p2 = (y1 + y1[j]) ** 2
        den_n = (x_diff2[None, :, :] + dy_n2[:, None, None] + a2) ** 1.5
        den_p = (x_diff2[None, :, :] + dy_p2[:, None, None] + a2) ** 1.5
        K1 = (a / den_n) + (a / den_p); K1 *= w2[:, None, :]; K1 *= (dx * dy)
        suffix = np.cumsum(K1[..., ::-1], axis=2)[..., ::-1]
        col0 = suffix[:, :, 0]; col1 = (dx / 2.0) * suffix[:, :, 1]
        mid = dx * suffix[:, :, 2:] + (dx / 2.0) * K1[:, :, 1:-1] if N > 2 else np.zeros((M, N - 1, 0), dtype=_FLOAT)
        last = (dx / 2.0) * K1[:, :, -1]
        B = np.concatenate([col0[..., None], col1[..., None], mid, last[..., None]], axis=2)
        out[base + 2: base + 2 + (N - 1), :] = B.transpose(1, 0, 2).reshape(N - 1, width)
    return out

def build_jacobian_blocks_flatbed(*, N, M, dx, dy, grid, H, Fr, mu, use_radiation=False):
    N = int(N); M = int(M); dx = float(dx); dy = float(dy)
    A = _xstar_blockdiag(N, M, False) + _star_blockdiag(N, M, dx, float(mu), False)
    B = _star_blockdiag(N, M, dx, 1.0 / (float(Fr) ** 2), False)
    C = _star_blockdiag(N, M, dx, -2.0 * np.pi, False) + _flatbed_E2_phi_block(M, N, dx, dy, x=grid.x, y=grid.y, xm=grid.xm, H=float(H))
    D = _flatbed_E2_zeta_block(M, N, dx, dy, x0=float(grid.x0), use_radiation=bool(use_radiation), x=grid.x, y=grid.y, xm=grid.xm, H=float(H))
    return A, B, C, D

# ---- infinite depth ----
def _infinite_depth_D(M, N, dx, dy, *, x0, use_radiation, x, y, xm):
    M = int(M); N = int(N); dx = float(dx); dy = float(dy); width = (N + 1) * M
    x1, y1, xm1, x_diff2, w2 = _dense_grids(M, N, x, y, xm)
    I0 = analytic_I0_integral(x1, y1, xm1)
    out = np.zeros((M * (N + 1), width), dtype=_FLOAT)
    bc_all = _dense_bc_rows_all(M, N, width, dx, float(x0), bool(use_radiation))
    scale0 = -dx * dy
    for j in range(M):
        base = j * (N + 1); out[base:base + 2, :] = bc_all[j]
        K0 = _dense_kernel_Rinv(x_diff2, w2, y1, j, 0.0, scale0); _apply_I0_correction(K0, I0[j, :], j, N)
        B = np.concatenate([np.zeros((M, N - 1, 1), dtype=_FLOAT), K0], axis=2)
        out[base + 2: base + 2 + (N - 1), :] = B.transpose(1, 0, 2).reshape(N - 1, width)
    return out

def build_blocks_infinite_depth(*, N, M, dx, dy, grid, Fr, mu, use_radiation):
    N = int(N); M = int(M); dx = float(dx); dy = float(dy)
    A = _xstar_blockdiag(N, M, True, dx=dx, x0=float(grid.x0), use_radiation=bool(use_radiation))
    A = A + _star_blockdiag(N, M, dx, mu, True)
    C = _star_blockdiag(N, M, dx, 2.0 * np.pi, True)
    B = _star_blockdiag(N, M, dx, 1.0 / (float(Fr) ** 2), True).toarray()
    D = _infinite_depth_D(M, N, dx, dy, x0=float(grid.x0), use_radiation=bool(use_radiation), x=grid.x, y=grid.y, xm=grid.xm)
    return csr_matrix(A), B, csr_matrix(C), D

# ---- bathymetry (3x3) ----
def _bathy_grids(M, N, x, y):
    x = np.asarray(x, dtype=_FLOAT).reshape(-1); y = np.asarray(y, dtype=_FLOAT).reshape(-1)
    x_half = 0.5 * (x[1:] + x[:-1])
    return _dense_grids(int(M), int(N), x, y, x_half)

def _bathy_dzeta_kernel(M, N, dx, dy, *, x, y, a2, scale, apply_I0, row_sign):
    M = int(M); N = int(N); dx = float(dx); dy = float(dy)
    x1, y1, xm1, x_diff2, w2 = _bathy_grids(M, N, x, y)
    I0 = analytic_I0_integral(x1, y1, xm1) if apply_I0 else None
    width = M * (N + 1); out = np.zeros((M * (N + 1), width), dtype=_FLOAT)
    for j in range(M):
        base = j * (N + 1)
        K0 = _dense_kernel_Rinv(x_diff2, w2, y1, j, float(a2), float(scale))
        if apply_I0: _apply_I0_correction(K0, I0[j, :], j, N)
        rows = float(row_sign) * K0
        B = np.concatenate([np.zeros((M, N - 1, 1), dtype=_FLOAT), rows], axis=2)
        out[base + 2: base + 2 + (N - 1), :] = B.transpose(1, 0, 2).reshape(N - 1, width)
    return out

def _bathy_psi_map(M, N, dx, dy, *, x, y):
    M = int(M); N = int(N); dx = float(dx); dy = float(dy)
    _x1, y1, _xm1, x_diff2, w2 = _bathy_grids(M, N, x, y)
    width = M * (N + 1); out = np.zeros((M * (N + 1), width), dtype=_FLOAT)
    for ell in range(M):
        base = ell * (N + 1)
        dy_n2 = (y1 - y1[ell]) ** 2; dy_p2 = (y1 + y1[ell]) ** 2
        den_n = (x_diff2[None, :, :] + dy_n2[:, None, None] + 1.0) ** 1.5
        den_p = (x_diff2[None, :, :] + dy_p2[:, None, None] + 1.0) ** 1.5
        K1 = (1.0 / den_n) + (1.0 / den_p); K1 *= w2[:, None, :]; K1 *= (dx * dy)
        sum_x = K1.sum(axis=2)
        suffix = np.cumsum(K1[..., ::-1], axis=2)[..., ::-1]
        col0 = sum_x; col1 = (dx / 2.0) * suffix[:, :, 1]
        mid = dx * suffix[:, :, 2:] + (dx / 2.0) * K1[:, :, 1:-1] if N > 2 else np.zeros((M, N - 1, 0), dtype=_FLOAT)
        last = (dx / 2.0) * K1[:, :, -1]
        B = np.concatenate([col0[..., None], col1[..., None], mid, last[..., None]], axis=2)
        out[base + 2: base + 2 + (N - 1), :] = B.transpose(1, 0, 2).reshape(N - 1, width)
    return out

def _bathy_astar_blockdiag(N, M, *, dx, x0, n_rad, fancy_bc, const):
    bc = _bc_radiation(N, dx, x0, n=float(n_rad)) if fancy_bc and (float(x0) != 0.0 or float(n_rad) != 0.0) else _bc_identity(N)
    base = _trapz_dx_op(N, dx)
    if np.isscalar(const): interior = base * float(const)
    else: interior = base * np.asarray(const, dtype=_FLOAT).reshape(-1)[:, None]
    return csr_matrix(_repeat_blockdiag(np.vstack((bc, interior)), int(M), True))

def _bathy_phi_xavg_blockdiag(N, M): return csr_matrix(_repeat_blockdiag(np.vstack((_bc_zero(N), _halfmesh_avg_op(N))), int(M), True))

def _bathy_I1_blockdiag(M, N, dx, *, x0, n_rad, fancy_bc, x, y, sign):
    M = int(M); N = int(N); dx = float(dx)
    x = np.asarray(x, dtype=_FLOAT).reshape(-1); y = np.asarray(y, dtype=_FLOAT).reshape(-1)
    x_half = 0.5 * (x[1:] + x[:-1])
    x, y, _xm, x_diff2, _w2 = _dense_grids(M, N, x, y, x_half)
    I1 = np.zeros((M, N - 1), dtype=_FLOAT)
    for ell in range(M):
        dy_n2 = (y - y[ell]) ** 2; dy_p2 = (y + y[ell]) ** 2
        integrand = (x_diff2[None, :, :] + dy_n2[:, None, None] + 1.0) ** (-1.5)
        integrand += (x_diff2[None, :, :] + dy_p2[:, None, None] + 1.0) ** (-1.5)
        int_x = _trapz(integrand, x=x, axis=2); I1[ell, :] = _trapz(int_x, x=y, axis=0)
    bc = _bc_radiation(N, dx, x0, n=float(n_rad)) if fancy_bc and (float(x0) != 0.0 or float(n_rad) != 0.0) else _bc_identity(N)
    base = _trapz_dx_op(N, dx)
    blocks = [csr_matrix(np.vstack((bc, (float(sign) * I1[ell, :])[:, None] * base))) for ell in range(M)]
    return block_diag(blocks, format="csr")

def build_jacobian_blocks_bathy(*, N, M, dx, dy, grid, Fr, mu, n_rad, fancy_bc):
    x = np.asarray(grid.x, dtype=_FLOAT).reshape(-1); y = np.asarray(grid.y, dtype=_FLOAT).reshape(-1); x0 = float(grid.x0)
    neq = int(M) * (int(N) + 1); Z = csr_matrix((neq, neq), dtype=_FLOAT)
    J11 = _bathy_astar_blockdiag(N, M, dx=dx, x0=x0, n_rad=n_rad, fancy_bc=fancy_bc, const=1.0 / (Fr * Fr))
    J12 = _bathy_phi_xavg_blockdiag(N, M) + _bathy_astar_blockdiag(N, M, dx=dx, x0=0.0, n_rad=0.0, fancy_bc=False, const=mu)
    J21 = csr_matrix(_bathy_dzeta_kernel(M, N, dx, dy, x=x, y=y, a2=0.0, scale=-dx * dy, apply_I0=True, row_sign=-1.0))
    J22 = _bathy_I1_blockdiag(M, N, dx, x0=x0, n_rad=n_rad, fancy_bc=fancy_bc, x=x, y=y, sign=-1.0)
    J23 = csr_matrix(_bathy_psi_map(M, N, dx, dy, x=x, y=y))
    J31 = csr_matrix(_bathy_dzeta_kernel(M, N, dx, dy, x=x, y=y, a2=1.0, scale=dx * dy, apply_I0=False, row_sign=+1.0))
    J33 = _bathy_astar_blockdiag(N, M, dx=dx, x0=0.0, n_rad=0.0, fancy_bc=False, const=2.0 * np.pi)
    return {"J11": J11, "J12": J12, "J13": Z, "J21": J21, "J22": J22, "J23": J23, "J31": J31, "J32": Z, "J33": J33}