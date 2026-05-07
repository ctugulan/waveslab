from __future__ import annotations

# my-packages/pypure/pywave/waves_helpers/block_schur_inv.py
import gc

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import LinearOperator, splu

def _sparsity_report(name, A):
    n, m = A.shape
    assert n == m
    if issparse(A):
        nnz = A.nnz
        frac = nnz / (n * n)
        print(f"[sparse] {name}: nnz={nnz:,} | nnz/n^2={frac:.3e} | fmt={A.getformat()}")
    else:
        nnz = int(np.count_nonzero(A))
        frac = nnz / (n * n)
        print(f"[dense ] {name}: nnz={nnz:,} | nnz/n^2={frac:.3e}")

def _rhs2d(x, *, dtype=None):
    if issparse(x):
        x = x.astype(dtype, copy=False).toarray() if dtype is not None else x.toarray()
    else:
        x = np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
    return x if x.ndim == 2 else x.reshape(-1, 1)

def _as_square(M, name="M"):
    if issparse(M):
        m, n = M.shape
        if m != n:
            raise ValueError(f"{name} must be square, got {M.shape}")
        return M
    A = np.asarray(M)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a 2-D square array, got shape={A.shape}")
    return A

def _make_solve(M):
    M = _as_square(M)
    if issparse(M):
        M_csc = csc_matrix(M)
        lu = splu(M_csc)
        fac_dtype = lu.L.dtype

        def _solve(B):
            return lu.solve(_rhs2d(B, dtype=fac_dtype))

        return _solve

    A = np.asarray(M)
    fac = lu_factor(A)
    fac_dtype = fac[0].dtype

    def _solve(B):
        return lu_solve(fac, _rhs2d(B, dtype=fac_dtype))

    return _solve

def _to_dense_fp32_from_sparse(S):
    return S.astype(np.float32, copy=False).toarray()

def _to_dense_fp32(M, *, name="M") -> np.ndarray:
    if issparse(M):
        A = _to_dense_fp32_from_sparse(M)
    else:
        A = np.asarray(M, dtype=np.float32, order="C")
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got ndim={A.ndim}")
    return np.asarray(A, dtype=np.float32, order="C")

def _add_sparse_into_dense_inplace(dense: np.ndarray, S):
    if not issparse(S):
        dense += np.asarray(S, dtype=dense.dtype)
        return
    coo = S.tocoo(copy=False)
    dense[coo.row, coo.col] += coo.data.astype(dense.dtype, copy=False)

class J3GMRESPrecond:
    def __init__(self, A, B, D, E, F, G, H, K, *, memlog: bool = False):
        A = _as_square(A, "A")
        E = _as_square(E, "E")
        K = _as_square(K, "K")
        n, m, k = A.shape[0], E.shape[0], K.shape[0]
        self.n, self.m, self.k = n, m, k

        self.dtype = np.result_type(
            (A.dtype if hasattr(A, "dtype") else np.float32),
            (E.dtype if hasattr(E, "dtype") else np.float32),
            (K.dtype if hasattr(K, "dtype") else np.float32),
            (B.dtype if hasattr(B, "dtype") else np.float32),
            (D.dtype if hasattr(D, "dtype") else np.float32),
            (F.dtype if hasattr(F, "dtype") else np.float32),
            (G.dtype if hasattr(G, "dtype") else np.float32),
            (H.dtype if hasattr(H, "dtype") else np.float32),
        )

        prec_dtype = np.float32
        self._prec_dtype = prec_dtype

        self.D = _to_dense_fp32(D, name="D")
        self.F = _to_dense_fp32(F, name="F")
        self.G = _to_dense_fp32(G, name="G")
        self.H = _to_dense_fp32(H, name="H")
        self.B = B.astype(prec_dtype, copy=False) if issparse(B) else np.asarray(B, dtype=prec_dtype, order="C")

        if not (
            self.B.shape == (n, m)
            and self.D.shape == (m, n)
            and self.F.shape == (m, k)
            and self.G.shape == (k, n)
            and self.H.shape == (k, m)
        ):
            raise ValueError(
                "Incompatible block shapes: "
                f"A{A.shape}, B{self.B.shape}, D{self.D.shape}, "
                f"E{E.shape}, F{self.F.shape}, G{self.G.shape}, "
                f"H{self.H.shape}, K{K.shape}"
            )

        self.solve_A = _make_solve(A)

        AinvB = self.solve_A(self.B)
        AinvB = np.asarray(AinvB, dtype=prec_dtype, order="C")

        item = np.dtype(prec_dtype).itemsize
        need_gib = (n * m * item) / 2**30

        DAinvB = self.D @ AinvB
        GAinvB = self.G @ AinvB

        del AinvB
        gc.collect()

        Eprime = DAinvB
        Eprime *= -1.0
        _add_sparse_into_dense_inplace(Eprime, E.astype(prec_dtype, copy=False))

        Hprime = self.H
        if issparse(Hprime):
            Hprime = np.asarray(Hprime.toarray(), dtype=prec_dtype, order="C")
        Hprime -= GAinvB

        del DAinvB, GAinvB
        gc.collect()

        self.solve_Ep = _make_solve(Eprime)

        del Eprime
        gc.collect()

        Ep_inv_F = self.solve_Ep(self.F)
        Ep_inv_F = np.asarray(Ep_inv_F, dtype=prec_dtype, order="C")

        S = Hprime @ Ep_inv_F
        S *= -1.0
        _add_sparse_into_dense_inplace(S, K.astype(prec_dtype, copy=False))

        self.Hprime = Hprime
        del Ep_inv_F, Hprime
        gc.collect()

        self.solve_S = _make_solve(S)

        del S
        gc.collect()

    def _apply(self, b):
        b2d = _rhs2d(b)
        n, m = self.n, self.m
        b1 = b2d[:n, :]
        b2 = b2d[n : n + m, :]
        b3 = b2d[n + m :, :]

        bd = self._prec_dtype
        b1p = np.asarray(b1, dtype=bd, order="C")
        b2p = np.asarray(b2, dtype=bd, order="C")
        b3p = np.asarray(b3, dtype=bd, order="C")

        s1 = self.solve_A(b1p)
        s1 = np.asarray(s1, dtype=bd)
        t2 = b2p - (self.D @ s1)
        t3 = b3p - (self.G @ s1)

        s2 = self.solve_Ep(t2)
        s2 = np.asarray(s2, dtype=bd)
        s3 = self.solve_S(t3 - (self.Hprime @ s2))
        s3 = np.asarray(s3, dtype=bd)

        r3 = s3
        r2 = s2 - self.solve_Ep(self.F @ r3)
        r2 = np.asarray(r2, dtype=bd)
        r1 = s1 - self.solve_A(self.B @ r2)
        r1 = np.asarray(r1, dtype=bd)

        out = np.vstack((r1, r2, r3)).astype(self.dtype, copy=False)
        return out if b2d.shape[1] > 1 else out.ravel()

    def as_linear_operator(self):
        N = self.n + self.m + self.k
        return LinearOperator((N, N), matvec=self._apply, dtype=self.dtype)

def ldu3_prec_operator(A, B, D, E, F, G, H, K, *, memlog: bool = True) -> LinearOperator:
    return J3GMRESPrecond(A, B, D, E, F, G, H, K, memlog=memlog).as_linear_operator()

class J2GMRESPrecond:
    def __init__(self, A, B, C, D, *, prec_dtype=np.float32):
        A = _as_square(A, "A")
        D = _as_square(D, "D")
        n, m = A.shape[0], D.shape[0]

        prec_dtype = np.dtype(prec_dtype)
        if prec_dtype != np.float32:
            raise ValueError(f"prec_dtype must be float32 for this preconditioner, got {prec_dtype}")
        self._prec_dtype = prec_dtype

        self.B = _to_dense_fp32(B, name="B")
        self.C = _to_dense_fp32(C, name="C")
        if self.B.shape != (n, m) or self.C.shape != (m, n):
            raise ValueError(f"Incompatible shapes: A{A.shape}, B{self.B.shape}, C{self.C.shape}, D{D.shape}")

        self.n, self.m = n, m
        self.dtype = np.result_type(
            (A.dtype if hasattr(A, "dtype") else np.float32),
            (D.dtype if hasattr(D, "dtype") else np.float32),
            self.B.dtype,
            self.C.dtype,
        )

        self.solve_A = _make_solve(A)

        AinvB = self.solve_A(self.B)
        AinvB = np.asarray(AinvB, dtype=prec_dtype, order="C")
        CAinvB = self.C @ AinvB

        del AinvB
        gc.collect()

        S = CAinvB
        S *= -1.0
        _add_sparse_into_dense_inplace(S, D)

        del CAinvB
        gc.collect()

        self.solve_S = _make_solve(S)

        del S
        gc.collect()

    def _apply(self, b):
        b = _rhs2d(b)
        n = self.n
        b1, b2 = b[:n, :], b[n:, :]
        s1 = self.solve_A(b1)
        t2 = b2 - (self.C @ s1)
        s2 = self.solve_S(t2)
        r1 = s1 - self.solve_A(self.B @ s2)
        out = np.vstack((r1, s2))
        return out if b.shape[1] > 1 else out.ravel()

    def as_linear_operator(self) -> LinearOperator:
        N = self.n + self.m
        return LinearOperator((N, N), matvec=self._apply, dtype=self.dtype)

def ldu2_prec_operator(A, B, C, D) -> LinearOperator:
    return J2GMRESPrecond(A, B, C, D).as_linear_operator()
