from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import NoConvergence, newton_krylov


@dataclass
class NewtonKrylovSolver:
    """Small wrapper around :func:`scipy.optimize.newton_krylov`.

    The old code used a private solver class with this interface.  Keeping this
    wrapper preserves the call sites while making the dependency explicit and
    public.
    """

    method: str = "lgmres"
    verbose: bool = False

    def solve(
        self,
        residual: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        *,
        preconditioner: Any | None = None,
        f_tol: float = 1e-10,
        maxiter: int = 50,
        line_search: str | None = "wolfe",
        callback: Callable[[np.ndarray, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        x0_arr = np.asarray(x0).reshape(-1)

        def fun(x: np.ndarray) -> np.ndarray:
            return np.asarray(residual(np.asarray(x))).reshape(-1)

        try:
            sol = newton_krylov(
                fun,
                x0_arr,
                method=self.method,
                inner_M=preconditioner,
                f_tol=float(f_tol),
                maxiter=int(maxiter),
                line_search=line_search,
                verbose=bool(self.verbose),
                callback=callback,
            )
        except NoConvergence as exc:
            sol = getattr(exc, "args", [None])[0]
            if sol is None:
                raise
            final = np.linalg.norm(fun(np.asarray(sol)), ord=np.inf)
            raise RuntimeError(f"Newton-Krylov did not converge; final ||F||_inf={final:.3e}") from exc
        return np.asarray(sol).reshape(np.asarray(x0).shape)
