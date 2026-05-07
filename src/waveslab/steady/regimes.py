# my-packages/pypure/pywave/waves_helpers/classify_regimes.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Tuple


def classify_infinite_depth_regime(F: float, aleph: float, rtol: float = 1e-9) -> Dict[str, float | str]:
    """
    Infinite-depth *elastic* dispersion regime classifier using nondimensional (F, aleph):
      - F = U / sqrt(g*L)  (nondimensional speed)
      - aleph = D / (rho g L^4)  (nondimensional flexural rigidity)
    """
    if F <= 0 or aleph <= 0:
        raise ValueError("F and aleph must be positive.")

    lam = aleph * (F ** 3)
    lam_star = 27.0 / 256.0

    # Ratio U/c_min = (lam_star/lam)^(1/8)
    U_over_cmin = (lam_star / lam) ** 0.125

    if lam > lam_star * (1.0 + rtol):
        regime = "U < c_min"
        code = 1
    elif lam < lam_star * (1.0 - rtol):
        regime = "c_min < U"
        code = 2
    else:
        regime = "U ≈ c_min"
        code = 0

    return {
        "regime": regime,
        "code": code,
        "F": F,
        "aleph": aleph,
        "lambda": lam,
        "lambda_star": lam_star,
        "U_over_cmin": U_over_cmin,  # <1 => subcritical wrt c_min
    }


def classify_finite_depth_regime(F: float, aleph: float, rtol: float = 1e-9) -> Dict[str, float | str]:
    """
    Finite-depth *elastic* (no viscoelastic term) regime classifier using nondimensional
    (F, aleph) with fixed H.
      - F = U / sqrt(g H)
      - aleph = D / (rho U^2 H^3)
      - beta = aleph * F^2 = D / (rho g H^4)

    The elastic finite-depth dispersion has a single interior minimum c_min,
    and the long-wave limit is c0 = sqrt(gH) (so c0_nd = 1).
    """
    if F <= 0 or aleph <= 0:
        raise ValueError("F and aleph must be positive.")

    beta = aleph * (F ** 2)

    def sech2(y: float) -> float:
        c = math.cosh(y)
        return 1.0 / (c * c)

    # g(y) = 0 gives critical y_*
    def g(y: float) -> float:
        return 1.0 - 3.0 * beta * (y ** 4) - (y + beta * (y ** 5)) * sech2(y)

    a = 1e-10
    fa = g(a)
    b = max(10.0, 1.5 * (1.0 / (3.0 * beta)) ** 0.25)
    fb = g(b)

    grow = 0
    while fa * fb > 0.0 and grow < 60:
        b *= 1.5
        fb = g(b)
        grow += 1

    if fa * fb > 0.0:
        raise RuntimeError("Failed to bracket the critical root y*>0.")

    for _ in range(200):
        m = 0.5 * (a + b)
        fm = g(m)
        if abs(fm) <= 1e-14 or (b - a) / max(m, 1.0) < 1e-12:
            y_star = m
            break
        if fa * fm > 0.0:
            a, fa = m, fm
        else:
            b, fb = m, fm
    else:
        y_star = 0.5 * (a + b)

    t = math.tanh(y_star)
    F_star = math.sqrt(t / y_star + beta * (y_star ** 3) * t)

    def le(x: float, y: float) -> bool:
        return x <= y * (1.0 + rtol)

    def ge(x: float, y: float) -> bool:
        return x >= y * (1.0 - rtol)

    if le(F, F_star):
        regime = "U < c_min"
        code = 1
    elif le(F_star, F) and le(F, 1.0):
        regime = "c_min < U < c0"
        code = 2
    elif ge(F, 1.0):
        regime = "U > c0"
        code = 3
    else:
        regime = "borderline (check tolerances)"
        code = 0

    return {
        "regime": regime,
        "code": code,
        "F": F,
        "aleph": aleph,
        "beta": beta,
        "y_star": y_star,
        "F_star": F_star,
    }


# -----------------------------------------------------------------------------
# Zhestkaya (viscoelastic) finite-depth dispersion: critical speeds + regimes
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ZhestkayaCritical:
    """
    Critical quantities for the Zhestkaya (viscoelastic) finite-depth dispersion,
    in *nondimensional* form.

    All speeds are normalized by sqrt(gH), i.e. c_nd = c / sqrt(gH).
    The wavenumber variable is y = kH.
    """
    beta: float
    tau: float  # tau_nd := tau_f * sqrt(g/H)
    c0: float
    c_min: Optional[float]
    c_max: Optional[float]
    y_cmin: Optional[float]
    y_cmax: Optional[float]
    y_cutoff: Optional[float]  # largest y>0 with c_nd(y)=0, if it exists


def _tanh_over_y(y: float) -> float:
    ay = abs(y)
    if ay < 1e-6:
        y2 = y * y
        return 1.0 - y2 / 3.0 + 2.0 * (y2 * y2) / 15.0
    return math.tanh(y) / y


def _sech2(y: float) -> float:
    c = math.cosh(y)
    return 1.0 / (c * c)


def zhestkaya_c2_nd(y: float, *, beta: float, tau: float) -> float:
    """
    Zhestkaya viscoelastic dispersion relation (phase speed squared), nondimensional.

    With y=kH, beta = D/(rho g H^4), and tau = tau_f * sqrt(g/H):

        c_nd(y)^2
          = tanh(y)/y
            + beta * y^3 * tanh(y)
            - (beta^2 * tau^2 / 4) * y^8 * tanh(y)^2
    """
    if y <= 0.0:
        raise ValueError("y must be positive (y = kH).")
    t = math.tanh(y)
    term_grav = _tanh_over_y(y)
    term_flex = beta * (y ** 3) * t
    term_visc = (beta * beta) * (tau * tau) * 0.25 * (y ** 8) * (t * t)
    return term_grav + term_flex - term_visc


def zhestkaya_dc2_dy(y: float, *, beta: float, tau: float) -> float:
    if y <= 0.0:
        raise ValueError("y must be positive (y = kH).")
    t = math.tanh(y)
    s2 = _sech2(y)

    d_term_grav = (s2 * y - t) / (y * y)
    d_term_flex = beta * (3.0 * (y ** 2) * t + (y ** 3) * s2)

    coef = (beta * beta) * (tau * tau) * 0.25
    d_y8_t2 = 8.0 * (y ** 7) * (t * t) + (y ** 8) * 2.0 * t * s2
    d_term_visc = coef * d_y8_t2

    return d_term_grav + d_term_flex - d_term_visc


def _bisect_root(
    f,
    a: float,
    b: float,
    *,
    max_iter: int = 200,
    atol: float = 1e-14,
    rtol: float = 1e-12,
) -> float:
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise RuntimeError("Bisection requires a sign change on [a,b].")
    lo, hi = a, b
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) <= atol or (hi - lo) <= rtol * max(1.0, abs(mid)):
            return mid
        if fa * fm > 0.0:
            lo, fa = mid, fm
        else:
            hi, fb = mid, fm
    return 0.5 * (lo + hi)


def _find_cutoff_y(beta: float, tau: float, *, y0: float = 1.0, y_max: float = 1e3) -> Optional[float]:
    def f(y: float) -> float:
        return zhestkaya_c2_nd(y, beta=beta, tau=tau)

    y = max(y0, 1e-6)
    f_y = f(y)

    if f_y < 0.0:
        for _ in range(80):
            y *= 0.5
            if y < 1e-12:
                return None
            f_y = f(y)
            if f_y > 0.0:
                break
        else:
            return None

    y_prev, _f_prev = y, f_y
    while y < y_max:
        y *= 1.5
        f_cur = f(y)
        if f_cur <= 0.0:
            return _bisect_root(f, y_prev, y)
        y_prev, _f_prev = y, f_cur

    return None


def find_zhestkaya_critical_speeds(
    *,
    beta: float,
    tau: float,
    n_scan: int = 4000,
    y_scan_max: float = 60.0,
) -> ZhestkayaCritical:
    if beta <= 0.0 or tau < 0.0:
        raise ValueError("beta must be positive and tau must be nonnegative.")

    c0 = 1.0
    y_cut = _find_cutoff_y(beta, tau) if tau > 0.0 else None

    y_hi = y_scan_max
    if y_cut is not None:
        y_hi = min(y_hi, 0.999 * y_cut)
        y_hi = max(y_hi, 1e-3)

    y_lo = 1e-6
    ys = [y_lo + (y_hi - y_lo) * (i / (n_scan - 1)) for i in range(n_scan)]
    dvals = []
    for y in ys:
        try:
            dvals.append(zhestkaya_dc2_dy(y, beta=beta, tau=tau))
        except ValueError:
            dvals.append(float("nan"))

    roots: List[float] = []
    for i in range(len(ys) - 1):
        a, b = ys[i], ys[i + 1]
        fa, fb = dvals[i], dvals[i + 1]
        if not (math.isfinite(fa) and math.isfinite(fb)):
            continue
        if fa == 0.0:
            roots.append(a)
            continue
        if fa * fb < 0.0:
            root = _bisect_root(lambda z: zhestkaya_dc2_dy(z, beta=beta, tau=tau), a, b)
            if not roots or abs(root - roots[-1]) > 1e-5 * max(1.0, abs(root)):
                roots.append(root)

    def c_nd(y: float) -> float:
        c2 = zhestkaya_c2_nd(y, beta=beta, tau=tau)
        return math.sqrt(c2) if c2 > 0.0 else float("nan")

    extrema = []
    for y0 in roots:
        eps = 1e-3 * max(1.0, y0)
        yL = max(y_lo, y0 - eps)
        yR = min(y_hi, y0 + eps)
        cL = c_nd(yL)
        c0v = c_nd(y0)
        cR = c_nd(yR)
        if not (math.isfinite(cL) and math.isfinite(c0v) and math.isfinite(cR)):
            continue
        if c0v <= cL and c0v <= cR:
            extrema.append((y0, c0v, "min"))
        elif c0v >= cL and c0v >= cR:
            extrema.append((y0, c0v, "max"))

    extrema.sort(key=lambda t: t[0])

    y_cmin = c_min = None
    y_cmax = c_max = None
    for (yy, cc, kind) in extrema:
        if kind == "min" and c_min is None:
            y_cmin, c_min = yy, cc
        if kind == "max":
            y_cmax, c_max = yy, cc

    if c_min is None or c_max is None:
        c_samples: List[Tuple[float, float]] = [(0.0, c0)]
        for yy in ys:
            c2v = zhestkaya_c2_nd(yy, beta=beta, tau=tau)
            if c2v > 0.0 and math.isfinite(c2v):
                c_samples.append((yy, math.sqrt(c2v)))
        if c_samples:
            y_glob_min, c_glob_min = min(c_samples, key=lambda t: t[1])
            y_glob_max, c_glob_max = max(c_samples, key=lambda t: t[1])
            if c_min is None:
                c_min = c_glob_min
                y_cmin = None if y_glob_min == 0.0 else y_glob_min
            if c_max is None:
                c_max = c_glob_max
                y_cmax = None if y_glob_max == 0.0 else y_glob_max

    return ZhestkayaCritical(
        beta=beta,
        tau=tau,
        c0=c0,
        c_min=c_min,
        c_max=c_max,
        y_cmin=y_cmin,
        y_cmax=y_cmax,
        y_cutoff=y_cut,
    )


def solve_zhestkaya_phase_speed_roots(
    *,
    F: float,
    beta: float,
    tau: float,
    n_scan: int = 8000,
    y_scan_max: float = 80.0,
) -> List[float]:
    if F <= 0.0:
        raise ValueError("F must be positive.")
    crit = find_zhestkaya_critical_speeds(beta=beta, tau=tau)
    y_hi = y_scan_max
    if crit.y_cutoff is not None:
        y_hi = min(y_hi, 0.999 * crit.y_cutoff)

    def h(y: float) -> float:
        c2 = zhestkaya_c2_nd(y, beta=beta, tau=tau)
        if c2 <= 0.0:
            return -F
        return math.sqrt(c2) - F

    y_lo = 1e-6
    ys = [y_lo + (y_hi - y_lo) * (i / (n_scan - 1)) for i in range(n_scan)]
    hs = [h(y) for y in ys]

    roots: List[float] = []
    for i in range(len(ys) - 1):
        a, b = ys[i], ys[i + 1]
        fa, fb = hs[i], hs[i + 1]
        if fa == 0.0:
            roots.append(a)
            continue
        if fa * fb < 0.0:
            root = _bisect_root(h, a, b)
            if not roots or abs(root - roots[-1]) > 1e-5 * max(1.0, abs(root)):
                roots.append(root)
    return roots


def classify_zhestkaya_finite_depth_regime(
    F: float,
    aleph: float,
    tau: float,
    *,
    rtol: float = 1e-6,
    return_roots: bool = True,
) -> Dict[str, float | str | List[float] | None]:
    if F <= 0.0 or aleph <= 0.0 or tau < 0.0:
        raise ValueError("F and aleph must be positive; tau must be nonnegative.")

    beta = aleph * (F ** 2)  # D/(rho g H^4)
    crit = find_zhestkaya_critical_speeds(beta=beta, tau=tau)

    c0 = crit.c0
    cmin = crit.c_min if crit.c_min is not None else float("nan")
    cmax = crit.c_max if crit.c_max is not None else c0

    cmin_eff = crit.c_min if crit.c_min is not None else c0
    cmax_eff = crit.c_max if crit.c_max is not None else c0

    def le(x: float, y: float) -> bool:
        return x <= y * (1.0 + rtol)

    def ge(x: float, y: float) -> bool:
        return x >= y * (1.0 - rtol)

    if crit.c_min is not None and le(F, cmin):
        regime = "U < c_min"
        code = 1
    elif crit.c_min is not None and le(cmin, F) and le(F, c0):
        regime = "c_min < U < c0"
        code = 2
    elif ge(F, c0) and le(F, cmax):
        regime = "c0 < U < c_max"
        code = 3
    elif ge(F, cmax):
        regime = "U > c_max"
        code = 4
    else:
        if le(F, c0):
            regime = "U < c0 (no interior c_min found)"
            code = 10
        else:
            regime = "U > c0 (no interior c_min found)"
            code = 11

    roots_y: Optional[List[float]] = None
    if return_roots:
        roots_y = solve_zhestkaya_phase_speed_roots(F=F, beta=beta, tau=tau)

    return {
        "regime": regime,
        "code": code,
        "F": F,
        "aleph": aleph,
        "tau": tau,
        "beta": beta,
        "c0": c0,
        "c_min": crit.c_min,
        "c_max": crit.c_max,
        "c_min_eff": cmin_eff,
        "c_max_eff": cmax_eff,
        "y_cmin": crit.y_cmin,
        "y_cmax": crit.y_cmax,
        "y_cutoff": crit.y_cutoff,
        "roots_y": roots_y,
        "n_roots": None if roots_y is None else len(roots_y),
    }


def demo_zhestkaya_regime_classification(
    *,
    H: float = 6.8,
    g: float = 9.81,
    tau_f: float = 0.1,
    cmin_dim: float = 5.2895,
    cmax_dim: float = 10.2633,
    U_values: Iterable[float] = (4.0, 6.0, 7.4, 8.4, 10.2, 10.5, 12.0),
    use_tau_f_directly: bool = False,
    rtol: float = 1e-6,
) -> Dict[str, object]:
    c0_dim = math.sqrt(g * H)
    target_cmin = cmin_dim / c0_dim
    target_cmax = cmax_dim / c0_dim

    def critical_speeds_from_beta_tau(beta: float, tau: float) -> Tuple[float, float, float]:
        out = classify_zhestkaya_finite_depth_regime(F=1.0, aleph=beta, tau=tau, return_roots=False, rtol=rtol)
        return float(out["c_min_eff"]), float(out["c_max_eff"]), float(out["c0"])

    def fit_beta_for_cmin(*, tau: float, beta_lo: float = 1e-6, beta_hi: float = 10.0) -> float:
        cmin_lo, _, _ = critical_speeds_from_beta_tau(beta_lo, tau)
        cmin_hi, _, _ = critical_speeds_from_beta_tau(beta_hi, tau)
        if not (cmin_lo < target_cmin < cmin_hi):
            raise RuntimeError(
                f"Could not bracket target c_min={target_cmin:.6g} with "
                f"beta_lo={beta_lo} (cmin={cmin_lo:.6g}) and beta_hi={beta_hi} (cmin={cmin_hi:.6g})."
            )
        lo, hi = beta_lo, beta_hi
        for _ in range(70):
            mid = math.sqrt(lo * hi)
            cmin_mid, _, _ = critical_speeds_from_beta_tau(mid, tau)
            if cmin_mid < target_cmin:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def fit_tau_and_beta() -> Tuple[float, float]:
        if use_tau_f_directly:
            beta = fit_beta_for_cmin(tau=tau_f)
            return tau_f, beta

        tau_lo, tau_hi = 0.02, 0.2

        def cmax_after_beta_fit(tau: float) -> Tuple[float, float]:
            beta = fit_beta_for_cmin(tau=tau)
            _, cmax, _ = critical_speeds_from_beta_tau(beta, tau)
            return cmax, beta

        cmax_lo, _ = cmax_after_beta_fit(tau_lo)
        cmax_hi, _ = cmax_after_beta_fit(tau_hi)
        if not (cmax_lo > target_cmax > cmax_hi):
            raise RuntimeError(
                "Could not bracket target c_max. Try widening tau_lo/tau_hi.\n"
                f"At tau_lo={tau_lo}: cmax={cmax_lo:.6g}, at tau_hi={tau_hi}: cmax={cmax_hi:.6g}, "
                f"target={target_cmax:.6g}."
            )

        lo, hi = tau_lo, tau_hi
        beta_mid = None
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            cmax_mid, beta_mid = cmax_after_beta_fit(mid)
            if cmax_mid > target_cmax:
                lo = mid
            else:
                hi = mid

        tau = 0.5 * (lo + hi)
        beta = fit_beta_for_cmin(tau=tau) if beta_mid is None else beta_mid
        return tau, beta

    tau, beta = fit_tau_and_beta()

    rows: List[Dict[str, object]] = []
    for U in U_values:
        F = U / c0_dim
        aleph = beta / (F * F)
        out = classify_zhestkaya_finite_depth_regime(F=F, aleph=aleph, tau=tau, return_roots=False, rtol=rtol)
        rows.append({"U": U, "F": F, "aleph": aleph, "regime": out["regime"]})

    cmin_nd, cmax_nd, _c0_nd = critical_speeds_from_beta_tau(beta, tau)
    return {
        "H": H,
        "g": g,
        "c0_dim": c0_dim,
        "tau_used": tau,
        "beta_used": beta,
        "cmin_dim_model": cmin_nd * c0_dim,
        "cmax_dim_model": cmax_nd * c0_dim,
        "rows": rows,
    }
    
    
# import math

def print_optionA_representative_F_table(beta: float = 0.1, *, margin: float = 0.2) -> None:
    """
    Option A summary (fixed dispersion curve, slide U):
      - Finite depth: fix beta = D/(rho g H^4). Then aleph_fd(F) = beta/F^2, and regimes are
            F < F*        : U < c_min
            F* < F < 1    : c_min < U < c0
            F > 1         : U > c0
      - Infinite depth (as in this file): classifier uses lambda = aleph_inf * F^3 with lambda* = 27/256.
        To "keep beta fixed" in the same spirit, we take aleph_inf := beta (constant).

    margin controls how far the representative F values sit from the regime boundaries.
      - below-boundary reps use (1-margin)
      - above-boundary reps use (1+margin)
      - supercritical wrt c0 uses F = 1+margin
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if not (0.0 < margin < 0.9):
        raise ValueError("margin should be in (0, 0.9).")

    # -----------------------------
    # Finite depth: compute F*(beta)
    # -----------------------------
    def sech2(y: float) -> float:
        c = math.cosh(y)
        return 1.0 / (c * c)

    def g(y: float) -> float:
        # g(y)=0 gives y_star; matches classify_finite_depth_regime() but with beta fixed.
        return 1.0 - 3.0 * beta * (y ** 4) - (y + beta * (y ** 5)) * sech2(y)

    def bisect_root(f, a: float, b: float, max_iter: int = 400) -> float:
        fa, fb = f(a), f(b)
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            raise RuntimeError("Bisection needs a sign change on [a,b].")
        lo, hi = a, b
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if abs(fm) <= 1e-14 or (hi - lo) <= 1e-12 * max(1.0, abs(mid)):
                return mid
            if fa * fm > 0.0:
                lo, fa = mid, fm
            else:
                hi, fb = mid, fm
        return 0.5 * (lo + hi)

    a = 1e-10
    fa = g(a)
    b = max(10.0, 1.5 * (1.0 / (3.0 * beta)) ** 0.25)
    fb = g(b)
    grow = 0
    while fa * fb > 0.0 and grow < 60:
        b *= 1.5
        fb = g(b)
        grow += 1
    if fa * fb > 0.0:
        raise RuntimeError("Failed to bracket the finite-depth critical root y*>0.")

    y_star = bisect_root(g, a, b)
    t = math.tanh(y_star)
    F_star = math.sqrt(t / y_star + beta * (y_star ** 3) * t)  # c_min / sqrt(gH)

    # Representative F's in each finite-depth regime (Option A: same beta curve)
    F_fd_1 = (1.0 - margin) * F_star             # U < c_min
    F_fd_2 = 0.5 * (F_star + 1.0)                # c_min < U < c0
    F_fd_3 = 1.0 + margin                        # U > c0

    # Corresponding finite-depth aleph values (since beta = aleph * F^2)
    def aleph_fd(F: float) -> float:
        return beta / (F * F)

    # -----------------------------
    # Infinite depth: two regimes from lambda = aleph * F^3 (lambda* = 27/256)
    # -----------------------------
    lam_star = 27.0 / 256.0
    aleph_inf = beta  # "keep beta fixed" -> treat infinite-depth aleph as this fixed coefficient
    Fcrit_inf = (lam_star / aleph_inf) ** (1.0 / 3.0)

    F_inf_1 = (1.0 - margin) * Fcrit_inf         # c_min < U
    F_inf_2 = (1.0 + margin) * Fcrit_inf         # U < c_min

    # -----------------------------
    # Print table
    # -----------------------------
    def row(depth: str, regime: str, F: float, aleph: float) -> str:
        return f"{depth:<13}  {regime:<16}  {F:>10.6f}  {aleph:>12.6f}"

    print(f"\nOption A representative speeds (fixed beta = {beta:g})")
    print("-" * 62)
    print(f"{'case':<13}  {'regime':<16}  {'F':>10}  {'aleph':>12}")
    print("-" * 62)

    print(row("finite depth", "U < c_min",        F_fd_1, aleph_fd(F_fd_1)))
    print(row("finite depth", "c_min < U < c0",   F_fd_2, aleph_fd(F_fd_2)))
    print(row("finite depth", "U > c0",           F_fd_3, aleph_fd(F_fd_3)))

    print(row("infinite depth", "c_min < U",      F_inf_1, aleph_inf))
    print(row("infinite depth", "U < c_min",      F_inf_2, aleph_inf))

    print("-" * 62)
    print(f"[finite depth]  y* = {y_star:.6f},  F* = c_min/sqrt(gH) = {F_star:.6f}")
    print(f"[infinite depth] Fcrit from lambda*=27/256:  Fcrit = {Fcrit_inf:.6f}  (with aleph_inf=beta)")
    print()

def main() -> None:
    print("Infinite-depth regimes:")
    r = classify_infinite_depth_regime(F=0.7, aleph=0.5)
    print(r["regime"], "U/c_min =", r["U_over_cmin"])
    r = classify_infinite_depth_regime(F=0.35, aleph=0.5)
    print(r["regime"], "U/c_min =", r["U_over_cmin"])

    print("Finite-depth regimes:")
    regime_ = classify_finite_depth_regime(F=0.3, aleph=0.5)
    print(regime_["regime"], "F*=", regime_["F_star"])
    regime_ = classify_finite_depth_regime(F=0.8, aleph=0.015)
    print(regime_["regime"], "F*=", regime_["F_star"])
    regime_ = classify_finite_depth_regime(F=2., aleph=0.1)
    print(regime_["regime"], "F*=", regime_["F_star"])

    print("Zhestkaya (viscoelastic) finite-depth regimes:")
    info = demo_zhestkaya_regime_classification()
    print(f"c0 = {info['c0_dim']:.4f} m/s")
    print(f"tau used = {info['tau_used']:.6g}, beta used = {info['beta_used']:.6g}")
    print(f"model cmin = {info['cmin_dim_model']:.4f} m/s, model cmax = {info['cmax_dim_model']:.4f} m/s")
    print()
    for row in info["rows"]:
        print(f"U={row['U']:>4.1f}  F={row['F']:.4f}  -> {row['regime']}")
        
if __name__ == "__main__":
    # main()
    print_optionA_representative_F_table(beta=0.1)