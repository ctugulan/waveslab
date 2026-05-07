# pywave/waves_helpers/names.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import os
from typing import Any, Literal, TypeAlias


# =============================================================================
# Types
# =============================================================================
System: TypeAlias = Literal["deep", "flatbed", "bathy"]
ProblemKind: TypeAlias = Literal["deep", "flatbed", "bathy3"]


# =============================================================================
# Shared encoding helper
# =============================================================================
def _enc_i(x: float, scale: int, width: int) -> str:
    """Encode a float as a zero-padded integer with given scale."""
    return f"{int(round(float(x) * scale)):0{width}d}"


# =============================================================================
# ---------------------------------------------------------------------
# A) run_cases-style outputs (caller-dir aware)
#   - used by run_cases/run_all helpers
#   - layout: <out_base>/out/results/<system> and <out_base>/out/fig/<system>
# ---------------------------------------------------------------------
# =============================================================================

def _find_repo_root(start: Path) -> Path | None:
    """Walk upward looking for markers of the repo root."""
    p = start.resolve()
    for cur in (p,) + tuple(p.parents):
        if (cur / "pyproject.toml").is_file():
            return cur
        if (cur / ".git").exists():
            return cur
    return None


def _infer_caller_dir() -> Path | None:
    """
    Return the directory containing the *user* file that called into this package.
    Skips frames that appear to be inside this package directory
    """
    pkg_dir = Path(__file__).resolve().parent

    # Start a bit deeper so helper wrappers don't capture themselves.
    for fr in inspect.stack()[3:]:
        fn = fr.filename
        if not fn or fn.startswith("<"):
            continue
        try:
            p = Path(fn).resolve()
        except Exception:
            continue

        # Skip anything inside the package
        if pkg_dir in p.parents:
            continue

        s = str(p).replace("\\", "/")
        if ("/site-packages/" in s) or ("/dist-packages/" in s):
            continue
        if "/lib/python" in s:
            continue

        return p.parent

    return None


def resolve_out_base(out_base: str | Path | None) -> Path:
    if out_base is not None:
        return Path(out_base).expanduser().resolve()

    env = os.environ.get("PYWAVE_OUT_BASE", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    caller = _infer_caller_dir()
    if caller is not None:
        return caller.resolve()

    repo = _find_repo_root(Path.cwd()) or _find_repo_root(Path(__file__).resolve())
    if repo is not None:
        return repo.resolve()

    return Path.cwd().resolve()


def out_paths(*, system: System, out_base: str | Path | None = None) -> tuple[Path, Path]:
    """
    Matches run_cases.py behavior:

      results_dir = <out_base>/out/results/<system>
      fig_dir     = <out_base>/out/fig/<system>
    """
    base = resolve_out_base(out_base)
    out = base / "out"
    rdir = out / "results" / str(system)
    fdir = out / "fig" / str(system)
    rdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    return rdir, fdir


def enc_step(x: float) -> str:
    """
    Step token used in filenames: supports 0.2->"02", 0.15->"015", 0.05->"005", etc.
    """
    xf = float(x)
    for scale, width in ((10, 2), (100, 3), (1000, 4), (10000, 5)):
        v = xf * scale
        if abs(v - round(v)) < 1e-10:
            return f"{int(round(v)):0{width}d}"
    return f"{xf:g}".replace(".", "p").replace("-", "m")


def enc_x0(x0: float) -> str:
    """Encode x0 (usually integer in your grids)."""
    x0f = float(x0)
    if abs(x0f - round(x0f)) < 1e-9:
        return str(int(round(x0f)))
    return f"{x0f:g}".replace(".", "p").replace("-", "m")


def dense_filename(*, M: int, N: int, dx: float, dy: float, x0: float) -> str:
    """run_cases mesh token."""
    return f"n{int(N)}m{int(M)}dx{enc_step(dx)}dy{enc_step(dy)}x0{enc_x0(x0)}"


def param_filename2(*, Fr: float, aleph: float, epsilon: float, mu: float) -> str:
    """run_cases physical token."""
    f = _enc_i(Fr, 10, 2)
    b = _enc_i(aleph, 1000, 4)
    e = _enc_i(epsilon, 10, 2)
    m = _enc_i(mu, 100, 3)
    return f"f{f}ice{b}eps{e}mu{m}"


def param_filename3(*, Fr: float, aleph: float, epsilon: float, mu: float, tauf: float | None = None) -> str:
    """run_cases physical token with tau (tau encoded as *100, width=3)."""
    base = param_filename2(Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu)
    if tauf is None:
        return base
    tau = _enc_i(tauf, 100, 3)
    return f"{base}tau{tau}"


def zeta_case_tag(
    *,
    system: System,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    tauf: float,
    use_radiation: bool,
    upstream_bc: Any,
    mode: Any,
) -> str:
    """
    Drop-in replacement for run_cases.py::_case_tag (keeps the same overall shape).
    """
    mesh = dense_filename(M=M, N=N, dx=dx, dy=dy, x0=x0)
    phys = param_filename3(Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu, tauf=tauf)
    rc = "rad" if bool(use_radiation) else "simple"
    tail = f"{rc}_{upstream_bc}_{mode}"
    return f"{system}_zeta_{mesh}_{phys}_{tail}"

#-------------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpectedZetaPaths:
    # naming
    tag: str
    stem: str

    # directories
    results_dir: Path
    fig_dir: Path

    # concrete outputs
    csv_path: Path
    fig_prefix: Path
    fig_path: Path

    def have_csv(self) -> bool:
        return self.csv_path.is_file()

    def have_fig(self) -> bool:
        return self.fig_path.is_file()

    def have_all(self) -> bool:
        return self.have_csv() and self.have_fig()


def expected_zeta_paths(
    *,
    system: System,
    out_base: str | Path | None = None,
    mkdir: bool = True,
    N: int,
    M: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    tauf: float,
    use_radiation: bool,
    upstream_bc: Any,
    mode: Any,
    fig_kind: str = "surface",
) -> ExpectedZetaPaths:
    """
    Single source of truth for expected run_cases outputs:
      CSV:  <out_base>/out/results/<system>/zeta_<tag>.csv
      FIG:  <out_base>/out/fig/<system>/zeta_<tag>_<fig_kind>.png
    """
    if mkdir:
        results_dir, fig_dir = out_paths(system=system, out_base=out_base)
    else:
        base = resolve_out_base(out_base)
        results_dir = base / "out" / "results" / str(system)
        fig_dir = base / "out" / "fig" / str(system)

    tag = zeta_case_tag(
        system=system,
        M=M, N=N, dx=dx, dy=dy, x0=x0,
        Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu, tauf=tauf,
        use_radiation=use_radiation,
        upstream_bc=upstream_bc,
        mode=mode,
    )
    stem = f"zeta_{tag}"
    csv_path = results_dir / f"{stem}.csv"
    fig_prefix = fig_dir / f"{stem}_{fig_kind}"
    fig_path = fig_prefix.with_suffix(".png")

    return ExpectedZetaPaths(
        tag=tag,
        stem=stem,
        results_dir=results_dir,
        fig_dir=fig_dir,
        csv_path=csv_path,
        fig_prefix=fig_prefix,
        fig_path=fig_path,
    )


# =============================================================================
# ---------------------------------------------------------------------
# B) steady_clean sweep-layout helpers (ported from names_old.py)
#   - used by sweep_flatbed_speed_tau.py and bathy3 sweep tooling
#   - layout: pythonwaves/steady_clean/out/...
# ---------------------------------------------------------------------
# =============================================================================

def enc_01(x: float) -> str:
    """Encode a float as two digits of tenths (legacy steady_clean convention)."""
    return f"{abs(int(round(float(x) * 10))):02d}"


def enc_x0_sc(x0: float) -> str:
    """Encode x0 as a rounded integer string (keeps the sign)."""
    return f"{int(round(float(x0)))}"


def dense_filename_sc(M: int, N: int, dx: float, dy: float, x0: float) -> str:
    """steady_clean mesh token (legacy dx/dy tenths encoding)."""
    x0_enc = enc_x0_sc(x0)
    return f"n{N}m{M}dx{enc_01(dx)}dy{enc_01(dy)}x0{x0_enc}"


def param_filename2_sc(Fr: float, aleph: float, epsilon: float, mu: float) -> str:
    """steady_clean physical token (legacy 'b' instead of 'ice')."""
    f = _enc_i(Fr, 10, 2)
    b = _enc_i(aleph, 1000, 4)
    e = _enc_i(epsilon, 10, 2)
    m = _enc_i(mu, 100, 3)
    return f"f{f}b{b}eps{e}mu{m}"


def param_filename3_sc(Fr: float, aleph: float, epsilon: float, mu: float, *, tauf: float | None = None) -> str:
    """
    steady_clean physical token with tau (legacy: tau encoded as *1000, width=4).
    """
    base = param_filename2_sc(Fr, aleph, epsilon, mu)
    if tauf is None or abs(float(tauf)) == 0.0:
        return base
    tau = _enc_i(tauf, 1000, 4)
    return f"{base}tau{tau}"


def zeta_output_prefix(
    *,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    rc_tag: str,
    tauf: float | None = None,
) -> str:
    """
    Prefix used by steady_clean sweeps and bathy3 tooling.

    NOTE: This intentionally preserves the legacy conventions so that
    existing sweep outputs are discovered and re-used.
    """
    tag = dense_filename_sc(M, N, dx, dy, x0)
    pname = f"{param_filename3_sc(Fr, aleph, epsilon, mu, tauf=tauf)}_{rc_tag}"
    return f"zeta_{tag}_{pname}"


@dataclass(frozen=True)
class ExpectedSweepCasePaths:
    """
    Expected output paths for *steady_clean sweep scripts*.

    Convention:
      - figures are written under:   layout.fig_dir / <out_prefix>*
      - CSV results are written to:  (layout.res_dir / <out_prefix>).csv

    Notes
    -----
    `fig_prefix` is intentionally a *prefix* (no suffix) because different sweep
    scripts may append different tags/suffixes when saving figures.
    """

    out_prefix: str
    fig_prefix: Path
    res_prefix: Path
    csv_path: Path

    def have_csv(self) -> bool:
        return self.csv_path.is_file()


def expected_sweep_case_paths(
    *,
    layout: "OutputLayout",
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    rc_tag: str,
    tauf: float | None = None,
    csv_suffix: str = ".csv",
) -> ExpectedSweepCasePaths:
    """
    One-call helper used by sweep scripts to standardize expected output paths.

    This bundles the legacy `zeta_output_prefix(...)` naming convention with the
    sweep layouts returned by `make_tau_sweeps_layout(...)`, `make_u_sweeps_layout(...)`,
    and related helpers.
    """
    out_prefix = zeta_output_prefix(
        M=int(M),
        N=int(N),
        dx=float(dx),
        dy=float(dy),
        x0=float(x0),
        Fr=float(Fr),
        aleph=float(aleph),
        epsilon=float(epsilon),
        mu=float(mu),
        tauf=None if tauf is None else float(tauf),
        rc_tag=str(rc_tag),
    )

    fig_prefix = Path(layout.fig_dir) / out_prefix
    res_prefix = Path(layout.res_dir) / out_prefix
    csv_path = res_prefix.with_suffix(str(csv_suffix))

    return ExpectedSweepCasePaths(
        out_prefix=out_prefix,
        fig_prefix=fig_prefix,
        res_prefix=res_prefix,
        csv_path=csv_path,
    )

@dataclass(frozen=True)
class OutputLayout:
    solver_dir: Path
    fig_dir: Path
    res_dir: Path

    @staticmethod
    def next_to(file_: Path) -> "OutputLayout":
        sdir = file_.resolve().parent
        return OutputLayout(solver_dir=sdir, fig_dir=sdir / "figures", res_dir=sdir / "results")

    @staticmethod
    def next_to_kind(file_: Path, *, kind: ProblemKind) -> "OutputLayout":
        sdir = file_.resolve().parent
        out_dir = sdir / "out"
        return OutputLayout(solver_dir=sdir, fig_dir=out_dir / f"figures_{kind}", res_dir=out_dir / f"results_{kind}")


def make_zeta_output_layout(
    file_: Path,
    *,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    rc_tag: str,
    tauf: float | None = None,
    kind: ProblemKind | None = None,
) -> tuple[OutputLayout, str, Path, Path]:
    """
    Script-local output layout helper (legacy), used in some older scripts.
    """
    layout = OutputLayout.next_to(file_) if kind is None else OutputLayout.next_to_kind(file_, kind=kind)
    layout.fig_dir.mkdir(parents=True, exist_ok=True)
    layout.res_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = zeta_output_prefix(
        M=M, N=N, dx=dx, dy=dy, x0=x0,
        Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu,
        tauf=tauf, rc_tag=rc_tag,
    )
    print(f"[output]: {out_prefix}")
    return layout, out_prefix, layout.fig_dir / out_prefix, layout.res_dir / out_prefix


def _find_pythonwaves_root(file_: Path) -> Path:
    """Best-effort locator for the `pythonwaves/` directory."""
    p = file_.resolve()

    for parent in [p] + list(p.parents):
        if parent.name == "pythonwaves":
            return parent

    for parent in p.parents:
        cand = parent / "pythonwaves"
        if cand.is_dir():
            return cand

    # Fallback (still lets callers build paths relative to the script location).
    return p.parent


def physics_dirname(*, aleph: float, epsilon: float, mu: float) -> str:
    """
    Directory tag for fixed physical parameters in U-sweeps.

    Uses the legacy steady_clean directory convention:
        ice{b}eps{e}mu{m}
    """
    b = _enc_i(aleph, 1000, 4)
    e = _enc_i(epsilon, 10, 2)
    m = _enc_i(mu, 10, 2)
    return f"ice{b}eps{e}mu{m}"


def u_sweeps_dir(
    file_: Path,
    *,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    aleph: float,
    epsilon: float,
    mu: float,
) -> Path:
    """Output directory for steady_clean flatbed U-sweeps."""
    pw = _find_pythonwaves_root(file_)
    mesh = dense_filename_sc(M, N, dx, dy, x0)
    physics = physics_dirname(aleph=aleph, epsilon=epsilon, mu=mu)
    return pw / "steady_clean" / "out" / "flatbed" / f"u_sweeps_{mesh}_{physics}"


def make_u_sweeps_layout(
    file_: Path,
    *,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    aleph: float,
    epsilon: float,
    mu: float,
) -> OutputLayout:
    """
    Layout that matches:

        pythonwaves/steady_clean/out/flatbed/u_sweeps_{mesh}_{physics}/

    with CSVs in `.../results/` and figures in the sweep directory itself.
    """
    pw = _find_pythonwaves_root(file_)
    sweep_dir = u_sweeps_dir(
        file_, M=M, N=N, dx=dx, dy=dy, x0=x0, aleph=aleph, epsilon=epsilon, mu=mu
    )
    fig_dir = sweep_dir
    res_dir = sweep_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return OutputLayout(solver_dir=pw / "steady_clean", fig_dir=fig_dir, res_dir=res_dir)


def physics_dirname_with_fr(*, Fr: float, aleph: float, epsilon: float, mu: float) -> str:
    """
    Directory tag for fixed physical parameters in tau-sweeps.

    Legacy convention:
        f{Fr}ice{b}eps{e}mu{m}
    """
    f = _enc_i(Fr, 10, 2)
    b = _enc_i(aleph, 1000, 4)
    e = _enc_i(epsilon, 10, 2)
    m = _enc_i(mu, 10, 2)
    return f"f{f}ice{b}eps{e}mu{m}"


def tau_sweeps_dir(
    file_: Path,
    *,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
) -> Path:
    """Output directory for steady_clean flatbed tau-sweeps."""
    pw = _find_pythonwaves_root(file_)
    mesh = dense_filename_sc(M, N, dx, dy, x0)
    physics = physics_dirname_with_fr(Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu)
    return pw / "steady_clean" / "out" / "flatbed" / f"tau_sweeps_{mesh}_{physics}"


def make_tau_sweeps_layout(
    file_: Path,
    *,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
) -> OutputLayout:
    """
    Layout that matches:

        pythonwaves/steady_clean/out/flatbed/tau_sweeps_{mesh}_{physics}/

    with CSVs in `.../results/` and figures in the sweep directory itself.
    """
    pw = _find_pythonwaves_root(file_)
    sweep_dir = tau_sweeps_dir(
        file_, M=M, N=N, dx=dx, dy=dy, x0=x0, Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu
    )
    fig_dir = sweep_dir
    res_dir = sweep_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return OutputLayout(solver_dir=pw / "steady_clean", fig_dir=fig_dir, res_dir=res_dir)


# ---------------------------
# Bathy3 helpers (steady_clean)
# ---------------------------

def bathy3_runs_dir(
    file_: Path,
    *,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    tauf: float | None = None,
) -> Path:
    """
    Output directory for steady_clean bathy3 runs.

    Matches:
        pythonwaves/steady_clean/out/bathy3/bathy3_{mesh}_{physics}/
    where `physics` uses param_filename3_sc encoding.
    """
    pw = _find_pythonwaves_root(file_)
    mesh = dense_filename_sc(M, N, dx, dy, x0)
    physics = param_filename3_sc(Fr, aleph, epsilon, mu, tauf=tauf)
    return pw / "steady_clean" / "out" / "bathy3" / f"bathy3_{mesh}_{physics}"


def make_bathy3_layout(
    file_: Path,
    *,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    rc_tag: str,
    tauf: float | None = None,
) -> tuple[OutputLayout, str, Path, Path]:
    """
    Like make_zeta_output_layout, but writes into pythonwaves/steady_clean/out/bathy3/.
    """
    run_dir = bathy3_runs_dir(
        file_, M=M, N=N, dx=dx, dy=dy, x0=x0,
        Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu, tauf=tauf
    )
    fig_dir = run_dir / "figures"
    res_dir = run_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    pw = _find_pythonwaves_root(file_)
    layout = OutputLayout(solver_dir=pw / "steady_clean", fig_dir=fig_dir, res_dir=res_dir)

    out_prefix = zeta_output_prefix(
        M=M, N=N, dx=dx, dy=dy, x0=x0,
        Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu,
        tauf=tauf, rc_tag=rc_tag
    )
    print(f"[output]: {out_prefix}")
    return layout, out_prefix, layout.fig_dir / out_prefix, layout.res_dir / out_prefix


def bathy3_mesh_sweeps_dir(file_: Path) -> Path:
    """
    Centralized output directory for bathy3 mesh sweeps.

    Matches:
        pythonwaves/steady_clean/out/bathy3/mesh_sweeps/
    with subfolders:
        - figures/
        - results/
    """
    pw = _find_pythonwaves_root(file_)
    return pw / "steady_clean" / "out" / "bathy3" / "mesh_sweeps"


def make_bathy3_mesh_sweeps_layout(
    file_: Path,
    *,
    M: int,
    N: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    rc_tag: str,
    tauf: float | None = None,
) -> tuple[OutputLayout, str, Path, Path]:
    """
    Like make_bathy3_layout, but routes *all* cases into one sweep folder.

    Figures go in:
        pythonwaves/steady_clean/out/bathy3/mesh_sweeps/figures/
    CSV results go in:
        pythonwaves/steady_clean/out/bathy3/mesh_sweeps/results/
    """
    sweep_dir = bathy3_mesh_sweeps_dir(file_)
    fig_dir = sweep_dir / "figures"
    res_dir = sweep_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    pw = _find_pythonwaves_root(file_)
    layout = OutputLayout(solver_dir=pw / "steady_clean", fig_dir=fig_dir, res_dir=res_dir)

    out_prefix = zeta_output_prefix(
        M=M, N=N, dx=dx, dy=dy, x0=x0,
        Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu,
        tauf=tauf, rc_tag=rc_tag
    )
    print(f"[output]: {out_prefix}")
    return layout, out_prefix, layout.fig_dir / out_prefix, layout.res_dir / out_prefix

# =============================================================================
# ---------------------------------------------------------------------
# C) Flexural cover workflow helpers
#   - used by pythonwaves/FlexuralCovers/run_flexural_cover_case.py
#   - centralizes run tags so mesh/physics tokens follow SolverConfig
# ---------------------------------------------------------------------
# =============================================================================

def param_filename_flexural_cover(*, Fr: float, aleph: float, epsilon: float, mu: float) -> str:
    """
    Physical token used by the FlexuralCovers direct-mask workflow.

    This intentionally preserves the existing cover-run filename convention:
        f07b0100eps01mu01
    rather than the older steady_clean helper's three-digit ``mu010`` token.
    """
    f = _enc_i(Fr, 10, 2)
    b = _enc_i(aleph, 1000, 4)
    e = _enc_i(epsilon, 10, 2)
    m = _enc_i(mu, 10, 2)
    return f"f{f}b{b}eps{e}mu{m}"


def flexural_cover_rc_tag(
    *,
    use_radiation: bool,
    rigidity_tag: str = "F1",
    cover_source: str = "direct",
) -> str:
    """Return the trailing solver/cover token, e.g. ``rad_F1_direct``."""
    rc = "rad" if bool(use_radiation) else "simple"
    return f"{rc}_{str(rigidity_tag)}_{str(cover_source)}"


def flexural_cover_base_tag(
    *,
    N: int,
    M: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    use_radiation: bool,
    rigidity_tag: str = "F1",
    cover_source: str = "direct",
) -> str:
    """
    Mesh/physics-aware base tag for FlexuralCovers runs.

    The tag is rebuilt from the active configuration, so changing ``N``, ``M``,
    ``dx``, ``dy``, ``x0``, or physical parameters automatically updates both
    the run directory and saved figure names.
    """
    mesh = dense_filename_sc(int(M), int(N), float(dx), float(dy), float(x0))
    phys = param_filename_flexural_cover(Fr=Fr, aleph=aleph, epsilon=epsilon, mu=mu)
    rc_tag = flexural_cover_rc_tag(
        use_radiation=use_radiation,
        rigidity_tag=rigidity_tag,
        cover_source=cover_source,
    )
    return f"zeta_{mesh}_{phys}_{rc_tag}"


def flexural_cover_run_tag(
    *,
    N: int,
    M: int,
    dx: float,
    dy: float,
    x0: float,
    Fr: float,
    aleph: float,
    epsilon: float,
    mu: float,
    use_radiation: bool,
    full_domain: bool,
    cover_case_label: str,
    rigidity_tag: str = "F1",
    cover_source: str = "direct",
) -> str:
    """Complete run tag including domain and selected-cover suffixes."""
    tag = flexural_cover_base_tag(
        N=N,
        M=M,
        dx=dx,
        dy=dy,
        x0=x0,
        Fr=Fr,
        aleph=aleph,
        epsilon=epsilon,
        mu=mu,
        use_radiation=use_radiation,
        rigidity_tag=rigidity_tag,
        cover_source=cover_source,
    )
    if bool(full_domain):
        tag = f"{tag}__full_y"
    if str(cover_case_label):
        tag = f"{tag}__{cover_case_label}"
    return tag


def flexural_cover_surface_path(
    run_dir: str | Path,
    *,
    run_tag: str,
    domain_mode: str = "full",
) -> Path:
    """Default filename for the single final surface-cover plot."""
    mode = str(domain_mode).strip().lower().replace("-", "_")
    suffix = "full_domain" if mode in {"full", "full_y", "full_domain", "native", "saved"} else "mirror_y0"
    return Path(run_dir) / f"{run_tag}.publication_surface_cover__{suffix}.png"