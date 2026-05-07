"""Steady hydroelastic solver components used by WavesLab.

The public examples mainly use :mod:`waveslab.steady.cover_runner`.
Lower-level modules are kept here so the old custom ``pywave`` package is no
longer the primary dependency.
"""

from .driver import make_grid, run_case_deep

__all__ = ["make_grid", "run_case_deep"]
