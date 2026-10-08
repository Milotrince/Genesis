"""Per-environment gravity shared with solvers and data consumers."""

from dataclasses import dataclass

import quadrants as qd

from . import SolverData


@dataclass(frozen=True, kw_only=True, eq=False)
class GravityData(SolverData):
    gravity: qd.Tensor
