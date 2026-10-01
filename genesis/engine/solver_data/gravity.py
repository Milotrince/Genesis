"""Per-environment gravity shared with solvers and data consumers."""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs

from . import SolverData


@dataclass(frozen=True, kw_only=True, eq=False)
class GravityData(SolverData):
    gravity: qd.Tensor


@dataclass(frozen=True, kw_only=True)
class GravityDescription:
    n_envs: int

    def allocate_gravity(self) -> qd.Tensor:
        # Kernels reading gravity through a solver attribute require field storage.
        return qd.tensor(gs.qd_vec3, (self.n_envs,), backend=qd.Backend.FIELD)
