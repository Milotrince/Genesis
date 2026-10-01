"""Grid state for the stable fluid (SF) solver."""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs

from . import SolverData, SolverDescription


@dataclass(frozen=True, kw_only=True, eq=False)
class SFData(SolverData):
    grid: qd.Field
    cell_size: float


@dataclass(frozen=True, kw_only=True)
class SFDescription(SolverDescription[SFData]):
    res: tuple[int, int, int]
    n_channels: int
    cell_size: float

    def allocate(self, owner) -> SFData:
        cell_state = qd.types.struct(
            v=gs.qd_vec3,
            v_tmp=gs.qd_vec3,
            div=gs.qd_float,
            p=gs.qd_float,
            q=qd.types.vector(self.n_channels, gs.qd_float),
        )
        return SFData(
            owner=owner, grid=cell_state.field(shape=self.res, layout=qd.Layout.SOA), cell_size=self.cell_size
        )
