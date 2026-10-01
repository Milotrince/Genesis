from dataclasses import dataclass

import quadrants as qd

import genesis as gs
from genesis.engine.solver_data import SolverData, SolverDescription
from genesis.engine.solver_data.articulated import JointsData, VisualGeomData
from genesis.engine.solvers import base_solver
from genesis.utils.array_class import DataItem, DataKind, V_VEC
from genesis.utils.misc import indices_to_mask, qd_to_torch


class Options(gs.options.SolverOptions):
    entity_name: str
    speed: float


class UnregisteredOptions(Options):
    pass


@dataclass(frozen=True, kw_only=True, eq=False)
class PositionsData(SolverData):
    positions: qd.Tensor


@dataclass(frozen=True, kw_only=True)
class Description(SolverDescription[PositionsData]):
    n_envs: int

    def allocate(self, owner):
        return PositionsData(owner=owner, positions=V_VEC(3, dtype=gs.qd_float, shape=(1, self.n_envs)))


class Solver(base_solver.Solver, options_cls=Options):
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        self._entity = None
        self._joints = None
        self._geometry = None

    @property
    def is_active(self):
        return True

    def describe(self):
        return Description(n_envs=self._B)

    def bind(self):
        self._entity = self.scene.get_entity(self._options.entity_name)
        joints = self.sim._solver_data.get(JointsData, self._entity.solver, self._entity.idx)
        self._joints = self.sim._solver_data.bind(joints, write="qpos")
        geom = self._entity.links[0].vgeoms[0]
        self._geometry = self.sim._solver_data.bind(
            self.sim._solver_data.get(VisualGeomData, self._entity.solver, geom.idx)
        )

    def build(self):
        self.update_positions()

    def substep_post_coupling(self, f):
        joints = self._joints.data
        qpos = self._entity.solver.get_qpos(qs_idx=slice(joints.q_start, joints.q_end))
        qpos[..., 0] += self._options.speed * self.sim.substep_dt
        self._joints.commit(qpos)
        self.update_positions()

    def update_positions(self):
        self._entity.solver.update_forward_pos()
        self._entity.solver.update_vgeoms()
        geom = self._geometry.data
        positions = qd_to_torch(geom.state.pos, col_mask=slice(geom.idx, geom.idx + 1), transpose=True)
        self._solver_data.positions.from_torch(positions.movedim(0, 1))

    def get_state(self, f):
        return qd_to_torch(self._solver_data.positions, transpose=True, copy=True)

    def set_state(self, f, state, envs_idx=None):
        positions = self.get_state(f)
        mask = indices_to_mask(envs_idx)
        positions[mask] = state[mask]
        self._solver_data.positions.from_torch(positions.movedim(0, 1))

    @gs.assert_built
    def get_positions(self):
        return self.get_state(f=0)

    @property
    def data(self):
        yield DataItem("positions", self._solver_data.positions, DataKind.STATE)


class OverlappingOptions(Options):
    pass


class OverlappingSolver(Solver, options_cls=OverlappingOptions):
    material_cls = gs.materials.Rigid
