import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import FloorBoundary
from genesis.engine.entities.tool_entity.tool_entity import ToolEntity
from genesis.engine.materials import Tool
from genesis.engine.solver_data.tool import ToolData, ToolDescription, ToolEntityDescription
from genesis.engine.states.solvers import ToolSolverState

from .base_solver import Solver, TimeBasedMixin


@qd.data_oriented
class ToolSolver(TimeBasedMixin, Solver):
    """
    Note
    ----
    !! This class will be removed once we added differntiability to the RigidSolver. This is just a temporary solution to obtain rigid->soft one-way differeitable coupling.
    """

    material_cls = Tool

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self.floor_height = options.floor_height

        # boundary
        self.setup_boundary()

    def describe(self) -> ToolDescription | None:
        if not self.is_active:
            return None
        return ToolDescription(
            entities=tuple(entity.desc for entity in self.entities),
            entity_idxs=tuple(entity.idx for entity in self.entities),
            n_envs=self._B,
            substeps_local=self.sim.substeps_local,
        )

    def bind(self):
        if not self.is_active:
            return
        assert isinstance(self._solver_data, ToolData)
        for entity, data in zip(self.entities, self._solver_data.geoms, strict=True):
            self.sim._solver_data.add(data)
            entity.pos = data.pos
            entity.quat = data.quat
            entity.vel = data.vel
            entity.ang = data.ang
            entity.mesh.init_vertices = data.init_vertices
            entity.mesh.init_vertex_normals = data.init_vertex_normals
            entity.mesh.faces = data.faces
            entity.mesh.vertices = data.vertices
            entity.mesh.vertex_normals = data.vertex_normals
            entity.mesh.sdf_voxels = data.sdf_voxels
            entity.mesh.T_mesh_to_sdf = data.T_mesh_to_sdf

    def build(self):
        for entity in self._entities:
            entity.build()

    @property
    def is_active(self):
        return self.n_entities > 0

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self.floor_height)

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "ToolEntity":
        if desc is None:
            desc = ToolEntityDescription.resolve(material, morph, surface, name)
        assert isinstance(desc, ToolEntityDescription)
        entity = ToolEntity(
            scene=self._scene,
            idx=idx,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            name=name,
            desc=desc,
        )
        self._entities.append(entity)
        return entity

    def reset_grad(self):
        for entity in self._entities:
            entity.reset_grad()

    def get_state(self, f):
        if self.is_active:
            state = ToolSolverState(self._scene)
            for entity in self._entities:
                state.entities.append(entity.get_state(f))
        else:
            state = None
        return state

    def set_state(self, f, state, envs_idx=None):
        if state is not None:
            assert len(state) == len(self._entities)
            for i, entity in enumerate(self._entities):
                entity.set_state(f, state[i])

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        for entity in self._entities:
            entity.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        for entity in self._entities[::-1]:
            entity.substep_pre_coupling_grad(f)

    def substep_post_coupling(self, f):
        for entity in self._entities:
            entity.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        for entity in self._entities[::-1]:
            entity.substep_post_coupling_grad(f)

    def add_grad_from_state(self, state):
        # Nothing needed here, since tool_solver state is composed of tool_entity.get_state(), which has already been cached inside each tool_entity.
        pass

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        if self.is_active:
            for entity in self._entities:
                entity.collect_output_grads()

    def save_ckpt(self, ckpt_name):
        for entity in self._entities:
            entity.save_ckpt(ckpt_name)

    def load_ckpt(self, ckpt_name):
        for entity in self._entities:
            entity.load_ckpt(ckpt_name=ckpt_name)

    @qd.func
    def pbd_collide(self, f, pos_world, thickness, dt):
        for entity in qd.static(self._entities):
            pos_world = entity.pbd_collide(f, pos_world, thickness, dt)
        return pos_world
