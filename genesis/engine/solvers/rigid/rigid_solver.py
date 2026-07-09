import math
import os
import sys
from typing import TYPE_CHECKING, Literal

import quadrants as qd
import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.entities import DroneEntity, RigidEntity
from genesis.engine.entities.base_entity import Entity
from genesis.engine.states import QueriedStates, RigidSolverState
from genesis.options.solvers import RigidOptions
from genesis.utils.misc import (
    DeprecationError,
    qd_to_torch,
    qd_to_numpy,
    qd_zero_grad,
    indices_to_mask,
    broadcast_tensor,
    sanitize_indexed_tensor,
    assign_indexed_tensor,
    tensor_to_array,
    get_gpu_core_count,
    fits_in_gpu_shared_memory,
)
from genesis.utils.sdf import SDF

from ..base_solver import Solver, StateChange, mutates
from ..kinematic_solver import (
    KinematicSolver,
    _select_links_offset,
    _offset_world_shift,
    _fill_base_link_geom_offsets,
    _update_geom_active_envs,
)
from .collider import Collider
from .constraint import ConstraintSolver
from .geom_pool_kernels import (
    kernel_entity_aabb_per_env,
    kernel_init_pool_geom_defaults,
    kernel_init_pool_vgeoms,
    kernel_set_pool_scale,
    kernel_update_pool_geoms,
    kernel_upload_geom_slot,
    kernel_upload_sdf_slot,
    kernel_upload_vert_slot,
)
from .abd.misc import (
    func_add_safe_backward,
    func_apply_coupling_force,
    func_apply_link_external_force,
    func_apply_external_torque,
    func_apply_link_external_torque,
    func_atomic_add_if,
    func_check_index_range,
    func_clear_external_force,
    func_read_field_if,
    func_write_field_if,
    func_write_and_read_field_if,
    kernel_init_invweight,
    kernel_init_meaninertia,
    kernel_init_dof_fields,
    kernel_reset_hibernation,
    kernel_init_link_fields,
    kernel_update_heterogeneous_link_info,
    kernel_init_joint_fields,
    kernel_init_vert_fields,
    kernel_init_vvert_fields,
    kernel_init_geom_fields,
    kernel_init_vgeom_fields,
    kernel_init_entity_fields,
    kernel_init_equality_fields,
    kernel_apply_links_external_force,
    kernel_apply_links_external_torque,
    kernel_update_geoms_render_T,
    kernel_update_vgeoms_render_T,
    kernel_bit_reduction,
    kernel_set_zero,
    kernel_clear_external_force,
)
from .abd.forward_kinematics import (
    func_aggregate_awake_entities,
    func_COM_links,
    func_COM_links_entity,
    func_forward_kinematics_entity,
    func_forward_kinematics_batch,
    func_forward_velocity_entity,
    func_forward_velocity_batch,
    func_forward_velocity,
    func_hibernate__for_all_awake_islands_either_hiberanate_or_update_aabb_sort_buffer,
    func_update_geoms_entity,
    func_update_geoms_batch,
    func_update_all_verts,
    func_update_cartesian_space,
    func_update_cartesian_space_entity,
    func_update_cartesian_space_batch,
    func_update_geoms,
    func_update_verts_for_geom,
    kernel_forward_kinematics_links_geoms,
    kernel_masked_forward_kinematics_links_geoms,
    kernel_forward_velocity,
    kernel_masked_forward_velocity,
    kernel_forward_kinematics_entity,
    kernel_update_geoms,
    kernel_update_verts_for_geoms,
    kernel_update_all_verts,
    kernel_update_geom_aabbs,
    kernel_update_vgeoms,
    kernel_update_cartesian_space,
)
from .abd.forward_dynamics import (
    func_actuation,
    func_bias_force,
    func_compute_mass_matrix,
    func_compute_qacc,
    func_factor_mass,
    func_forward_dynamics,
    func_solve_mass_entity,
    func_solve_mass_batch,
    func_solve_mass,
    func_torque_and_passive_force,
    func_update_acc,
    func_update_force,
    func_integrate,
    func_implicit_damping,
    func_vel_at_point,
    kernel_compute_mass_matrix,
    kernel_forward_dynamics,
    kernel_update_acc,
    kernel_compute_qacc,
    kernel_forward_dynamics_without_qacc,
    update_qacc_from_qvel_delta,
    update_qvel,
)
from .abd.accessor import (
    kernel_get_state,
    kernel_set_state,
    kernel_set_links_pos,
    kernel_set_links_quat,
    kernel_set_links_mass_shift,
    kernel_set_links_COM_shift,
    kernel_set_links_inertial_mass,
    kernel_wake_up_entities_by_links,
    kernel_wake_up_entities_by_dofs,
    kernel_wake_up_entities_by_qs,
    kernel_wake_up_entities_on_new_contact,
    kernel_set_geoms_friction_ratio,
    kernel_set_geoms_scale,
    kernel_set_vgeoms_scale,
    kernel_set_links_inertial,
    kernel_set_qpos,
    kernel_set_global_sol_params,
    kernel_set_sol_params,
    kernel_set_dofs_kp,
    kernel_set_dofs_kv,
    kernel_set_dofs_act_gain,
    kernel_set_dofs_act_bias,
    kernel_set_dofs_force_range,
    kernel_set_dofs_stiffness,
    kernel_set_dofs_armature,
    kernel_set_dofs_damping,
    kernel_set_dofs_frictionloss,
    kernel_set_dofs_limit,
    kernel_set_dofs_velocity,
    kernel_set_dofs_velocity_grad,
    kernel_set_dofs_zero_velocity,
    kernel_set_dofs_position,
    kernel_control_dofs_force,
    kernel_control_dofs_velocity,
    kernel_control_dofs_position,
    kernel_control_dofs_position_velocity,
    kernel_get_links_vel,
    kernel_get_links_acc,
    kernel_get_dofs_control_force,
    kernel_set_drone_rpm,
    kernel_update_drone_propeller_vgeoms,
    kernel_set_geom_friction,
    kernel_set_geoms_friction,
    kernel_adjust_link_inertia,
)
from .abd.diff import (
    func_copy_cartesian_space,
    func_copy_next_to_curr,
    func_copy_next_to_curr_grad,
    func_integrate_dq_entity,
    func_is_grad_valid,
    func_load_adjoint_cache,
    func_save_adjoint_cache,
    kernel_save_adjoint_cache,
    kernel_prepare_backward_substep,
    kernel_begin_backward_substep,
    kernel_copy_acc,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)

# minimum constraint impedance
IMP_MIN = 0.0001
# maximum constraint impedance
IMP_MAX = 0.9999

# Minimum ratio between simulation timestep `_substep_dt` and time constant of constraints
TIME_CONSTANT_SAFETY_FACTOR = 2.0


def _sanitize_sol_params(
    sol_params,
    min_timeconst: float,
    default_timeconst: float | None = None,
):
    timeconst, dampratio, dmin, dmax, width, mid, power = sol_params.reshape((-1, 7)).T
    if default_timeconst is None:
        default_timeconst = min_timeconst
    if (timeconst < gs.EPS).any():
        gs.logger.debug(
            f"Constraint solver time constant not specified. Using default value (`{default_timeconst:0.6g}`)."
        )
    invalid_mask = (timeconst > gs.EPS) & (timeconst + gs.EPS < min_timeconst)
    if invalid_mask.any():
        gs.logger.warning(
            "Constraint solver time constant should be greater than 2*substep_dt. timeconst is changed from "
            f"`{min(timeconst[invalid_mask]):0.6g}` to `{min_timeconst:0.6g}`). Decrease simulation timestep or "
            "increase timeconst to avoid altering the original value."
        )
    timeconst[timeconst < gs.EPS] = default_timeconst
    timeconst[:] = timeconst.clip(min_timeconst)
    if (dampratio < gs.EPS).any():
        gs.raise_exception(
            "Constraint solver `dampratio` must be strictly positive. Despite its name, it controls spring stiffness, "
            "not damping. See `genesis.utils.geom.default_solver_params` for details."
        )
    dmin[:] = dmin.clip(IMP_MIN, IMP_MAX)
    dmax[:] = dmax.clip(IMP_MIN, IMP_MAX)
    mid[:] = mid.clip(IMP_MIN, IMP_MAX)
    width[:] = width.clip(0.0)
    power[:] = power.clip(1)
    return sol_params


class RigidSolver(KinematicSolver):
    # override typing
    _entities: list[RigidEntity] = gs.List()

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: RigidOptions) -> None:
        super().__init__(scene, sim, options)

        self._enable_collision = options.enable_collision
        self._enable_multi_contact = options.enable_multi_contact
        self._enable_mujoco_compatibility = options.enable_mujoco_compatibility
        self._enable_joint_limit = options.enable_joint_limit
        self._enable_self_collision = options.enable_self_collision
        self._enable_neutral_collision = options.enable_neutral_collision
        self._enable_adjacent_collision = options.enable_adjacent_collision
        self._disable_constraint = options.disable_constraint
        self._max_collision_pairs = options.max_collision_pairs
        self._integrator = options.integrator
        self._box_box_detection = options.box_box_detection
        self._requires_grad = self._sim.options.requires_grad
        self._enable_heterogeneous = False  # Set to True when any entity has heterogeneous morphs
        self._enable_geom_scaling = options.enable_geom_scaling  # per-env runtime geom scale (set_scale)
        # Dynamic geometry residency pool (Phase 3). Declared per-entity via add_entity(geom_pool=...), which
        # flips _enable_geom_pool; the aggregated static layout is built in build() once real counts are known.
        self._enable_geom_pool = False
        self._geom_pool = None

        # Contact islands are off by default (opt in explicitly). The gate further below still disables them under
        # requires_grad (the differentiable adjoint reads the dense global Hessian) and for single-island scenes
        # (where the partition is pure overhead, unless hibernation needs it).
        self._use_contact_island = options.use_contact_island
        # Hibernation builds on islands, so requesting it without islands is a genuine conflict.
        self._use_hibernation = options.use_hibernation
        if self._use_hibernation and not self._use_contact_island:
            gs.raise_exception(
                "`use_hibernation=True` requires `use_contact_island=True`, as hibernation builds on islands."
            )

        # Resolve the hibernation velocity tolerance. MuJoCo compatibility uses MuJoCo's own default (1e-4); otherwise
        # use a coarser floor that a body reliably settles below across float precisions and dense contact piles, where
        # the contact solve leaves a larger residual resting-velocity jitter.
        if options.hibernation_thresh_vel is None:
            self._hibernation_thresh_vel = 1e-4 if self._enable_mujoco_compatibility else 2e-3
        else:
            self._hibernation_thresh_vel = options.hibernation_thresh_vel

        self._sol_min_timeconst = TIME_CONSTANT_SAFETY_FACTOR * self._substep_dt
        self._sol_default_timeconst = max(options.constraint_timeconst, self._sol_min_timeconst)

        self.collider = None
        self.constraint_solver = None

        self.qpos: qd.Tensor | qd.Field | qd.Ndarray | None = None

        self._is_backward: bool = False

        self._ckpt = dict()

    def init_ckpt(self):
        pass

    def add_entity(
        self, idx, material, morph, surface, visualize_contact, name: str | None = None, *, geom_pool=None
    ) -> RigidEntity:
        # Handle heterogeneous morphs (list/tuple of morphs)
        morph_heterogeneous = []
        if isinstance(morph, (tuple, list)):
            morph, *morph_heterogeneous = morph
            self._enable_heterogeneous |= bool(morph_heterogeneous)

        # A per-entity geometry pool (Phase 3) reserves runtime-swappable slots bound to this entity's base
        # link; enabling it forces the same batched-links_info / SAP path heterogeneous variants use. As a
        # shorthand, a bare list of object morphs is promoted to a GeomPoolOptions that auto-sizes the pool.
        if isinstance(geom_pool, (list, tuple)):
            geom_pool = gs.options.GeomPoolOptions(objects=list(geom_pool))
        pool_enabled = geom_pool is not None and bool(geom_pool.n_slots or geom_pool.objects)
        if pool_enabled:
            self._enable_geom_pool = True

        if isinstance(morph, gs.morphs.Drone):
            EntityClass = DroneEntity
        else:
            EntityClass = RigidEntity

        morph._enable_mujoco_compatibility = self._enable_mujoco_compatibility

        entity = EntityClass(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            idx_in_solver=self.n_entities,
            link_start=self.n_links,
            joint_start=self.n_joints,
            q_start=self.n_qs,
            dof_start=self.n_dofs,
            geom_start=self.n_geoms,
            cell_start=self.n_cells,
            vert_start=self.n_verts,
            free_verts_state_start=self.n_free_verts,
            fixed_verts_state_start=self.n_fixed_verts,
            face_start=self.n_faces,
            edge_start=self.n_edges,
            vgeom_start=self.n_vgeoms,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            custom_vvert_start=self.n_custom_vverts,
            custom_vface_start=self.n_custom_vfaces,
            visualize_contact=visualize_contact,
            morph_heterogeneous=morph_heterogeneous,
            geom_pool=geom_pool if pool_enabled else None,
            name=name,
        )
        assert isinstance(entity, RigidEntity)
        self._entities.append(entity)

        return entity

    def build(self):
        self._n_geoms = self.n_geoms
        self._n_cells = self.n_cells
        self._n_verts = self.n_verts
        self._n_free_verts = self.n_free_verts
        self._n_fixed_verts = self.n_fixed_verts
        self._n_faces = self.n_faces
        self._n_edges = self.n_edges
        self._n_equalities = self.n_equalities

        self._geoms = self.geoms
        self._equalities = self.equalities

        # Reserve a dynamic geometry pool block (Phase 3) appended to the global arrays. The logical _n_*
        # counts stay at the real entity count (so n_geoms == len(geoms) and Python-side iteration is
        # unaffected); only the device ALLOCATION sizes (n_*_) grow by the reserved block, giving kernels
        # (which bound on field .shape[0]) extra rows to hold runtime uploads. Base indices are the first free
        # row after the real entities. Reserved rows stay zero (inert: a zero quat rotates to a finite point
        # and contype == conaffinity == 0 yields no collision pairs) until set_active_object fills a slot.
        pool_geoms = pool_cells = pool_verts = pool_free_verts = pool_faces = pool_edges = 0
        if self._enable_geom_pool:
            from .geom_pool import GeometryPool

            # Aggregate every poolable entity's per-entity pool into one contiguous reserved block, starting at
            # the first free row in each global array. Each segment's slots are bound to that entity's base
            # link, so a slot's owning link is known at build (collision-pair validity precomputes cleanly).
            self._geom_pool = GeometryPool(
                base={
                    "geom": self._n_geoms,
                    "vert": self._n_verts,
                    "face": self._n_faces,
                    "edge": self._n_edges,
                    "cell": self._n_cells,
                    "free_vert": self._n_free_verts,
                    # Visual pooling reserves vgeom rows appended to the global vgeom array; the vgeom count is
                    # an entity-sum property readable pre-build. n_vgeoms_ is bumped in KinematicSolver.build.
                    "vgeom": self.n_vgeoms,
                }
            )
            for entity in self._entities:
                if entity._geom_pool_options is not None:
                    options = entity._geom_pool_options
                    # When a catalog is declared, process each object once now, cache it, and derive the
                    # per-slot budgets + slot count from the set (see _pool_derive_from_objects).
                    per_slot, n_slots, declared = (None, None, None)
                    if options.objects:
                        per_slot, declared = self._pool_derive_from_objects(entity, options.objects)
                        n_slots = int(options.n_slots) or len(options.objects)
                    segment = self._geom_pool.add_segment(
                        entity_idx=entity._idx_in_solver,
                        link_idx=entity.base_link.idx,
                        entity=entity,
                        options=options,
                        per_slot=per_slot,
                        n_slots=n_slots,
                    )
                    if declared is not None:
                        segment.declared = declared
            pool_geoms = self._geom_pool.n_geoms
            pool_cells = self._geom_pool.n_cells
            pool_verts = self._geom_pool.n_verts
            pool_free_verts = self._geom_pool.n_free_verts
            pool_faces = self._geom_pool.n_faces
            pool_edges = self._geom_pool.n_edges

        self.n_geoms_ = max(1, self._n_geoms + pool_geoms)
        self.n_cells_ = max(1, self._n_cells + pool_cells)
        self.n_verts_ = max(1, self._n_verts + pool_verts)
        self.n_faces_ = max(1, self._n_faces + pool_faces)
        self.n_edges_ = max(1, self._n_edges + pool_edges)
        self.n_free_verts_ = max(1, self._n_free_verts + pool_free_verts)
        self.n_fixed_verts_ = max(1, self._n_fixed_verts)
        self.n_candidate_equalities_ = max(1, self.n_equalities + self._options.max_dynamic_constraints)

        # Resolve precision-dependent tolerance default
        if self._options.tolerance is None:
            self._options.tolerance = 1e-5 if gs.qd_float == qd.f32 else 1e-8

        super().build()

        self._init_mass_mat()

        self._init_vert_fields()
        self._init_geom_fields()
        self._init_equality_fields()
        self._init_dof_length()

        # Seed the reserved pool block to finite defaults (identity quats) so unfilled slots pose to a finite
        # AABB rather than NaN. The init kernels above only touch the real geoms.
        if self._enable_geom_pool:
            kernel_init_pool_geom_defaults(
                self._n_geoms, self.n_geoms_, self.geoms_info, self.geoms_state, self._static_rigid_sim_config
            )
            self._init_pool_vgeoms()

        self._init_invweight_and_meaninertia(force_update=False)
        self._func_update_geoms(self._scene._envs_idx, force_update_fixed_geoms=True)

        self._init_collider()
        self._init_constraint_solver()

        # Morph pose offset of each collision geom, conjugated into the geom's own frame so the relative getters
        # revert it for geoms rotated relative to the link. Each root link carries its own offset; child-link geoms
        # inherit it through the kinematic chain and keep an identity offset. Forward offset device tensors, None when
        # everything is identity; the relative geom getters recompute the inverse.
        geoms_offset_pos = np.zeros((self.n_geoms, 3), dtype=gs.np_float)
        geoms_offset_quat = np.tile(gu.identity_quat(), (self.n_geoms, 1))
        for entity in self._entities:
            ranges = entity.base_link._variant_geom_ranges if entity._variant_offset_pos is not None else None
            _fill_base_link_geom_offsets(geoms_offset_pos, geoms_offset_quat, entity, entity.geoms, ranges)
        self._geoms_offset_pos = self._geoms_offset_quat = None
        if not (
            np.allclose(geoms_offset_pos, 0.0, atol=gs.EPS)
            and np.allclose(gu.quat_to_xyz(geoms_offset_quat), 0.0, atol=gs.EPS)
        ):
            self._geoms_offset_pos = torch.from_numpy(geoms_offset_pos).to(device=gs.device, dtype=gs.tc_float)
            self._geoms_offset_quat = torch.from_numpy(geoms_offset_quat).to(device=gs.device, dtype=gs.tc_float)

        # FIXME: when the migration is finished, we will remove the about two lines
        self._func_vel_at_point = func_vel_at_point
        self._func_apply_coupling_force = func_apply_coupling_force

    def _resolve_broadphase_traversal(self):
        if self._options.broadphase_traversal is not None:
            return self._options.broadphase_traversal
        # For hibernation, the main missing piece is skipping hibernated-vs-hibernated pairs. This means reading two
        # additional values from global memory, and the associated pipeline stall etc associated with this.
        # For heterogeneous, the valid_collision_pairs array is built once at init from the global geom pair
        # matrix, but with heterogeneous entities different batch elements have different geoms (different geom_start/
        # geom_end per link per batch), so a pair (ga, gb) might be valid in batch 0 but not exist in batch 3. To
        # support this we'd either need per-batch valid pair lists or runtime filtering that checks both geoms exist
        # in the current batch element. Per-batch lists multiply the memory footprint by the batch size, increasing
        # memory usage, and increasing L1/L2 cache contention. Runtime filtering keeps the single list, but it will
        # no longer be compact, and we will have thread divergence.
        # A dynamic geometry pool binds different slots to different envs at runtime (like heterogeneous
        # variants), so per-batch geom sets diverge and the single global valid-pair list requires SAP.
        if gs.backend == gs.cpu or self._use_hibernation or self._enable_heterogeneous or self._enable_geom_pool:
            return gs.broadphase_traversal.SAP
        return gs.broadphase_traversal.ALL_VS_ALL

    def _build_static_config(self):
        # The scene has multi-island block structure when it holds several independent DOF-carrying bodies or free
        # joints (the Hessian then splits into per-island blocks instead of one dense tree). This gates both the CPU
        # skyline solver and the GPU per-island force below: a single dense-coupled tree (e.g. one big robot) is one
        # island and gains nothing from either.
        n_dof_entities = sum(entity.n_dofs > 0 for entity in self.entities)
        n_free_joints = sum(joint.type == gs.JOINT_TYPE.FREE for joint in self.joints)
        has_multi_island_structure = n_dof_entities >= 2 or n_free_joints >= 2

        # Islands only reduce work when the scene splits into several blocks. With a single dense-coupled tree (one
        # island) the partition is pure overhead, so disable it in computation even if the user opted in. Hibernation
        # does not force islands on (a scene with no island structure has nothing to gain from sleeping a lone tree);
        # use_hibernation is gated off below to follow this decision. The differentiable solve reads the dense global
        # Hessian (nt_H), not the per-island tiles, so islands stay off under requires_grad regardless.
        self._use_contact_island = self._use_contact_island and has_multi_island_structure and not self._requires_grad

        # Hibernation builds on the island partition, so it cannot outlive islands being turned off by any gate above.
        # Re-sync it to the final island decision so the two never disagree.
        self._use_hibernation = self._use_hibernation and self._use_contact_island

        # A heterogeneous entity has a different body size (hence rotational dof_length) per variant, so its dof_length
        # is genuinely per-env and dofs_info must be batched to hold it. dof_length is read only by the hibernation
        # rest test, so this is needed exactly when both features are active. We must update options because
        # get_dofs_info reads from solver._options.batch_dofs_info.
        if self._enable_heterogeneous and self._use_hibernation:
            self._options.batch_dofs_info = True

        # sparse_solve=None resolves automatically: the skyline-envelope solver pays off on CPU only when the scene
        # has block structure, whereas a single dense-coupled tree gains nothing and pays the per-step envelope tax. An
        # explicit value overrides this. On GPU the envelope factorization is dropped (the dense tiled path is faster
        # there); an explicit True still enables the assembly-level sparsity, with a warning.
        if self._options.sparse_solve is None:
            sparse_solve = gs.backend == gs.cpu and not self._enable_mujoco_compatibility and has_multi_island_structure
        else:
            sparse_solve = self._options.sparse_solve
            if sparse_solve and gs.backend != gs.cpu:
                gs.logger.warning(
                    "Enabling 'sparse_solve' on the GPU backend likely impedes performance; the dense tiled "
                    "factorization is faster there. Use with caution."
                )

        # sparse-skyline and per-island exploit the block-diagonal Hessian from complementary angles, so on CPU
        # they COMPOSE rather than compete: islands give each block its own cheap Hessian factorization, while the
        # sparse Jacobian representation makes the per-iteration Jacobian-vector products, the constraint-to-island
        # lookup, and the Hessian assembly cost O(nonzeros) instead of O(n_constraints * n_dofs). With both on, the
        # many-small-bodies solve scales near-linearly in body count (measured ~2.7x faster than sparse alone and
        # ~8x faster than islands alone at 256 boxes); the island Hessian branch naturally bypasses the skyline
        # envelope factorization. The differentiable adjoint solve reads the dense Hessian, so the composition is
        # restricted to the forward (non-grad) path. On GPU the dense tiled path is faster, so sparse is dropped and
        # islands stand alone.
        if gs.backend == gs.cpu and self._use_contact_island and sparse_solve and not self.sim.options.requires_grad:
            pass  # compose islands + sparse Jacobian
        elif sparse_solve and gs.backend == gs.cpu:
            self._use_contact_island = False
        elif self._use_contact_island:
            sparse_solve = False

        # The skyline-envelope factorization and its DOF reorder are CPU-only and incompatible with the differentiable
        # adjoint solve (which reuses nt_H with natural, dense indexing). Under requires_grad only the assembly-level
        # sparsity applies, matching the pre-existing behaviour. When islands are also active (the CPU composition),
        # the per-island Hessian branch factorizes each block directly and never reads the skyline envelope, so the
        # O(n_dofs^2) per-step envelope computation would be pure waste - drop it and let islands own the factorization.
        sparse_envelope = (
            sparse_solve
            and gs.backend == gs.cpu
            and not self.sim.options.requires_grad
            and not self._use_contact_island
        )

        # The layout-flippable constraint-state tensors are stored batch-first either for the GPU cooperative kernels or
        # under serialized execution, where the env loop is outermost and per-env rows must be contiguous to avoid
        # stride-n_envs access. Batched sweeps key their iteration-axis order on the same flag, so that iteration order
        # always follows the physical layout.
        #
        # The subgroup-cooperative constraint kernels (and the batch-first layout they expect) win when per-env compute
        # density amortizes the warp-per-env overhead, and lose when envs are sparse and many (the 1-thread-per-env path
        # is already coalesced under (len_constraints_, _B)). They are also the layout the decomposed solve arm requires.
        # Empirically the cooperative path wins around 4096 envs at n_dofs >= ~18 and washes out by ~30000 envs at
        # n_dofs <= ~12; the n_envs <= 8192 and n_dofs >= 16 thresholds bound that crossover. Sparse solve is excluded
        # (the cooperative qfrc kernel and the flipped-layout jac readers are dense-only).
        enable_cooperative_constraint_kernels = (
            gs.backend != gs.cpu
            and not self.sim.options.requires_grad
            and not sparse_solve
            and self._sim._B <= 8192
            and self.n_dofs >= 16
        )
        constraint_layout_batch_first = (
            enable_cooperative_constraint_kernels or self.sim._para_level < gs.PARA_LEVEL.ALL
        )

        static_rigid_sim_config = dict(
            backend=gs.backend,
            para_level=self.sim._para_level,
            requires_grad=self.sim.options.requires_grad,
            use_hibernation=self._use_hibernation,
            batch_links_info=self._options.batch_links_info,
            batch_dofs_info=self._options.batch_dofs_info,
            batch_joints_info=self._options.batch_joints_info,
            enable_mujoco_compatibility=self._enable_mujoco_compatibility,
            enable_multi_contact=self._enable_multi_contact,
            enable_collision=self._enable_collision,
            enable_joint_limit=self._enable_joint_limit,
            box_box_detection=self._box_box_detection,
            use_contact_island=self._use_contact_island,
            # The per-island solve engages wherever islands are on by default (CPU, where it composes with the sparse
            # skyline). The GPU block below narrows it to exclude the whole-env-fits-shared no-hibernation case, which
            # factors faster through the whole-env path (its block-diagonal Cholesky is the exact per-island result).
            enable_per_island_solve=self._use_contact_island,
            sparse_solve=sparse_solve,
            sparse_envelope=sparse_envelope,
            integrator=self._integrator,
            solver_type=self._options.constraint_solver,
            broadphase_traversal=self._resolve_broadphase_traversal(),
            # Parallelize init over (constraints, envs) when envs alone don't saturate the GPU.
            parallel_init=(
                gs.backend != gs.cpu and not self.sim.options.requires_grad and self.n_envs <= get_gpu_core_count()
            ),
            enable_cooperative_constraint_kernels=enable_cooperative_constraint_kernels,
            constraint_layout_batch_first=constraint_layout_batch_first,
            enable_geom_scaling=self._enable_geom_scaling,
        )

        # Prefer the monolith solver on CPU (always faster there, perf dispatch is a waste of effort)
        if gs.backend == gs.cpu or self.sim.options.requires_grad:
            static_rigid_sim_config["prefer_decomposed_solver"] = 0

        if self.is_active:
            # The tiled and cooperative Cholesky kernels trade per-env serial work for cross-lane parallelism, so they
            # only help while envs alone do not already saturate the GPU. Above that env count one-thread-per-env keeps
            # every core busy and the scalar path wins; below it the parallel kernels hide latency by swapping warps.
            # The crossover is also hardware- and kernel-dependent, so the env threshold (GPU core count) is a heuristic
            # and a dynamic timer-based selection would be more accurate still.
            max_n_dofs_per_entity = max(entity.n_dofs for entity in self.entities) if self.entities else 0
            if gs.backend != gs.cpu:
                max_tiled_envs = get_gpu_core_count()
                envs_undersaturate = self.n_envs <= max_tiled_envs

                # n_dofs-based dispatch between Tile16x16 and Tile32x32 Cholesky kernels (Hessian only).
                # Derived from a padded-volume + sub-warp utilization model:
                #   n_dofs in [1..16]    -> T=16 (one tight tile, no benefit going to T=32)
                #   n_dofs in [17..32]   -> T=32 (single 32-lane tile beats two sequential 16-lane tiles)
                #   n_dofs in [33..48]   -> T=16 (T=32 pads to 64 = ~29 wasted lanes; T=16 pads to 48 = ~13 wasted)
                #   n_dofs in [49..]     -> T=32 (lane utilization wins, T=16 needs many sequential tiles)
                # Confirmed by dex_hand (n_dofs=62, T=32 +2.6 %) and g1_fall (n_dofs=35, T=16 +2.9 %).
                cholesky_tile_size = 16 if (self.n_dofs <= 16 or 32 < self.n_dofs <= 48) else 32
                tiled_n_dofs = max(math.ceil(self.n_dofs / cholesky_tile_size), 1) * cholesky_tile_size
                tiled_n_dofs_per_entity = max(math.ceil(max_n_dofs_per_entity / 32), 1) * 32

                # The decomposed arm's cooperative per-island solve stages one island's tile in shared memory.
                # Size it to the largest tile-size multiple that fits shared (precision-aware), but no larger
                # than tiled_n_dofs; an island exceeding this falls back to the serial per-island solve. Unlike
                # hessian_fits_shared (which sizes the whole-env tile and is often False for big envs), this is
                # always usable because islands are small - it only caps how big a single island may be before
                # it loses the cooperative path.
                tiled_n_island_dofs = tiled_n_dofs
                while tiled_n_island_dofs > cholesky_tile_size and not fits_in_gpu_shared_memory(
                    tiled_n_island_dofs, tiled_n_island_dofs
                ):
                    tiled_n_island_dofs -= cholesky_tile_size

                # enable_tiled_cholesky_hessian selects the register-streaming tiled factor (no shared-memory cap):
                # worth tiling from n_dofs >= 16, and below the shared cap only when envs undersaturate (above it the
                # scalar O(n_dofs^3) per-env factor is always worse). hessian_fits_shared additionally gates the
                # shared-memory tiled triangular solve and fused factor+solve, which stage the full L tile in shared.
                hessian_fits_shared = fits_in_gpu_shared_memory(tiled_n_dofs, tiled_n_dofs + 1)
                enable_tiled_cholesky_hessian = self.n_dofs >= 16 and (not hessian_fits_shared or envs_undersaturate)

                # The cooperative in-place LDL^T has no cap; the shared-memory tile is faster but capped. Same env logic
                # as the Hessian: tile from n_dofs_per_entity >= 8, drop the env guard above the cap where the scalar
                # O(n_dofs^3) per-(entity, env) factor is always worse.
                mass_matrix_fits_shared = fits_in_gpu_shared_memory(
                    tiled_n_dofs_per_entity, tiled_n_dofs_per_entity + 1
                )
                enable_tiled_cholesky_mass_matrix = max_n_dofs_per_entity >= 8 and (
                    not mass_matrix_fits_shared or envs_undersaturate
                )

                # Register-streaming tiled mass factor for the >shared-cap forward GPU path: it factors each mass block
                # (kinematic tree) in registers via the same primitive as the Hessian, and is faster than and
                # numerically matches the cooperative LDL^T. Reuses cholesky_tile_size (always 32 here, since the path
                # needs a per-entity block exceeding shared memory).
                enable_register_tiled_mass = (
                    enable_tiled_cholesky_mass_matrix and not mass_matrix_fits_shared and not self._requires_grad
                )

                # Route the per-step warm-start factor+solve through the fused kernel whenever the shared tiled solve is
                # available (factor tiled and L fits shared). The monolith body's incremental rank-1 update needs L in
                # nt_H, so the fused kernel also writes L back via the ``write_L_to_nt_H`` argument; see
                # ``func_update_gradient_tiled``. Disabled for ``sparse_solve`` because the sparse path runs the per-env
                # factor inside ``func_hessian_and_cholesky_factor_direct_batch`` (leaving nt_H = L); routing the
                # warm-start through the fused kernel would then re-factor L as if it were H.
                enable_fused_factor_solve_init = (
                    enable_tiled_cholesky_hessian and hessian_fits_shared and not sparse_solve
                )

                static_rigid_sim_config.update(
                    enable_tiled_cholesky_mass_matrix=enable_tiled_cholesky_mass_matrix,
                    mass_matrix_fits_shared=mass_matrix_fits_shared,
                    enable_register_tiled_mass=enable_register_tiled_mass,
                    enable_tiled_cholesky_hessian=enable_tiled_cholesky_hessian,
                    hessian_fits_shared=hessian_fits_shared,
                    cholesky_tile_size=cholesky_tile_size,
                    enable_fused_factor_solve_init=enable_fused_factor_solve_init,
                    enable_per_island_solve=(
                        self._use_contact_island and (self._use_hibernation or not hessian_fits_shared)
                    ),
                    tiled_n_dofs_per_entity=tiled_n_dofs_per_entity,
                    tiled_n_dofs=tiled_n_dofs,
                    tiled_n_island_dofs=tiled_n_island_dofs,
                    # Persistent block grid for the cooperative per-island factor+solve: enough T-lane blocks to fill the
                    # GPU (one block ~= one tile = cholesky_tile_size lanes). The blocks grid-stride over the (env,
                    # island) work-list, so a small batch with many islands fans out across blocks instead of
                    # serializing inside one block-per-env. The count is independent of the body/env count (only the GPU
                    # size and cholesky_tile_size, which already varies the kernels via n_dofs): an ndarray-mode kernel
                    # must compile once and run for any n_objs/n_envs, and a block with no work exits at the grid-stride
                    # guard (blk >= work_size) within the same scheduling wave, so over-launching a tiny work-list is free.
                    island_factor_n_blocks=max(1, max_tiled_envs // cholesky_tile_size),
                )

                # Manually pin the solve arm only where the winner is determinable in advance AND confirmed across
                # CUDA + Metal; genuinely backend-dependent cases fall through to the per-step autotuner.
                if not enable_cooperative_constraint_kernels:
                    # No cooperative layout (n_envs > 8192 already saturates the GPU, or n_dofs < 16): the decomposed
                    # arm has nothing to exploit, so the scalar one-thread-per-env monolith is the clear winner.
                    static_rigid_sim_config["prefer_decomposed_solver"] = 0

            # Add terms for static inner loops, use -1 if not requires_grad to avoid re-compilation
            if self.sim.options.requires_grad:
                static_rigid_sim_config.update(
                    max_n_geoms_per_entity=max(len(entity.geoms) for entity in self.entities) if self.links else 0,
                    n_entities=self._n_entities,
                    n_links=self._n_links,
                    n_geoms=self._n_geoms,
                )

        self._static_rigid_sim_config = array_class.RigidSimStaticConfig(**static_rigid_sim_config)

        if self._static_rigid_sim_config.requires_grad:
            if self._static_rigid_sim_config.use_hibernation:
                gs.raise_exception("Hibernation is not supported yet when requires_grad is True")
            if self._static_rigid_sim_config.integrator != gs.integrator.approximate_implicitfast:
                gs.raise_exception(
                    "Only approximate_implicitfast integrator is supported yet when requires_grad is True."
                )
            from genesis.engine.couplers import SAPCoupler, IPCCoupler

            if isinstance(self.sim.coupler, (SAPCoupler, IPCCoupler)):
                gs.raise_exception(
                    f"{type(self.sim.coupler).__name__} is not supported yet when requires_grad is True."
                )

            if getattr(self._options, "noslip_iterations", 0) > 0:
                gs.raise_exception("Noslip is not supported yet when requires_grad is True.")

    def _create_data_manager(self):
        # We initialize data even if the solver is not active because the coupler needs arguments like
        # rigid_solver.links_state, etc. regardless of the solver is active or not.
        self.data_manager = array_class.DataManager(self, kinematic_only=False)
        self._errno = self.data_manager.errno

        self._rigid_global_info = self.data_manager.rigid_global_info
        self._rigid_adjoint_cache = self.data_manager.rigid_adjoint_cache
        if self._use_hibernation:
            self.n_awake_dofs = self._rigid_global_info.n_awake_dofs
            self.awake_dofs = self._rigid_global_info.awake_dofs
            self.n_awake_links = self._rigid_global_info.n_awake_links
            self.awake_links = self._rigid_global_info.awake_links
            self.n_awake_entities = self._rigid_global_info.n_awake_entities
            self.awake_entities = self._rigid_global_info.awake_entities
        if self._requires_grad:
            self.dofs_state_adjoint_cache = self.data_manager.dofs_state_adjoint_cache
            self.links_state_adjoint_cache = self.data_manager.links_state_adjoint_cache
            self.joints_state_adjoint_cache = self.data_manager.joints_state_adjoint_cache
            self.geoms_state_adjoint_cache = self.data_manager.geoms_state_adjoint_cache

    def _sanitize_joint_sol_params(self, sol_params):
        return _sanitize_sol_params(sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

    def _sanitize_geom_sol_params(self, sol_params):
        return _sanitize_sol_params(sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

    def _init_invweight_and_meaninertia(self, envs_idx=None, *, force_update=True):
        # Early return if no DoFs. This is essential to avoid segfault on CUDA.
        if self._n_dofs == 0:
            return

        # Handling default arguments
        batched = self._options.batch_dofs_info or self._options.batch_links_info
        if not batched and envs_idx is not None:
            gs.raise_exception(
                "Links and dofs must be batched to selectively update invweight and meaninertia for some environment."
            )
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        # Compute state in neutral configuration at rest
        qpos = qd_to_torch(self.qpos0, envs_idx, transpose=True)
        if self.n_envs == 0:
            qpos = qpos[0]
        self.set_qpos(qpos, envs_idx=envs_idx if self.n_envs > 0 else None)

        # Compute mass matrix without any implicit damping terms
        # TODO: This kernel could be optimized to take `envs_idx` as input if performance is critical.
        kernel_compute_mass_matrix(
            links_state=self.links_state,
            links_info=self.links_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            decompose=True,
        )

        # Define some proxies for convenience
        mass_mat_D_inv = qd_to_numpy(self._rigid_global_info.mass_mat_D_inv)
        mass_mat_L = qd_to_numpy(self._rigid_global_info.mass_mat_L)
        offsets = qd_to_numpy(self.links_state.i_pos)
        cdof_ang = qd_to_numpy(self.dofs_state.cdof_ang)
        cdof_vel = qd_to_numpy(self.dofs_state.cdof_vel)
        links_joint_start = qd_to_numpy(self.links_info.joint_start)
        links_joint_end = qd_to_numpy(self.links_info.joint_end)
        links_dof_end = qd_to_numpy(self.links_info.dof_end)
        links_n_dofs = qd_to_numpy(self.links_info.n_dofs)
        links_parent_idx = qd_to_numpy(self.links_info.parent_idx)
        joints_type = qd_to_numpy(self.joints_info.type)
        joints_dof_start = qd_to_numpy(self.joints_info.dof_start)
        joints_n_dofs = qd_to_numpy(self.joints_info.n_dofs)

        links_invweight = np.zeros((len(envs_idx), self._n_links, 2), dtype=gs.np_float)
        dofs_invweight = np.zeros((len(envs_idx), self._n_dofs), dtype=gs.np_float)

        # TODO: Simple numpy-based for-loop for now as it is not performance critical
        for i_b_, i_b in enumerate(envs_idx):
            # Compute the inverted mass matrix efficiently
            mass_mat_L_inv = np.eye(self.n_dofs_)
            for i_d in range(self.n_dofs_):
                for j_d in range(i_d):
                    mass_mat_L_inv[i_d] -= mass_mat_L[i_d, j_d, i_b] * mass_mat_L_inv[j_d]
            mass_mat_inv = (mass_mat_L_inv * mass_mat_D_inv[:, i_b]) @ mass_mat_L_inv.T

            # Compute links invweight if necessary
            if i_b_ == 0 or self._options.batch_links_info:
                for i_l in range(self._n_links):
                    jacp = np.zeros((3, self._n_dofs))
                    jacr = np.zeros((3, self._n_dofs))

                    offset = offsets[i_l, i_b]

                    j_l = i_l
                    while j_l != -1:
                        link_n_dofs = links_n_dofs[j_l]
                        if self._options.batch_links_info:
                            link_n_dofs = link_n_dofs[i_b]
                        for i_d_ in range(link_n_dofs):
                            link_dof_end = links_dof_end[j_l]
                            if self._options.batch_links_info:
                                link_dof_end = link_dof_end[i_b]
                            i_d = link_dof_end - i_d_ - 1
                            jacp[:, i_d] = cdof_vel[i_d, i_b] + np.cross(cdof_ang[i_d, i_b], offset)
                            jacr[:, i_d] = cdof_ang[i_d, i_b]
                        link_parent_idx = links_parent_idx[j_l]
                        if self._options.batch_links_info:
                            link_parent_idx = link_parent_idx[i_b]
                        j_l = link_parent_idx

                    jac = np.concatenate((jacp, jacr), axis=0)

                    A = jac @ mass_mat_inv @ jac.T
                    A_diag = np.diag(A)

                    tran = A_diag[:3].mean()
                    rot = A_diag[3:].mean()

                    # If one component is zero, use the other to prevent degenerate constraints.
                    # See https://github.com/google-deepmind/mujoco/commit/1cda1e7a
                    if tran < gs.EPS and rot > gs.EPS:
                        tran = rot
                    elif rot < gs.EPS and tran > gs.EPS:
                        rot = tran

                    links_invweight[i_b_, i_l, 0] = tran
                    links_invweight[i_b_, i_l, 1] = rot

            # Compute dofs invweight
            if i_b_ == 0 or self._options.batch_dofs_info:
                for i_l in range(self._n_links):
                    link_joint_start = links_joint_start[i_l]
                    link_joint_end = links_joint_end[i_l]
                    if self._options.batch_links_info:
                        link_joint_start = link_joint_start[i_b]
                        link_joint_end = link_joint_end[i_b]
                    for i_j in range(link_joint_start, link_joint_end):
                        joint_type = joints_type[i_j]
                        if self._options.batch_joints_info:
                            joint_type = joint_type[i_b]
                        if joint_type == gs.JOINT_TYPE.FIXED:
                            continue

                        dof_start = joints_dof_start[i_j]
                        n_dofs = joints_n_dofs[i_j]
                        if self._options.batch_joints_info:
                            dof_start = dof_start[i_b]
                            n_dofs = n_dofs[i_b]
                        jac = np.zeros((n_dofs, self._n_dofs))
                        for i_d_ in range(n_dofs):
                            jac[i_d_, dof_start + i_d_] = 1.0

                        A = jac @ mass_mat_inv @ jac.T
                        A_diag = np.diag(A)

                        if joint_type == gs.JOINT_TYPE.FREE:
                            dofs_invweight[i_b_, dof_start : (dof_start + 3)] = A_diag[:3].mean()
                            dofs_invweight[i_b_, (dof_start + 3) : (dof_start + 6)] = A_diag[3:].mean()
                        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                            dofs_invweight[i_b_, dof_start : (dof_start + 3)] = A_diag[:3].mean()
                        else:  # REVOLUTE or PRISMATIC
                            dofs_invweight[i_b_, dof_start] = A_diag[0]

            # Stop there if not batched
            if not batched:
                break

        # Update links and dofs invweight if necessary
        if not self._options.batch_links_info:
            links_invweight = links_invweight[0]
        if not self._options.batch_dofs_info:
            dofs_invweight = dofs_invweight[0]
        kernel_init_invweight(
            envs_idx,
            links_invweight,
            dofs_invweight,
            links_info=self.links_info,
            dofs_info=self.dofs_info,
            force_update=force_update,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

        # Compute meaninertia from mass matrix
        kernel_init_meaninertia(
            envs_idx=envs_idx,
            rigid_global_info=self._rigid_global_info,
            entities_info=self.entities_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def _init_mass_mat(self):
        self.mass_mat = self._rigid_global_info.mass_mat
        self.mass_mat_L = self._rigid_global_info.mass_mat_L
        self.mass_mat_D_inv = self._rigid_global_info.mass_mat_D_inv
        self.mass_mat_mask = self._rigid_global_info.mass_mat_mask
        self.meaninertia = self._rigid_global_info.meaninertia

        self.mass_mat_mask.fill(True)

        # tree structure information
        mass_parent_mask = np.zeros((self.n_dofs_, self.n_dofs_), dtype=gs.np_float)
        for i_l in range(self.n_links):
            j_l = i_l
            while j_l != -1:
                for i_d, j_d in qd.ndrange(
                    (self.links[i_l].dof_start, self.links[i_l].dof_end),
                    (self.links[j_l].dof_start, self.links[j_l].dof_end),
                ):
                    mass_parent_mask[i_d, j_d] = 1.0
                j_l = self.links[j_l].parent_idx

        # Partition each entity's DOFs into contiguous, independently-factorable blocks. M is block-diagonal between
        # DOFs whose links are not kinematic ancestor/descendant of one another, so the per-block bounds let the
        # assemble/factor/solve restrict to one block instead of the whole entity. A block is the set of DOFs coupled
        # through a chain of moving joints; a fixed link (0 DOFs) does not couple, so several bodies attached to the
        # world - or the independent arms of a fixed-base robot - factor as separate per-branch blocks. Each DOF's block
        # is rooted at the topmost DOF-bearing ancestor reachable before the world; DOFs are numbered depth-first, so a
        # block is a contiguous range whose root link's first DOF is the block start.
        links_by_idx = {link.idx: link for link in self.links}
        block_start = np.arange(self.n_dofs_, dtype=gs.np_int)
        for link in self.links:
            if link.n_dofs == 0:
                continue
            root_link = link
            node = link
            while node.parent_idx != -1:
                node = links_by_idx[node.parent_idx]
                if node.n_dofs > 0:
                    root_link = node
            block_start[link.dof_start : link.dof_end] = root_link.dof_start
        block_end = np.empty(self.n_dofs_, dtype=gs.np_int)
        for i_d in range(self.n_dofs_):
            block_end[block_start[i_d]] = i_d + 1
        block_end = block_end[block_start]

        # An aligned free body whose only DOFs are its own free joint has a diagonal joint-space mass block, so zero its
        # within-link off-diagonal mask to make the assembled mass exactly diagonal (else ~1e-6 round-off once it
        # rotates) and the skyline envelope tighter. A DOF-bearing (articulated) descendant adds off-diagonal base
        # coupling, so the block must be exactly the root's own DOFs for the diagonalization to be valid.
        for link in self.links:
            # 'aligned' already implies a free joint; the block bounds must additionally be exactly the link's own DOFs
            # (no DOF-bearing ancestor or descendant), otherwise the coupled block is not diagonal.
            if (
                not link.aligned
                or block_start[link.dof_start] != link.dof_start
                or block_end[link.dof_start] != link.dof_end
            ):
                continue
            for i_d in range(link.dof_start, link.dof_end):
                for j_d in range(link.dof_start, link.dof_end):
                    if i_d != j_d:
                        mass_parent_mask[i_d, j_d] = 0.0
                block_start[i_d] = i_d
                block_end[i_d] = i_d + 1

        self._rigid_global_info.mass_parent_mask.from_numpy(mass_parent_mask)
        self._rigid_global_info.dofs_mass_block_start.from_numpy(block_start)
        self._rigid_global_info.dofs_mass_block_end.from_numpy(block_end)

        self._rigid_global_info.gravity.from_numpy(self.gravity)

    def _bind_link_variant(self, link, variant_idx, envs_idx):
        """Bind a per-environment variant selection for one link.

        Extends the base (vgeom ranges + active-env masks) to also bind collision geom ranges and
        per-variant inertial (mass/COM/orientation/tensor). Per-variant inertial is pre-computed during
        link._build() from the variant geoms, using analytic formulas for primitives. Shared by
        build-time dispatch and runtime rebind; 'variant_idx[i]' is the variant used by 'envs_idx[i]'.
        """
        geom_starts = np.array([link._variant_geom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
        geom_ends = np.array([link._variant_geom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)
        vgeom_starts = np.array([link._variant_vgeom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
        vgeom_ends = np.array([link._variant_vgeom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)
        links_inertial_mass = np.array([link._variant_inertial[v][0] for v in variant_idx], dtype=gs.np_float)
        links_inertial_pos = np.array([link._variant_inertial[v][1] for v in variant_idx], dtype=gs.np_float)
        links_inertial_quat = np.array([link._variant_inertial[v][2] for v in variant_idx], dtype=gs.np_float)
        links_inertial_i = np.array([link._variant_inertial[v][3] for v in variant_idx], dtype=gs.np_float)

        # When batch_links_info is True, links_info has shape (n_links, B) so each env holds its variant.
        kernel_update_heterogeneous_link_info(
            link.idx,
            envs_idx,
            geom_starts,
            geom_ends,
            vgeom_starts,
            vgeom_ends,
            links_inertial_mass,
            links_inertial_pos,
            links_inertial_quat,
            links_inertial_i,
            self.links_info,
        )

        for geom in link.geoms:
            _update_geom_active_envs(geom, geom_starts, geom_ends, envs_idx, self._B)
        for vgeom in link.vgeoms:
            _update_geom_active_envs(vgeom, vgeom_starts, vgeom_ends, envs_idx, self._B)

    def _init_vert_fields(self):
        self.verts_info = self.data_manager.verts_info
        self.faces_info = self.data_manager.faces_info
        self.edges_info = self.data_manager.edges_info
        self.free_verts_state = self.data_manager.free_verts_state
        self.fixed_verts_state = self.data_manager.fixed_verts_state

        if self.n_verts > 0:
            geoms = self.geoms
            kernel_init_vert_fields(
                verts=np.concatenate([geom.init_verts for geom in geoms], dtype=gs.np_float),
                faces=np.concatenate([geom.init_faces + geom.vert_start for geom in geoms], dtype=gs.np_int),
                edges=np.concatenate([geom.init_edges + geom.vert_start for geom in geoms], dtype=gs.np_int),
                normals=np.concatenate([geom.init_normals for geom in geoms], dtype=gs.np_float),
                verts_geom_idx=np.concatenate([np.full(geom.n_verts, geom.idx) for geom in geoms], dtype=gs.np_int),
                init_center_pos=np.concatenate([geom.init_center_pos for geom in geoms], dtype=gs.np_float),
                verts_state_idx=np.concatenate(
                    [np.arange(geom.verts_state_start, geom.verts_state_start + geom.n_verts) for geom in geoms],
                    dtype=gs.np_int,
                ),
                is_fixed=np.concatenate(
                    [np.full(geom.n_verts, geom.is_fixed and not geom.entity._batch_fixed_verts) for geom in geoms],
                    dtype=gs.np_bool,
                ),
                # Quadrants variables
                verts_info=self.verts_info,
                faces_info=self.faces_info,
                edges_info=self.edges_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_dof_length(self):
        # Characteristic length of each dof (1 for translation, the body radius for rotation), used to weight dof
        # velocities in the hibernation rest test. Computed here, after geom dispatch, because a heterogeneous entity's
        # per-variant geoms (and hence body radius) are only assigned to environments at that point. Only needed when
        # hibernation is on, which already implies use_contact_island and a non-differentiable solve.
        if not self._use_hibernation:
            return

        joints = self.joints
        if sum(joint.n_dofs for joint in joints) == 0:
            return

        # dofs_length is per-env only for a heterogeneous entity; broadcast the shared row across envs when dofs_info
        # is batched (always so for a heterogeneous entity, optionally for a homogeneous one).
        dof_length = np.concatenate([joint.dofs_length for joint in joints], axis=0)
        if self._options.batch_dofs_info and dof_length.ndim == 1:
            dof_length = np.broadcast_to(dof_length[:, None], (len(dof_length), self._B))
        self.dofs_info.dof_length.from_numpy(dof_length)

    def _init_geom_fields(self):
        self.geoms_info: array_class.GeomsInfo = self.data_manager.geoms_info
        self.geoms_state: array_class.GeomsState = self.data_manager.geoms_state
        self.geoms_init_AABB = self._rigid_global_info.geoms_init_AABB
        self._geoms_render_T = np.empty((self.n_geoms_, self._B, 4, 4), dtype=np.float32)

        if self.n_geoms > 0:
            geoms = self.geoms
            geoms_sol_params = np.array([geom.sol_params for geom in geoms], dtype=gs.np_float)
            _sanitize_sol_params(geoms_sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

            # Accurately compute the center of mass of each geometry if possible.
            # Note that the mean vertex position is a bad approximation, which is impeding the ability of MPR to
            # estimate the exact contact information.
            geoms_center = []
            for geom in geoms:
                tmesh = geom.mesh.trimesh
                if tmesh.is_watertight:
                    geoms_center.append(tmesh.center_mass)
                else:
                    # Still fallback to mean vertex position if no better option...
                    geoms_center.append(np.mean(tmesh.vertices, axis=0))

            kernel_init_geom_fields(
                geoms_pos=np.array([geom.init_pos for geom in geoms], dtype=gs.np_float),
                geoms_center=np.array(geoms_center, dtype=gs.np_float),
                geoms_quat=np.array([geom.init_quat for geom in geoms], dtype=gs.np_float),
                geoms_link_idx=np.array([geom.link.idx for geom in geoms], dtype=gs.np_int),
                geoms_type=np.array([geom.type for geom in geoms], dtype=gs.np_int),
                geoms_friction=np.array([geom.friction for geom in geoms], dtype=gs.np_float),
                geoms_sol_params=geoms_sol_params,
                geoms_vert_start=np.array([geom.vert_start for geom in geoms], dtype=gs.np_int),
                geoms_face_start=np.array([geom.face_start for geom in geoms], dtype=gs.np_int),
                geoms_edge_start=np.array([geom.edge_start for geom in geoms], dtype=gs.np_int),
                geoms_verts_state_start=np.array([geom.verts_state_start for geom in geoms], dtype=gs.np_int),
                geoms_vert_end=np.array([geom.vert_end for geom in geoms], dtype=gs.np_int),
                geoms_face_end=np.array([geom.face_end for geom in geoms], dtype=gs.np_int),
                geoms_edge_end=np.array([geom.edge_end for geom in geoms], dtype=gs.np_int),
                geoms_verts_state_end=np.array([geom.verts_state_end for geom in geoms], dtype=gs.np_int),
                geoms_data=np.array([geom.data for geom in geoms], dtype=gs.np_float),
                geoms_is_convex=np.array([geom.is_convex for geom in geoms], dtype=gs.np_bool),
                geoms_needs_coup=np.array([geom.needs_coup for geom in geoms], dtype=gs.np_int),
                geoms_contype=np.array([geom.contype for geom in geoms], dtype=np.int32),
                geoms_conaffinity=np.array([geom.conaffinity for geom in geoms], dtype=np.int32),
                geoms_coup_softness=np.array([geom.coup_softness for geom in geoms], dtype=gs.np_float),
                geoms_coup_friction=np.array([geom.coup_friction for geom in geoms], dtype=gs.np_float),
                geoms_coup_restitution=np.array([geom.coup_restitution for geom in geoms], dtype=gs.np_float),
                geoms_is_fixed=np.array([geom.is_fixed for geom in geoms], dtype=gs.np_bool),
                geoms_is_decomp=np.array([geom.metadata.get("decomposed", False) for geom in geoms], dtype=gs.np_bool),
                # Quadrants variables
                geoms_info=self.geoms_info,
                geoms_state=self.geoms_state,
                verts_info=self.verts_info,
                geoms_init_AABB=self.geoms_init_AABB,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_entity_fields(self):
        self.entities_info = self.data_manager.entities_info
        self.entities_state = self.data_manager.entities_state

        if self._entities:
            entities = self._entities
            kernel_init_entity_fields(
                entities_dof_start=np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
                entities_dof_end=np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
                entities_link_start=np.array([entity.link_start for entity in entities], dtype=gs.np_int),
                entities_link_end=np.array([entity.link_end for entity in entities], dtype=gs.np_int),
                entities_geom_start=np.array([entity.geom_start for entity in entities], dtype=gs.np_int),
                entities_geom_end=np.array([entity.geom_end for entity in entities], dtype=gs.np_int),
                entities_gravity_compensation=np.array(
                    [entity.gravity_compensation for entity in entities], dtype=gs.np_float
                ),
                entities_is_local_collision_mask=np.array(
                    [entity.is_local_collision_mask for entity in entities], dtype=gs.np_bool
                ),
                # Quadrants variables
                entities_info=self.entities_info,
                entities_state=self.entities_state,
                links_info=self.links_info,
                dofs_info=self.dofs_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_equality_fields(self):
        self.equalities_info = self.data_manager.equalities_info
        if self.n_equalities > 0:
            equalities = self.equalities

            equalities_sol_params = np.array([equality.sol_params for equality in equalities], dtype=gs.np_float)
            _sanitize_sol_params(equalities_sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

            kernel_init_equality_fields(
                equalities_type=np.array([equality.type for equality in equalities], dtype=gs.np_int),
                equalities_eq_obj1id=np.array([equality.eq_obj1id for equality in equalities], dtype=gs.np_int),
                equalities_eq_obj2id=np.array([equality.eq_obj2id for equality in equalities], dtype=gs.np_int),
                equalities_eq_data=np.array([equality.eq_data for equality in equalities], dtype=gs.np_float),
                equalities_eq_type=np.array([equality.type for equality in equalities], dtype=gs.np_int),
                equalities_sol_params=equalities_sol_params,
                # Quadrants variables
                equalities_info=self.equalities_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_collider(self):
        self.collider = Collider(self)

        if self.collider._collider_static_config.has_terrain:
            link_idx_ = next(
                i for i, _type in enumerate(qd_to_numpy(self.geoms_info.type)) if _type == gs.GEOM_TYPE.TERRAIN
            )
            link_idx = qd_to_numpy(self.geoms_info.link_idx, link_idx_, keepdim=False)
            entity_idx = qd_to_numpy(self.links_info.entity_idx, link_idx, keepdim=False)
            if self._options.batch_links_info:
                entity_idx = entity_idx[0]
            entity = self._entities[entity_idx]

            scale = np.asarray(entity.terrain_scale, dtype=gs.np_float)
            rc = np.array(entity.terrain_hf.shape, dtype=gs.np_int)
            hf = entity.terrain_hf.astype(gs.np_float, copy=False) * scale[1]
            xyz_maxmin = np.array(
                [rc[0] * scale[0], rc[1] * scale[0], hf.max(), 0, 0, hf.min() - 1.0],
                dtype=gs.np_float,
            )

            self.terrain_hf = qd.field(dtype=gs.qd_float, shape=hf.shape)
            self.terrain_rc = qd.field(dtype=gs.qd_int, shape=(2,))
            self.terrain_scale = qd.field(dtype=gs.qd_float, shape=(2,))
            self.terrain_xyz_maxmin = qd.field(dtype=gs.qd_float, shape=(6,))

            self.terrain_hf.from_numpy(hf)
            self.terrain_rc.from_numpy(rc)
            self.terrain_scale.from_numpy(scale)
            self.terrain_xyz_maxmin.from_numpy(xyz_maxmin)

    def _init_constraint_solver(self):
        # Islands are a per-island Newton solve inside ConstraintSolver.resolve, gated on use_contact_island.
        self.constraint_solver = ConstraintSolver(self)

    def substep(self, f):
        # from genesis.utils.tools import create_timer
        from genesis.engine.couplers import SAPCoupler

        if self._requires_grad and f == 0:
            kernel_save_adjoint_cache(
                f=f,
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                rigid_adjoint_cache=self._rigid_adjoint_cache,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        kernel_step_1(
            self.links_state,
            self.links_info,
            self.joints_state,
            self.joints_info,
            self.dofs_state,
            self.dofs_info,
            self.geoms_state,
            self.geoms_info,
            self.entities_state,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
            self.constraint_solver.island_state,
            self._is_forward_pos_updated,
            self._is_forward_vel_updated,
            self._is_backward,
        )

        # kernel_step_1 poses only the entity geoms (its FK iterates entity ranges); pose the reserved pool
        # block from its links and refresh its AABB before collision detection reads them.
        self._func_update_pool_geoms()

        if isinstance(self.sim.coupler, SAPCoupler):
            update_qvel(
                self.dofs_state,
                self._rigid_global_info,
                self._static_rigid_sim_config,
                self._is_backward,
            )
        else:
            self._func_constraint_force()
            kernel_step_2(
                self.dofs_state,
                self.dofs_info,
                self.links_info,
                self.links_state,
                self.joints_info,
                self.joints_state,
                self.entities_state,
                self.entities_info,
                self.geoms_info,
                self.geoms_state,
                self.collider._collider_state,
                self._rigid_global_info,
                self._static_rigid_sim_config,
                self.constraint_solver.island_state,
                self._is_backward,
                self._errno,
            )
            self._is_forward_pos_updated = not self._enable_mujoco_compatibility
            self._is_forward_vel_updated = not self._enable_mujoco_compatibility
            if self._requires_grad:
                kernel_save_adjoint_cache(
                    f + 1,
                    self.dofs_state,
                    self._rigid_global_info,
                    self._rigid_adjoint_cache,
                    self._static_rigid_sim_config,
                )

    def get_error_envs_mask(self):
        return qd_to_torch(self._errno) > 0

    def check_errno(self):
        # FIXME: qd.atomic_or return value is broken on Metal — always returns 0.
        # See repro_metal_kernel_return.py. Falling back to numpy reduction.
        if gs.use_zerocopy or sys.platform == "darwin":
            errno = np.bitwise_or.reduce(qd_to_numpy(self._errno))
        else:
            errno = kernel_bit_reduction(self._errno)

        if errno & array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS:
            max_collision_pairs_broad = self.collider._collider_info.max_collision_pairs_broad[None]
            gs.raise_exception(
                f"Exceeding max number of broad phase candidate contact pairs ({max_collision_pairs_broad}). "
                f"Please increase the value of RigidSolver's option 'multiplier_collision_broad_phase'."
            )
        if errno & array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS:
            max_candidate_contacts = self.collider._collider_info.max_candidate_contacts[None]
            gs.raise_exception(
                f"Exceeding max number of candidate contact points ({max_candidate_contacts}). Please increase the "
                "value of RigidSolver's option 'max_collision_pairs'."
            )
        if errno & array_class.ErrorCode.OVERFLOW_CONTACTS:
            max_contacts = self.collider._collider_info.max_contacts[None]
            gs.raise_exception(
                f"Exceeding max number of post-pruning contact points ({max_contacts}) supported by the constraint "
                "solver. Please increase the value of RigidSolver's option 'max_contacts'."
            )
        if errno & array_class.ErrorCode.INVALID_FORCE_NAN:
            gs.raise_exception("Invalid constraint forces causing 'nan'. Please decrease Rigid simulation timestep.")
        if errno & array_class.ErrorCode.INVALID_ACC_NAN:
            gs.raise_exception("Invalid accelerations causing 'nan'. Please decrease Rigid simulation timestep.")
        if errno & array_class.ErrorCode.OVERFLOW_HIBERNATION_ISLANDS:
            gs.raise_exception("Contact island buffer overflow. Please increase RigidOptions 'max_collision_pairs'.")

    def _kernel_detect_collision(self):
        self.collider.clear()
        self.collider.detection()

    def detect_collision(self, env_idx=0):
        # TODO: support batching
        self._kernel_detect_collision()

        n_collision = qd_to_numpy(self.collider._collider_state.n_contacts)[env_idx]
        collision_pairs = np.empty((n_collision, 2), dtype=np.int32)
        collision_pairs[:, 0] = qd_to_numpy(self.collider._collider_state.contact_data.geom_a)[:n_collision, env_idx]
        collision_pairs[:, 1] = qd_to_numpy(self.collider._collider_state.contact_data.geom_b)[:n_collision, env_idx]

        return collision_pairs

    def _func_constraint_force(self):
        if not self._disable_constraint:
            self.constraint_solver.add_equality_constraints()

        if self._enable_collision:
            self.collider.detection()
            # A collision against a sleeping body must wake it before the solve, so it joins the island partition
            # and responds dynamically this step instead of letting the awake body pass through.
            if self._use_hibernation:
                kernel_wake_up_entities_on_new_contact(
                    self.collider._collider_state,
                    self.links_info,
                    self.links_state,
                    self.entities_state,
                    self.entities_info,
                    self.dofs_state,
                    self.geoms_state,
                    self._rigid_global_info,
                    self.constraint_solver.island_state,
                    self._static_rigid_sim_config,
                )

        if not self._disable_constraint:
            self.constraint_solver.add_inequality_constraints()
            self.constraint_solver.resolve(self.entities_info, self._rigid_global_info)

    def _func_forward_dynamics(self):
        kernel_forward_dynamics(
            self.links_state,
            self.links_info,
            self.dofs_state,
            self.dofs_info,
            self.joints_info,
            self.entities_state,
            self.entities_info,
            self.geoms_state,
            self._rigid_global_info,
            self._static_rigid_sim_config,
            self.constraint_solver.island_state,
        )

    def _func_update_acc(self):
        kernel_update_acc(
            self.dofs_state,
            self.links_info,
            self.links_state,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

    def _func_forward_kinematics_entity(self, i_e, envs_idx):
        kernel_forward_kinematics_entity(
            i_e,
            envs_idx,
            self.links_state,
            self.links_info,
            self.joints_state,
            self.joints_info,
            self.dofs_state,
            self.dofs_info,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

    def _func_integrate_dq_entity(self, dq, i_e, i_b, respect_joint_limit):
        func_integrate_dq_entity(
            dq,
            i_e,
            i_b,
            respect_joint_limit,
            links_info=self.links_info,
            joints_info=self.joints_info,
            dofs_info=self.dofs_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def _func_update_geoms(self, envs_idx, *, force_update_fixed_geoms=False):
        kernel_update_geoms(
            envs_idx,
            self.entities_info,
            self.geoms_state,
            self.geoms_info,
            self.links_state,
            self._rigid_global_info,
            self._static_rigid_sim_config,
            force_update_fixed_geoms,
        )
        # The reserved geometry-pool block lies outside every entity's contiguous geom range, so the
        # entity-based pass above skips it. Pose those geoms from their owning link (set at upload; link 0 and
        # inert while a slot is empty). Small extra pass, only when a pool is reserved.
        self._func_update_pool_geoms()

    def get_entity_aabb_per_env(self, entity, envs_idx=None):
        """Per-env AABB of a geometry-pool entity, reduced over the geoms active in each env.

        The entity's per-env active geoms live in `links_info.geom_start/end` (the bound pool slot for its base
        link, static ranges for any other link), so aggregate the device `geoms_state.aabb_min/max` over those
        ranges rather than the entity's contiguous build-time range.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        aabb = torch.empty((len(envs_idx), 2, 3), dtype=gs.tc_float, device=gs.device)
        kernel_entity_aabb_per_env(
            entity._link_start,
            entity._link_start + entity.n_links,
            tensor_to_array(envs_idx),
            self.links_info,
            self.geoms_state,
            aabb,
            self._static_rigid_sim_config,
        )
        return aabb

    def _init_pool_vgeoms(self):
        """Create placeholder visual geoms for reserved vgeom slots (visual pooling) and seed their device rows.

        A per-slot RigidVisGeom placeholder (host trimesh, swapped on set_active_object) is appended to its
        entity's vgeom list so the rasterizer renders it; per-env visibility is driven by its active_envs_mask
        (all-False until a slot is bound to an env). Rigid visual pooling uses the host trimesh + the device
        vgeom pose, so no device vvert data is needed. No-op when no slot reserves visual geometry.
        """
        import genesis.utils.mesh as mu
        from genesis.engine.entities.rigid_entity.rigid_geom import RigidVisGeom

        if self._geom_pool is None or self._geom_pool.n_vgeoms == 0:
            return

        placeholder_mesh = gs.Mesh.from_trimesh(
            mu.create_box(extents=(1e-3, 1e-3, 1e-3)), surface=gs.surfaces.Default()
        )
        vgeom_idx_all, link_idx_all = [], []
        for seg in self._geom_pool.segments:
            if seg.per_slot["vgeom"] == 0:
                continue
            base_link = seg.entity.base_link
            # Make the entity's base (morph) vgeoms maskable so they can be hidden per env once a pooled object
            # is shown there. An all-True mask renders everywhere (visible) but via per-env nodes on_rigid can
            # later cull; without a mask on_rigid would build an env-shared node that cannot be hidden per env.
            for base_vgeom in seg.entity.vgeoms:
                if base_vgeom.active_envs_mask is None:
                    base_vgeom.active_envs_mask = torch.ones(self._B, dtype=torch.bool, device=gs.device)
                    (base_vgeom.active_envs_idx,) = np.where(tensor_to_array(base_vgeom.active_envs_mask))
            for slot in range(seg.n_slots):
                vgeom_range = seg.slots[slot].vgeom
                # One placeholder per slot (the object's merged visual mesh); extra reserved vgeom rows stay
                # inert. It renders nowhere until set_active_object sets its mesh + active_envs.
                placeholder = RigidVisGeom(
                    base_link, vgeom_range[0], 0, 0, placeholder_mesh, gu.zero_pos(), gu.identity_quat()
                )
                placeholder._is_pool_slot = True
                placeholder.active_envs_mask = torch.zeros(self._B, dtype=torch.bool, device=gs.device)
                placeholder.active_envs_idx = np.zeros(0, dtype=gs.np_int)
                seg.vgeom_placeholders[slot] = placeholder
                seg.entity._vgeoms.append(placeholder)
                for i in range(vgeom_range[0], vgeom_range[1]):
                    vgeom_idx_all.append(i)
                    link_idx_all.append(base_link.idx)

        kernel_init_pool_vgeoms(
            np.array(vgeom_idx_all, dtype=gs.np_int),
            np.array(link_idx_all, dtype=gs.np_int),
            self.vgeoms_info,
            self.vgeoms_state,
            self._static_rigid_sim_config,
        )

    def _func_update_pool_geoms(self):
        """Pose the reserved geometry-pool block from its links and refresh its AABBs (no-op without a pool)."""
        if self._enable_geom_pool:
            kernel_update_pool_geoms(
                self._n_geoms,
                self.n_geoms_,
                self.geoms_state,
                self.geoms_info,
                self.geoms_init_AABB,
                self.links_state,
                self._static_rigid_sim_config,
            )

    def apply_links_external_force(
        self,
        force,
        links_idx=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        local: bool = False,
    ):
        """
        Apply some external linear force on a set of links.

        Parameters
        ----------
        force : array_like
            The force to apply.
        links_idx : None | array_like, optional
            The indices of the links on which to apply force. None to specify all links. Default to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: "link_origin" | "link_com" | "root_com", optional
            The reference frame on which the linear force will be applied. "link_origin" refers to the origin of the
            link, "link_com" refers to the center of mass of the link, and "root_com" refers to the center of mass of
            the entire kinematic tree to which a link belong (see `get_links_root_COM` for details).
        local: bool, optional
            Whether the force is expressed in the local coordinates associated with the reference frame instead of
            world frame. Only supported for `ref="link_origin"` or `ref="link_com"`.
        """
        force, links_idx, envs_idx = self._sanitize_io_variables(
            force, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            force = force[None]

        if ref == "root_com" and local:
            raise ValueError("'local=True' not compatible with ref='root_com'.")
        ref_idx = self._convert_ref_to_idx(ref)

        # A force on a sleeping body must revive it, otherwise the input is silently dropped.
        if self._use_hibernation:
            kernel_wake_up_entities_by_links(
                links_idx,
                envs_idx,
                links_info=self.links_info,
                links_state=self.links_state,
                entities_state=self.entities_state,
                entities_info=self.entities_info,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                island_state=self.constraint_solver.island_state,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        kernel_apply_links_external_force(
            force, links_idx, envs_idx, ref_idx, 1 if local else 0, self.links_state, self._static_rigid_sim_config
        )

    def apply_links_external_torque(
        self,
        torque,
        links_idx=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        local: bool = False,
    ):
        """
        Apply some external torque on a set of links.

        Parameters
        ----------
        torque : array_like
            The torque to apply.
        links_idx : None | array_like, optional
            The indices of the links on which to apply torque. None to specify all links. Default to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: "link_origin" | "link_com" | "root_com", optional
            The reference frame on which the torque will be applied. "link_origin" refers to the origin of the link,
            "link_com" refers to the center of mass of the link, and "root_com" refers to the center of mass of
            the entire kinematic tree to which a link belong (see `get_links_root_COM` for details). Note that this
            argument has no effect unless `local=True`.
        local: bool, optional
            Whether the torque is expressed in the local coordinates associated with the reference frame instead of
            world frame. Only supported for `ref="link_origin"` or `ref="link_com"`.
        """
        torque, links_idx, envs_idx = self._sanitize_io_variables(
            torque, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            torque = torque[None]

        if ref == "root_com" and local:
            raise ValueError("'local=True' not compatible with ref='root_com'.")
        ref_idx = self._convert_ref_to_idx(ref)

        # A torque on a sleeping body must revive it, otherwise the input is silently dropped.
        if self._use_hibernation:
            kernel_wake_up_entities_by_links(
                links_idx,
                envs_idx,
                links_info=self.links_info,
                links_state=self.links_state,
                entities_state=self.entities_state,
                entities_info=self.entities_info,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                island_state=self.constraint_solver.island_state,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        kernel_apply_links_external_torque(
            torque, links_idx, envs_idx, ref_idx, 1 if local else 0, self.links_state, self._static_rigid_sim_config
        )

    def substep_pre_coupling(self, f):
        if self.is_active:
            # Skip rigid body computation when using IPCCoupler (IPC handles rigid simulation)
            from genesis.engine.couplers import IPCCoupler

            if isinstance(self.sim.coupler, IPCCoupler):
                # If any rigid entity is coupled to IPC, skip pre-coupling rigid simulation
                # The rigid simulation will be done in post-coupling phase instead
                if self.sim.coupler.has_any_rigid_coupling:
                    return

            # Run Genesis rigid simulation step for non-IPC couplers
            self.substep(f)

    def reset_grad(self):
        # Rigid additionally owns `geoms_state`, `entities_state`, and the `*_adjoint_cache` structs written by the
        # backward substep chain. All carry `needs_grad=True` fields that accumulate via `atomic_add` during backward,
        # so they must start at zero between consecutive `loss.backward()`s.
        super().reset_grad()
        if self._requires_grad:
            qd_zero_grad(self.geoms_state)
            qd_zero_grad(self.entities_state)
            qd_zero_grad(self.dofs_state_adjoint_cache)
            qd_zero_grad(self.links_state_adjoint_cache)
            qd_zero_grad(self.joints_state_adjoint_cache)
            qd_zero_grad(self.geoms_state_adjoint_cache)
            qd_zero_grad(self._rigid_adjoint_cache)

    def substep_pre_coupling_grad(self, f):
        # Change to backward mode
        self._is_backward = True

        # Run forward substep again to restore this step's information, this is needed because we do not store info
        # of every substep.
        kernel_prepare_backward_substep(
            f=f,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            dofs_state_adjoint_cache=self.dofs_state_adjoint_cache,
            links_state_adjoint_cache=self.links_state_adjoint_cache,
            joints_state_adjoint_cache=self.joints_state_adjoint_cache,
            geoms_state_adjoint_cache=self.geoms_state_adjoint_cache,
            rigid_adjoint_cache=self._rigid_adjoint_cache,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self.substep(f)

        # =================== Backward substep ======================
        envs_idx = self._scene._sanitize_envs_idx(None)
        if not self._enable_mujoco_compatibility:
            kernel_forward_velocity.grad(
                envs_idx=envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                is_backward=True,
            )
            kernel_update_cartesian_space.grad(
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                force_update_fixed_geoms=False,
                is_backward=True,
            )

        is_grad_valid = kernel_begin_backward_substep(
            f=f,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            dofs_state_adjoint_cache=self.dofs_state_adjoint_cache,
            links_state_adjoint_cache=self.links_state_adjoint_cache,
            joints_state_adjoint_cache=self.joints_state_adjoint_cache,
            geoms_state_adjoint_cache=self.geoms_state_adjoint_cache,
            rigid_adjoint_cache=self._rigid_adjoint_cache,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        if not is_grad_valid:
            gs.raise_exception(f"Nan grad in qpos or dofs_vel found at step {self._sim.cur_step_global}")

        kernel_step_2.grad(
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            links_info=self.links_info,
            links_state=self.links_state,
            joints_info=self.joints_info,
            joints_state=self.joints_state,
            entities_state=self.entities_state,
            entities_info=self.entities_info,
            geoms_info=self.geoms_info,
            geoms_state=self.geoms_state,
            collider_state=self.collider._collider_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            island_state=self.constraint_solver.island_state,
            is_backward=True,
            errno=self._errno,
        )

        # We cannot use [kernel_forward_dynamics.grad] because we read [dofs_state.acc] and overwrite it in the kernel,
        # which is prohibited (https://docs.taichi-lang.org/docs/differentiable_programming#global-data-access-rules).
        # In [kernel_forward_dynamics], we read [acc] in [func_update_acc] and overwrite it in [kernel_compute_qacc].
        # As [kenrel_compute_qacc] is called at the end of [kernel_forward_dynamics], we first backpropagate through
        # [kernel_compute_qacc] and then restore the original [acc] from the adjoint cache. This copy operation
        # cannot be merged with [kernel_compute_qacc.grad] because .grad function itself is a standalone kernel.
        # We could possibly merge this small kernel later if (1) .grad function is regarded as a function instead of a
        # kernel, (2) we add another variable to store the new [acc] from [kernel_compute_qacc] and thus can avoid
        # the data access violation. However, both of these require major changes.
        kernel_compute_qacc.grad(
            dofs_state=self.dofs_state,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            is_backward=True,
        )
        kernel_copy_acc(
            f=f,
            dofs_state=self.dofs_state,
            rigid_adjoint_cache=self._rigid_adjoint_cache,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

        kernel_forward_dynamics_without_qacc.grad(
            links_state=self.links_state,
            links_info=self.links_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            joints_info=self.joints_info,
            entities_state=self.entities_state,
            entities_info=self.entities_info,
            geoms_state=self.geoms_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            island_state=self.constraint_solver.island_state,
            is_backward=True,
        )

        # If it was the very first substep, we need to backpropagate through the initial update of the cartesian space
        if self._enable_mujoco_compatibility or self._sim.cur_substep_global == 0:
            kernel_forward_velocity.grad(
                envs_idx=envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                is_backward=True,
            )
            kernel_update_cartesian_space.grad(
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                force_update_fixed_geoms=False,
                is_backward=True,
            )

        # Change back to forward mode
        self._is_backward = False

    def substep_post_coupling(self, f):
        from genesis.engine.couplers import SAPCoupler, IPCCoupler

        if not self.is_active:
            return

        if isinstance(self.sim.coupler, SAPCoupler):
            update_qacc_from_qvel_delta(
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                is_backward=self._is_backward,
            )
            kernel_step_2(
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                links_info=self.links_info,
                links_state=self.links_state,
                joints_info=self.joints_info,
                joints_state=self.joints_state,
                entities_state=self.entities_state,
                entities_info=self.entities_info,
                geoms_info=self.geoms_info,
                geoms_state=self.geoms_state,
                collider_state=self.collider._collider_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                island_state=self.constraint_solver.island_state,
                is_backward=self._is_backward,
                errno=self._errno,
            )
        elif isinstance(self.sim.coupler, IPCCoupler):
            # If any rigid entity is coupled to IPC, perform rigid simulation in post-coupling phase.
            # Collision exclusion for IPC-coupled links is handled in the collider at build time.
            if self.sim.coupler.has_any_rigid_coupling:
                self.substep(f)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- render -----------------------------------------
    # ------------------------------------------------------------------------------------

    def update_geoms_render_T(self):
        kernel_update_geoms_render_T(
            self._geoms_render_T,
            geoms_state=self.geoms_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    # ------------------------------------------------------------------------------------
    # -------------------------------- state get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f=None):
        s_global = self.sim.cur_step_global
        if self.is_active:
            if s_global in self._queried_states:
                return self._queried_states[s_global][0]

            state = RigidSolverState(self._scene, s_global)

            kernel_get_state(
                qpos=state.qpos,
                vel=state.dofs_vel,
                acc=state.dofs_acc,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                mass_shift=state.mass_shift,
                friction_ratio=state.friction_ratio,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._queried_states.append(state)
        else:
            state = None
        return state

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None, *, partial: bool = False) -> None:
        if not self.is_active:
            return

        if partial:
            self.collider.reset(envs_idx)
            self.constraint_solver.reset(envs_idx)
        else:
            self.collider.clear(envs_idx)
            self.constraint_solver.clear(envs_idx)

        if (
            not self._requires_grad
            and gs.use_zerocopy
            and (not isinstance(envs_idx, torch.Tensor) or (not IS_OLD_TORCH or envs_idx.dtype == torch.bool))
        ):
            errno = qd_to_torch(self._errno, copy=False)
            qpos_dst = qd_to_torch(self._rigid_global_info.qpos, transpose=True, copy=False)
            vel_dst = qd_to_torch(self.dofs_state.vel, transpose=True, copy=False)
            acc_dst = qd_to_torch(self.dofs_state.acc, transpose=True, copy=False)
            ctrl_force_dst = qd_to_torch(self.dofs_state.ctrl_force, transpose=True, copy=False)
            ctrl_mode_dst = qd_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            pos_dst = qd_to_torch(self.links_state.pos, transpose=True, copy=False)
            quat_dst = qd_to_torch(self.links_state.quat, transpose=True, copy=False)
            shift_dst = qd_to_torch(self.links_state.i_pos_shift, transpose=True, copy=False)
            cfrc_vel_dst = qd_to_torch(self.links_state.cfrc_applied_vel, transpose=True, copy=False)
            cfrc_ang_dst = qd_to_torch(self.links_state.cfrc_applied_ang, transpose=True, copy=False)
            mass_dst = qd_to_torch(self.links_state.mass_shift, transpose=True, copy=False)
            fric_dst = qd_to_torch(self.geoms_state.friction_ratio, transpose=True, copy=False)
            # Setting the state is a discontinuity: wake every body in the affected envs (a body left hibernated would
            # stay frozen), restoring the flags and the compact awake lists alongside the other state buffers.
            if self._use_hibernation:
                links_hibernated_dst = qd_to_torch(self.links_state.is_hibernated, transpose=True, copy=False)
                awake_steps_dst = qd_to_torch(self.links_state.awake_steps, transpose=True, copy=False)
                dofs_hibernated_dst = qd_to_torch(self.dofs_state.is_hibernated, transpose=True, copy=False)
                geoms_hibernated_dst = qd_to_torch(self.geoms_state.is_hibernated, transpose=True, copy=False)
                entities_hibernated_dst = qd_to_torch(self.entities_state.is_hibernated, transpose=True, copy=False)
                islands_hibernated_dst = qd_to_torch(
                    self.constraint_solver.island_state.is_hibernated, transpose=True, copy=False
                )
                islands_next_link_dst = qd_to_torch(
                    self.constraint_solver.island_state.hibernated_next_link, transpose=True, copy=False
                )
                awake_links_dst = qd_to_torch(self._rigid_global_info.awake_links, transpose=True, copy=False)
                awake_dofs_dst = qd_to_torch(self._rigid_global_info.awake_dofs, transpose=True, copy=False)
                awake_entities_dst = qd_to_torch(self._rigid_global_info.awake_entities, transpose=True, copy=False)
                n_awake_links_dst = qd_to_torch(self._rigid_global_info.n_awake_links, copy=False)
                n_awake_dofs_dst = qd_to_torch(self._rigid_global_info.n_awake_dofs, copy=False)
                n_awake_entities_dst = qd_to_torch(self._rigid_global_info.n_awake_entities, copy=False)
                # Fill to the padded buffer capacity but keep n_awake at the real count below, so a scene with no
                # DOFs writes its padded slot yet reports zero awake DOFs.
                awake_links_src = torch.arange(self.n_links_, device=gs.device, dtype=gs.tc_int)
                awake_dofs_src = torch.arange(self.n_dofs_, device=gs.device, dtype=gs.tc_int)
                awake_entities_src = torch.arange(self.n_entities_, device=gs.device, dtype=gs.tc_int)

            if envs_idx is not None and not isinstance(envs_idx, torch.Tensor):
                (envs_idx,) = indices_to_mask(envs_idx)
            if isinstance(envs_idx, torch.Tensor):
                if envs_idx.dtype == torch.bool:
                    envs_mask = envs_idx
                else:
                    envs_mask = torch.zeros(self._B, dtype=torch.bool, device=gs.device)
                    envs_mask[envs_idx] = True

                errno.masked_fill_(envs_mask, 0)
                if self.n_qs:
                    torch.where(envs_mask[:, None], state.qpos, qpos_dst, out=qpos_dst)
                    torch.where(envs_mask[:, None], state.dofs_vel, vel_dst, out=vel_dst)
                    torch.where(envs_mask[:, None], state.dofs_acc, acc_dst, out=acc_dst)
                    ctrl_force_dst.masked_fill_(envs_mask[:, None], 0.0)
                    ctrl_mode_dst.masked_fill_(envs_mask[:, None], gs.CTRL_MODE.FORCE)
                torch.where(envs_mask[:, None, None], state.links_pos, pos_dst, out=pos_dst)
                torch.where(envs_mask[:, None, None], state.links_quat, quat_dst, out=quat_dst)
                torch.where(envs_mask[:, None, None], state.i_pos_shift, shift_dst, out=shift_dst)
                cfrc_vel_dst.masked_fill_(envs_mask[:, None, None], 0.0)
                cfrc_ang_dst.masked_fill_(envs_mask[:, None, None], 0.0)
                torch.where(envs_mask[:, None], state.mass_shift, mass_dst, out=mass_dst)
                if self.n_geoms:
                    torch.where(envs_mask[:, None], state.friction_ratio, fric_dst, out=fric_dst)
                if self._use_hibernation:
                    links_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    awake_steps_dst.masked_fill_(envs_mask[:, None], 0)
                    dofs_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    geoms_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    entities_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    islands_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    islands_next_link_dst.masked_fill_(envs_mask[:, None], -1)
                    torch.where(envs_mask[:, None], awake_links_src, awake_links_dst, out=awake_links_dst)
                    torch.where(envs_mask[:, None], awake_dofs_src, awake_dofs_dst, out=awake_dofs_dst)
                    torch.where(envs_mask[:, None], awake_entities_src, awake_entities_dst, out=awake_entities_dst)
                    n_awake_links_dst.masked_fill_(envs_mask, self.n_links)
                    n_awake_dofs_dst.masked_fill_(envs_mask, self.n_dofs)
                    n_awake_entities_dst.masked_fill_(envs_mask, self.n_entities)
            else:
                if self.n_qs:
                    errno[envs_idx] = 0
                    qpos_dst[envs_idx] = state.qpos[envs_idx]
                    vel_dst[envs_idx] = state.dofs_vel[envs_idx]
                    acc_dst[envs_idx] = state.dofs_acc[envs_idx]
                    ctrl_force_dst[envs_idx] = 0.0
                    ctrl_mode_dst[envs_idx] = gs.CTRL_MODE.FORCE
                pos_dst[envs_idx] = state.links_pos[envs_idx]
                quat_dst[envs_idx] = state.links_quat[envs_idx]
                shift_dst[envs_idx] = state.i_pos_shift[envs_idx]
                cfrc_vel_dst[envs_idx] = 0.0
                cfrc_ang_dst[envs_idx] = 0.0
                mass_dst[envs_idx] = state.mass_shift[envs_idx]
                if self.n_geoms:
                    fric_dst[envs_idx] = state.friction_ratio[envs_idx]
                if self._use_hibernation:
                    links_hibernated_dst[envs_idx] = 0
                    awake_steps_dst[envs_idx] = 0
                    dofs_hibernated_dst[envs_idx] = 0
                    geoms_hibernated_dst[envs_idx] = 0
                    entities_hibernated_dst[envs_idx] = 0
                    islands_hibernated_dst[envs_idx] = 0
                    islands_next_link_dst[envs_idx] = -1
                    awake_links_dst[envs_idx] = awake_links_src
                    awake_dofs_dst[envs_idx] = awake_dofs_src
                    awake_entities_dst[envs_idx] = awake_entities_src
                    n_awake_links_dst[envs_idx] = self.n_links
                    n_awake_dofs_dst[envs_idx] = self.n_dofs
                    n_awake_entities_dst[envs_idx] = self.n_entities
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            kernel_set_zero(envs_idx, self._errno)
            kernel_set_state(
                envs_idx=envs_idx,
                qpos=state.qpos,
                dofs_vel=state.dofs_vel,
                dofs_acc=state.dofs_acc,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                mass_shift=state.mass_shift,
                friction_ratio=state.friction_ratio,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            if self._use_hibernation:
                kernel_reset_hibernation(
                    envs_idx,
                    links_info=self.links_info,
                    links_state=self.links_state,
                    dofs_state=self.dofs_state,
                    geoms_state=self.geoms_state,
                    entities_state=self.entities_state,
                    island_state=self.constraint_solver.island_state,
                    rigid_global_info=self._rigid_global_info,
                    static_rigid_sim_config=self._static_rigid_sim_config,
                )

        if not partial:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

        for entity in self.entities:
            if isinstance(entity, DroneEntity):
                entity._prev_prop_t = -1

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities:
            entity.process_input_grad()

    def save_ckpt(self, ckpt_name):
        # Save ckpt only if we need gradients, because this operation is costly
        if self._requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()

            self._ckpt[ckpt_name]["qpos"] = qd_to_numpy(self._rigid_adjoint_cache.qpos)
            self._ckpt[ckpt_name]["dofs_vel"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_vel)
            self._ckpt[ckpt_name]["dofs_acc"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_acc)

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

    def load_ckpt(self, ckpt_name):
        # Set first frame
        self._rigid_global_info.qpos.from_numpy(self._ckpt[ckpt_name]["qpos"][0])
        self.dofs_state.vel.from_numpy(self._ckpt[ckpt_name]["dofs_vel"][0])
        self.dofs_state.acc.from_numpy(self._ckpt[ckpt_name]["dofs_acc"][0])

        if not self._enable_mujoco_compatibility:
            kernel_update_cartesian_space(
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                force_update_fixed_geoms=False,
                is_backward=False,
            )

        for entity in self._entities:
            entity.load_ckpt(ckpt_name)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def set_links_pos(self, pos, links_idx=None, envs_idx=None):
        raise DeprecationError("This method has been removed. Please use 'set_base_links_pos' instead.")

    @mutates(StateChange.GEOMETRY)
    def set_base_links_pos(self, pos, links_idx=None, envs_idx=None, *, relative=False, skip_forward=False):
        if links_idx is None:
            links_idx = self._base_links_idx

        # Without any pose offset, the user and world frames coincide, so a relative set is just an absolute one.
        if relative and self._links_offset_pos is None:
            relative = False

        # Map a single base link's user position to world here (keeping the current orientation) so the zero-copy
        # in-place write below still applies, both for an environment-uniform and a per-environment offset. Multi-link
        # relative sets are composed in the kernel branch instead.
        if relative and isinstance(links_idx, int):
            cur_quat = self.get_links_quat(links_idx, envs_idx, relative=False)[..., 0, :]
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)[..., 0, :]
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)[..., 0, :]
            pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device) + _offset_world_shift(
                offset_pos, offset_quat, cur_quat
            )
            relative = False

        # Zero-copy fast path: single base link, non-relative. Write the position buffer in place instead of
        # launching a kernel. The kernel path below handles relative or multi-link updates, as well as waking up
        # hibernated entities, which is required whenever hibernation is enabled.
        if gs.use_zerocopy and not relative and isinstance(links_idx, int) and not self._use_hibernation:
            link = self.links[links_idx]
            if link.is_fixed:
                data = qd_to_torch(self.links_state.pos, transpose=True, copy=False)
                target = data[:, links_idx]
            else:
                data = qd_to_torch(self._rigid_global_info.qpos, transpose=True, copy=False)
                target = data[:, link.q_start : link.q_start + 3]
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                if pos.ndim == 2 and len(pos) not in (1, len(target)):
                    # A fresh source view is needed because masked_scatter_ may reshape it in-place. Metal
                    # mis-scatters a stride-0 broadcast mask, so it must be materialized to a dense mask there.
                    envs_mask = envs_idx[:, None]
                    if gs.backend == gs.metal:
                        envs_mask = envs_mask.expand_as(target).contiguous()
                    target.masked_scatter_(envs_mask, pos.view_as(pos))
                else:
                    pos = broadcast_tensor(pos, gs.tc_float, target.shape)
                    torch.where(envs_idx[:, None], pos, target, out=target)
            else:
                # Fixed links with at least one geom and non-batched vertices cannot take env-specific positions
                if link.is_fixed and link.geoms and not link.entity._batch_fixed_verts:
                    pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
                    same_pos = pos.ndim < 2 or len(pos) == 1 or (torch.diff(pos, dim=0).abs() < gs.EPS).all()
                    set_all_envs = envs_idx is None or torch.equal(
                        torch.sort(self._scene._sanitize_envs_idx(envs_idx)).values, self._scene._envs_idx
                    )
                    if not (set_all_envs and same_pos):
                        gs.raise_exception(
                            "Specifying env-specific pos for fixed links with at least one geometry requires "
                            "setting morph option 'batch_fixed_verts=True'."
                        )
                mask = (0,) if self.n_envs == 0 else indices_to_mask(envs_idx)
                assign_indexed_tensor(target, mask, pos)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            pos, links_idx, envs_idx = self._sanitize_io_variables(
                pos, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
            if self.n_envs == 0:
                pos = pos[None]

            if relative:
                # Compose the body-frame offset onto the user position, keeping the current orientation, then set the
                # resulting world position absolutely.
                cur_quat = qd_to_torch(self.links_state.quat, envs_idx, links_idx, transpose=True, copy=True)
                offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                pos = pos + _offset_world_shift(offset_pos, offset_quat, cur_quat)
                relative = False

            # Raise exception for fixed links with at least one geom and non-batched fixed vertices, except if setting
            # same location for all envs at once
            set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
            has_fixed_verts = any(
                link.is_fixed and link.geoms and not link.entity._batch_fixed_verts
                for link in (self.links[i_l] for i_l in links_idx)
            )
            if has_fixed_verts and not (set_all_envs and (torch.diff(pos, dim=0).abs() < gs.EPS).all()):
                gs.raise_exception(
                    "Specifying env-specific pos for fixed links with at least one geometry requires setting morph "
                    "option 'batch_fixed_verts=True'."
                )

            # Wake up hibernated entities before setting position (fixed links don't need wake-up)
            if self._use_hibernation and not all(self.links[i_l].is_fixed for i_l in links_idx):
                kernel_wake_up_entities_by_links(
                    links_idx,
                    envs_idx,
                    links_info=self.links_info,
                    links_state=self.links_state,
                    entities_state=self.entities_state,
                    entities_info=self.entities_info,
                    dofs_state=self.dofs_state,
                    geoms_state=self.geoms_state,
                    rigid_global_info=self._rigid_global_info,
                    island_state=self.constraint_solver.island_state,
                    static_rigid_sim_config=self._static_rigid_sim_config,
                )

            kernel_set_links_pos(
                pos,
                links_idx,
                envs_idx,
                links_info=self.links_info,
                links_state=self.links_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def set_links_quat(self, quat, links_idx=None, envs_idx=None):
        raise DeprecationError("This method has been removed. Please use 'set_base_links_quat' instead.")

    @mutates(StateChange.GEOMETRY)
    def set_base_links_quat(self, quat, links_idx=None, envs_idx=None, *, relative=False, skip_forward=False):
        if links_idx is None:
            links_idx = self._base_links_idx

        # Without any pose offset, the user and world frames coincide, so a relative set is just an absolute one.
        if relative and self._links_offset_quat is None:
            relative = False

        # Computed once and reused by the int fast path and the kernel branch below.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        relative_pos_passthrough = relative and self._links_offset_pos_is_identity[idx].all()

        # Compose a single base link's user orientation to world here so the zero-copy in-place write below still
        # applies, both for an environment-uniform and a per-environment offset. This only preserves the user-frame
        # position when the offset position is identity; a non-zero offset position rotates with the orientation and
        # is handled (together with multi-link relative sets) in the kernel branch instead.
        if isinstance(links_idx, int) and relative_pos_passthrough:
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)[..., 0, :]
            quat = gu.transform_quat_by_quat(offset_quat, torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device))
            relative = False

        # Zero-copy fast path: single base link, non-relative. Write the quaternion buffer in place instead of
        # launching a kernel. The kernel path below handles relative or multi-link updates, as well as waking up
        # hibernated entities, which is required whenever hibernation is enabled.
        if gs.use_zerocopy and not relative and isinstance(links_idx, int) and not self._use_hibernation:
            link = self.links[links_idx]
            if link.is_fixed:
                data = qd_to_torch(self.links_state.quat, transpose=True, copy=False)
                target = data[:, links_idx]
            else:
                data = qd_to_torch(self._rigid_global_info.qpos, transpose=True, copy=False)
                target = data[:, link.q_start + 3 : link.q_start + 7]
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                if quat.ndim == 2 and len(quat) not in (1, len(target)):
                    # A fresh source view is needed because masked_scatter_ may reshape it in-place. Metal
                    # mis-scatters a stride-0 broadcast mask, so it must be materialized to a dense mask there.
                    envs_mask = envs_idx[:, None]
                    if gs.backend == gs.metal:
                        envs_mask = envs_mask.expand_as(target).contiguous()
                    target.masked_scatter_(envs_mask, quat.view_as(quat))
                else:
                    quat = broadcast_tensor(quat, gs.tc_float, target.shape)
                    torch.where(envs_idx[:, None], quat, target, out=target)
            else:
                # Fixed links with at least one geom and non-batched vertices cannot take env-specific orientations
                if link.is_fixed and link.geoms and not link.entity._batch_fixed_verts:
                    quat = torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device)
                    same_quat = quat.ndim < 2 or len(quat) == 1 or (torch.diff(quat, dim=0).abs() < gs.EPS).all()
                    set_all_envs = envs_idx is None or torch.equal(
                        torch.sort(self._scene._sanitize_envs_idx(envs_idx)).values, self._scene._envs_idx
                    )
                    if not (set_all_envs and same_quat):
                        gs.raise_exception(
                            "Impossible to set env-specific quat for fixed links with at least one geometry."
                        )
                mask = (0,) if self.n_envs == 0 else indices_to_mask(envs_idx)
                assign_indexed_tensor(target, mask, quat)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            quat, links_idx, envs_idx = self._sanitize_io_variables(
                quat, links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
            )
            if self.n_envs == 0:
                quat = quat[None]

            if relative:
                offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                if not relative_pos_passthrough:
                    # The offset position rotates with the orientation, so keep the user-frame position fixed by
                    # rewriting the world position from the current user position and the new user orientation.
                    cur_pos = qd_to_torch(self.links_state.pos, envs_idx, links_idx, transpose=True, copy=True)
                    cur_quat = qd_to_torch(self.links_state.quat, envs_idx, links_idx, transpose=True, copy=True)
                    offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                    user_pos = cur_pos - _offset_world_shift(offset_pos, offset_quat, cur_quat)
                    world_pos = user_pos + gu.transform_by_quat(offset_pos, quat)
                    kernel_set_links_pos(
                        world_pos,
                        links_idx,
                        envs_idx,
                        links_info=self.links_info,
                        links_state=self.links_state,
                        rigid_global_info=self._rigid_global_info,
                        static_rigid_sim_config=self._static_rigid_sim_config,
                    )
                # Compose the offset onto the user orientation, then set the resulting world orientation absolutely.
                quat = gu.transform_quat_by_quat(offset_quat, quat)
                relative = False

            set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
            has_fixed_verts = any(
                link.is_fixed and link.geoms and not link.entity._batch_fixed_verts
                for link in (self.links[i_l] for i_l in links_idx)
            )
            if has_fixed_verts and not (set_all_envs and (torch.diff(quat, dim=0).abs() < gs.EPS).all()):
                gs.raise_exception("Impossible to set env-specific quat for fixed links with at least one geometry.")

            # Wake up hibernated entities before setting quaternion (fixed links don't need wake-up)
            if self._use_hibernation and not all(self.links[i_l].is_fixed for i_l in links_idx):
                kernel_wake_up_entities_by_links(
                    links_idx,
                    envs_idx,
                    links_info=self.links_info,
                    links_state=self.links_state,
                    entities_state=self.entities_state,
                    entities_info=self.entities_info,
                    dofs_state=self.dofs_state,
                    geoms_state=self.geoms_state,
                    rigid_global_info=self._rigid_global_info,
                    island_state=self.constraint_solver.island_state,
                    static_rigid_sim_config=self._static_rigid_sim_config,
                )

            kernel_set_links_quat(
                quat,
                links_idx,
                envs_idx,
                links_info=self.links_info,
                links_state=self.links_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def set_links_mass_shift(self, mass, links_idx=None, envs_idx=None):
        mass, links_idx, envs_idx = self._sanitize_io_variables(
            mass, links_idx, self.n_links, "links_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            mass = mass[None]
        kernel_set_links_mass_shift(
            mass,
            links_idx,
            envs_idx,
            links_state=self.links_state,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def set_links_COM_shift(self, com, links_idx=None, envs_idx=None):
        com, links_idx, envs_idx = self._sanitize_io_variables(
            com, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            com = com[None]
        kernel_set_links_COM_shift(com, links_idx, envs_idx, self.links_state, self._static_rigid_sim_config)

    def set_links_inertial_mass(self, mass, links_idx=None, envs_idx=None):
        mass, links_idx, envs_idx = self._sanitize_io_variables(
            mass,
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
            batched=self._options.batch_links_info,
            skip_allocation=True,
        )
        if self.n_envs == 0 and self._options.batch_links_info:
            mass = mass[None]
        kernel_set_links_inertial_mass(mass, links_idx, envs_idx, self.links_info, self._static_rigid_sim_config)

    def set_links_inertia(self, ratio, links_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mass_data = qd_to_torch(self.links_info.inertial_mass, transpose=True, copy=False)
            inertial_i_data = qd_to_torch(self.links_info.inertial_i, transpose=True, copy=False)
            invweight_data = qd_to_torch(self.links_info.invweight, transpose=True, copy=False)
            links_mask = indices_to_mask(links_idx)
            if self._options.batch_links_info:
                mask = (0, *links_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *links_mask)
            else:
                mask = links_mask
            ratio_t = broadcast_tensor(ratio, gs.tc_float, mass_data[mask].shape)
            assign_indexed_tensor(mass_data, mask, mass_data[mask] * ratio_t)
            assign_indexed_tensor(inertial_i_data, mask, inertial_i_data[mask] * ratio_t[..., None, None])
            assign_indexed_tensor(invweight_data, mask, invweight_data[mask] / ratio_t[..., None])
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        ratio, links_idx, envs_idx = self._sanitize_io_variables(
            ratio,
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
            batched=self._options.batch_links_info,
            skip_allocation=True,
        )
        if self.n_envs == 0 and self._options.batch_links_info:
            ratio = ratio[None]
        kernel_adjust_link_inertia(ratio, links_idx, envs_idx, self.links_info, self._static_rigid_sim_config)

    def set_geoms_friction_ratio(self, friction_ratio, geoms_idx=None, envs_idx=None):
        friction_ratio, geoms_idx, envs_idx = self._sanitize_io_variables(
            friction_ratio, geoms_idx, self.n_geoms, "geoms_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            friction_ratio = friction_ratio[None]
        kernel_set_geoms_friction_ratio(
            friction_ratio, geoms_idx, envs_idx, self.geoms_state, self._static_rigid_sim_config
        )

    @mutates(StateChange.GEOMETRY)
    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False):
        if self.collider is not None:
            self.collider.reset(envs_idx)
        if self.constraint_solver is not None:
            self.constraint_solver.reset(envs_idx)

        if gs.use_zerocopy:
            data = qd_to_torch(self._rigid_global_info.qpos, transpose=True, copy=False)
            errno = qd_to_torch(self._errno, copy=False)
            qs_mask = indices_to_mask(qs_idx)
            if (
                (not qs_mask or isinstance(qs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and envs_idx.dtype == torch.bool
            ):
                qs_data = data[(slice(None), *qs_mask)]
                if qpos.ndim == 2 and len(qpos) not in (1, len(qs_data)):
                    # A fresh source view is needed because masked_scatter_ may reshape it in-place. Metal mis-scatters
                    # a stride-0 broadcast mask, so it must be materialized to a dense full-shape mask there.
                    envs_mask = envs_idx[:, None]
                    if gs.backend == gs.metal:
                        envs_mask = envs_mask.expand_as(qs_data).contiguous()
                    qs_data.masked_scatter_(envs_mask, qpos.view_as(qpos))
                else:
                    qpos = broadcast_tensor(qpos, gs.tc_float, qs_data.shape)
                    torch.where(envs_idx[:, None], qpos, qs_data, out=qs_data)
                errno.masked_fill_(envs_idx, 0.0)
            else:
                mask = (0, *qs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *qs_mask)
                assign_indexed_tensor(data, mask, qpos)
                errno[envs_idx] = 0
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            qpos, qs_idx, envs_idx = self._sanitize_io_variables(
                qpos, qs_idx, self.n_qs, "qs_idx", envs_idx, skip_allocation=True
            )
            if self.n_envs == 0:
                qpos = qpos[None]

            # Teleporting a sleeping body must revive it, otherwise the new pose is silently dropped. Nothing can be
            # hibernated while the scene is still being built.
            if self._use_hibernation and self.is_built:
                kernel_wake_up_entities_by_qs(
                    qs_idx,
                    envs_idx,
                    links_info=self.links_info,
                    links_state=self.links_state,
                    entities_state=self.entities_state,
                    entities_info=self.entities_info,
                    dofs_state=self.dofs_state,
                    geoms_state=self.geoms_state,
                    rigid_global_info=self._rigid_global_info,
                    island_state=self.constraint_solver.island_state,
                    static_rigid_sim_config=self._static_rigid_sim_config,
                )

            kernel_set_qpos(qpos, qs_idx, envs_idx, self._rigid_global_info, self._static_rigid_sim_config)
            kernel_set_zero(envs_idx, self._errno)

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    @mutates(StateChange.GEOMETRY)
    def set_active_variant(self, entity, variant_idx, envs_idx=None):
        """Rebind which heterogeneous geometry variant each selected environment simulates, at runtime.

        Rewrites the per-environment geom/vgeom ranges and inertial for every heterogeneous link of
        'entity', then refreshes the inertia-derived quantities (invweight/meaninertia), contact caches
        and geom poses for the selected environments. The joint configuration is preserved. All variants
        must already be resident (declared at build via 'morph=[...]'); this only rebinds the active one.
        'variant_idx' is a single index applied to all selected environments, or one index per selected
        environment (variant 0 is the primary morph, 1..N the additional variants).
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        n_variants = len(entity._morph_heterogeneous) + 1
        if isinstance(variant_idx, torch.Tensor):
            variant_idx = tensor_to_array(variant_idx)
        variant_idx = np.broadcast_to(np.atleast_1d(variant_idx).astype(gs.np_int), (len(envs_idx),))
        if ((variant_idx < 0) | (variant_idx >= n_variants)).any():
            gs.raise_exception(f"variant_idx must be in [0, {n_variants}) for this entity.")
        envs_idx_np = tensor_to_array(envs_idx)

        for link in entity.links:
            if link._variant_vgeom_ranges is None:
                continue
            self._bind_link_variant(link, variant_idx, envs_idx_np)

        # The inertial changed, so the neutral-rest invweight/meaninertia must be recomputed. That routine
        # momentarily drives the selected environments to qpos0, so the current joint configuration is saved
        # and restored; the restoring set_qpos also resets contact caches and re-runs forward kinematics to
        # refresh geom poses and AABBs for the newly bound geometry.
        if self._n_dofs > 0:
            restore_envs = envs_idx if self.n_envs > 0 else None
            qpos_saved = qd_to_torch(self.qpos, envs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
                qpos_saved = qpos_saved[0]
            self._init_invweight_and_meaninertia(envs_idx=restore_envs)
            self.set_qpos(qpos_saved, envs_idx=restore_envs)
        else:
            if self.collider is not None:
                self.collider.reset(envs_idx)
            if self.constraint_solver is not None:
                self.constraint_solver.reset(envs_idx)

    def set_active_object(self, entity, object_ref, envs_idx=None):
        """Load a processed object into the entity's geometry pool and bind the selected environments to it.

        The object (a `gs.morphs.Mesh` or `gs.morphs.Primitive`) is processed to collision geometry, uploaded
        into a free pool slot (reused if the same object is already resident), and the selected environments
        are rebound to it via the same machinery as `set_active_variant`. The joint configuration is preserved;
        the object inherits the poolable entity's collision filters (contype/conaffinity) and base link.
        Requires the entity to have been built with `add_entity(geom_pool=...)`.
        """
        if self._geom_pool is None:
            gs.raise_exception("set_active_object requires an entity built with add_entity(geom_pool=...).")
        segment = self._geom_pool.segment_for_entity(entity._idx_in_solver)
        if segment is None:
            gs.raise_exception(f"Entity '{entity.name}' has no geometry pool.")

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        envs_idx_np = tensor_to_array(envs_idx)

        # Release the selected envs from their current slots first, so any slot they fully vacate becomes
        # evictable and can be reused for this object (its refcount drops to 0 before acquire runs).
        self._pool_release_envs(segment, envs_idx_np)

        # Resolve (or fill) the slot holding this object. A slot is keyed by the morph object's id; the segment
        # keeps a reference to each resident morph (resident_morph) so its id stays valid while resident -
        # otherwise a garbage-collected transient morph's id could be reused by a later one, aliasing slots.
        # Re-binding the same morph reuses its slot; uploading happens only the first time it is made resident.
        key = id(object_ref)
        slot = segment.key_to_slot.get(key)
        if slot is None:
            slot = self._pool_acquire_slot(segment)
            inertial, n_live = self._pool_upload_object(entity, segment, slot, object_ref)
            segment.resident_key[slot] = key
            segment.resident_morph[slot] = object_ref
            segment.inertial[slot] = inertial
            segment.n_live_geoms[slot] = n_live
            segment.key_to_slot[key] = slot
        # Mark most-recently-used (tail of the LRU list) and account the new env bindings.
        if slot in segment.lru:
            segment.lru.remove(slot)
        segment.lru.append(slot)
        for env in envs_idx_np:
            segment.env_slot[int(env)] = slot
            segment.refcount[slot] += 1

        # Bind the selected envs to the slot's geom range + inertial (reuse the heterogeneous rebind kernel),
        # then refresh inertia-derived quantities and reset caches exactly like set_active_variant.
        self._pool_bind_slot(entity, segment, slot, envs_idx_np)
        self._pool_bind_visual(entity, segment, slot, envs_idx_np)

        if self._n_dofs > 0:
            restore_envs = envs_idx if self.n_envs > 0 else None
            qpos_saved = qd_to_torch(self.qpos, envs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
                qpos_saved = qpos_saved[0]
            self._init_invweight_and_meaninertia(envs_idx=restore_envs)
            self.set_qpos(qpos_saved, envs_idx=restore_envs)
        else:
            if self.constraint_solver is not None:
                self.constraint_solver.reset(envs_idx)

        # The bound geom SET changed (not just poses), so the SAP broadphase sort buffer must be rebuilt for
        # the rebound envs. set_qpos only clears the contact cache (cache_only=True); force first_time=True so
        # the next broadphase re-reads the new per-env link geom ranges.
        if self.collider is not None:
            self.collider.reset(envs_idx, cache_only=False)

        # Pose the just-bound slot geoms and refresh their AABBs so getters/rendering are correct before the
        # next step (the per-substep pass would otherwise be the first to touch them).
        self._func_update_pool_geoms()

    def _pool_release_envs(self, segment, envs_idx_np):
        """Drop the given envs' bindings to their current slots, decrementing per-slot refcounts."""
        for env in envs_idx_np:
            prev = segment.env_slot.pop(int(env), None)
            if prev is not None:
                segment.refcount[prev] -= 1

    def _pool_acquire_slot(self, segment):
        """Return a slot for a new object: a free slot, else the LRU evictable slot (refcount 0, unpinned).

        Raises when every slot is currently bound to an environment or pinned.
        """
        for slot in range(segment.n_slots):
            if segment.resident_key[slot] is None:
                return slot
        # Full: evict the least-recently-used slot no env is bound to and that is not pinned.
        for slot in segment.lru:
            if segment.refcount[slot] == 0 and not segment.pinned[slot]:
                segment.key_to_slot.pop(segment.resident_key[slot], None)
                segment.resident_key[slot] = None
                segment.resident_morph[slot] = None
                segment.inertial[slot] = None
                segment.n_live_geoms[slot] = 0
                return slot
        gs.raise_exception(
            f"Geometry pool for this entity is full: all {segment.n_slots} slots are bound to an environment "
            "or pinned, so none can be evicted. Increase GeomPoolOptions.n_slots, or unbind/unpin some objects."
        )

    def _pool_process_object(self, entity, object_ref):
        """Load + postprocess a pool-object morph into (cg_infos, vg_infos); pure preprocessing, no slot writes.

        This is the nondeterministic step (convex decomposition, decimation), so declared objects run it once
        at build (`_pool_derive_from_objects`) and cache the result in `segment.declared`.
        """
        if isinstance(object_ref, gs.morphs.Mesh):
            g_infos = entity._load_mesh(object_ref, entity._surface, load_geom_only_for_heterogeneous=True)
        elif isinstance(object_ref, gs.morphs.Primitive):
            g_infos = entity._load_primitive(object_ref, entity._surface, load_geom_only_for_heterogeneous=True)
        else:
            gs.raise_exception("set_active_object only supports gs.morphs.Mesh and gs.morphs.Primitive objects.")
        return entity._postprocess_geoms_info(object_ref, g_infos, is_robot=False)

    def _pool_make_geom(self, entity, g_info, idx, vert_start, face_start, edge_start, cell_start, verts_state_start):
        """Build one transient RigidGeom from a processed geom info at the given (slot) array indices.

        It is not appended to any link - it just reuses the full geom preprocessing (vert/face/edge/normal
        extraction, data padding, center of mass). The size fields (n_verts/faces/edges/cells) are
        mesh-intrinsic and independent of the passed indices, so measurement can pass zeros.
        """
        from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom

        friction = entity.material.friction
        return RigidGeom(
            link=entity.base_link,
            idx=idx,
            cell_start=cell_start,
            vert_start=vert_start,
            face_start=face_start,
            edge_start=edge_start,
            verts_state_start=verts_state_start,
            mesh=g_info["mesh"],
            init_pos=g_info.get("pos", gu.zero_pos()),
            init_quat=g_info.get("quat", gu.identity_quat()),
            type=g_info["type"],
            friction=friction if friction is not None else g_info.get("friction", gu.default_friction()),
            sol_params=g_info["sol_params"],
            needs_coup=entity.material.needs_coup,
            contype=entity._morph.contype,
            conaffinity=entity._morph.conaffinity,
            data=g_info.get("data"),
        )

    def _pool_derive_from_objects(self, entity, objects):
        """Process each declared object once and derive per-slot budgets + the residency cache from the set.

        Returns (per_slot, declared): per_slot is the per-array-key maximum over the catalog (so every declared
        object fits a slot), and declared maps id(morph) -> (cg_infos, vg_infos) for reuse at upload time.
        """
        per_slot = {key: 0 for key in ("geom", "vert", "face", "edge", "cell", "free_vert", "vgeom")}
        declared = {}
        for object_ref in objects:
            cg_infos, vg_infos = self._pool_process_object(entity, object_ref)
            declared[id(object_ref)] = (cg_infos, vg_infos)
            per_slot["geom"] = max(per_slot["geom"], len(cg_infos))
            per_slot["vgeom"] = max(per_slot["vgeom"], len(vg_infos))
            verts = faces = edges = cells = 0
            for g_info in cg_infos:
                geom = self._pool_make_geom(entity, g_info, 0, 0, 0, 0, 0, 0)
                verts += geom.n_verts
                faces += geom.n_faces
                edges += geom.n_edges
                # Only nonconvex geoms use the SDF narrowphase and thus consume reserved cells.
                if not geom.is_convex:
                    cells += geom.n_cells
            per_slot["vert"] = max(per_slot["vert"], verts)
            per_slot["face"] = max(per_slot["face"], faces)
            per_slot["edge"] = max(per_slot["edge"], edges)
            per_slot["cell"] = max(per_slot["cell"], cells)
        per_slot["geom"] = max(per_slot["geom"], 1)
        per_slot["free_vert"] = per_slot["vert"]
        return per_slot, declared

    def _pool_upload_object(self, entity, segment, slot, object_ref):
        """Process a morph into collision geoms and upload them into `slot`'s reserved ranges.

        Returns ((mass, com, quat, inertia_tensor), n_live_geoms): the object's aggregate inertial for the link
        binding, and how many sub-geoms were uploaded (the bound geom range is the live sub-range of the slot).
        """
        from genesis.engine.entities.rigid_entity.rigid_link import (
            GeomInertialInfo,
            compose_inertial_properties,
            get_local_inertial_from_geom_info,
        )

        # A declared object was processed (and its nondeterministic decomposition frozen) at build; reuse that
        # so its uploaded geometry always matches the derived slot budgets. An undeclared object (a morph not
        # in GeomPoolOptions.objects) is processed on demand and must fit the budgets.
        cached = segment.declared.get(id(object_ref))
        cg_infos, vg_infos = cached if cached is not None else self._pool_process_object(entity, object_ref)
        if segment.per_slot["vgeom"] > 0:
            self._pool_set_slot_visual(segment, slot, vg_infos)

        ranges = segment.slots[slot]
        # Build transient RigidGeoms at the slot's reserved indices. Cursors advance within the slot's reserved
        # ranges; a geom's vertex-state rows track its vertex rows.
        geoms, geoms_inertial = [], []
        vert_cur, face_cur, edge_cur, cell_cur = ranges.vert[0], ranges.face[0], ranges.edge[0], ranges.cell[0]
        for i, g_info in enumerate(cg_infos):
            geom = self._pool_make_geom(
                entity,
                g_info,
                idx=ranges.geom[0] + i,
                vert_start=vert_cur,
                face_start=face_cur,
                edge_start=edge_cur,
                cell_start=cell_cur,
                verts_state_start=ranges.free_vert[0] + (vert_cur - ranges.vert[0]),
            )
            geoms.append(geom)
            geoms_inertial.append(
                GeomInertialInfo(
                    get_local_inertial_from_geom_info(g_info, self._pool_rho(entity)), geom.init_pos, geom.init_quat
                )
            )
            vert_cur += geom.n_verts
            face_cur += geom.n_faces
            edge_cur += geom.n_edges
            # Only nonconvex geoms use the SDF narrowphase, so only they consume reserved cells (convex and
            # primitives use analytic/MPR support). Advance the cell cursor solely for them.
            if not geom.is_convex:
                cell_cur += geom.n_cells

        # Validate the object fits the slot's uniform budgets (a static reservation cannot grow).
        for name, cur, rng in (
            ("geom", ranges.geom[0] + len(geoms), ranges.geom),
            ("vert", vert_cur, ranges.vert),
            ("face", face_cur, ranges.face),
            ("edge", edge_cur, ranges.edge),
            ("cell", cell_cur, ranges.cell),
        ):
            used, budget = cur - rng[0], rng[1] - rng[0]
            if used > budget:
                gs.raise_exception(
                    f"Object exceeds the pool slot's {name} budget ({used} > {budget}). Increase the "
                    f"corresponding GeomPoolOptions per-slot budget."
                )

        self._pool_write_slot(geoms, ranges)

        mass, com, inertia = compose_inertial_properties(geoms_inertial)
        inertial = (
            float(mass),
            np.asarray(com, dtype=gs.np_float),
            gu.identity_quat(),
            np.asarray(inertia, dtype=gs.np_float),
        )
        return inertial, len(geoms)

    def _pool_set_slot_visual(self, segment, slot, vg_infos):
        """Set a slot's placeholder visual mesh from an uploaded object's visual geoms (merged into one mesh).

        The placeholder holds the host trimesh only; rigid rendering + get_vAABB use it plus the per-env device
        vgeom pose, so no device vvert upload is needed. Flags the placeholder for the rasterizer to re-mesh.
        """
        placeholder = segment.vgeom_placeholders[slot]
        if placeholder is None or not vg_infos:
            return
        vmeshes = [vg["vmesh"] for vg in vg_infos]
        if len(vmeshes) == 1:
            vmesh = vmeshes[0]
        else:
            vmesh = gs.Mesh.from_trimesh(
                trimesh.util.concatenate([vm.trimesh for vm in vmeshes]), surface=vmeshes[0].surface
            )
        placeholder.set_pool_mesh(vmesh)

    def _pool_bind_visual(self, entity, segment, slot, envs_idx_np):
        """Per-env visual switch: in the given envs, show slot `slot`'s placeholder and hide the entity's base
        morph visuals plus the other slots' placeholders."""
        if segment.per_slot["vgeom"] == 0:
            return
        envs = torch.as_tensor(envs_idx_np, device=gs.device)
        placeholder_ids = {id(ph) for ph in segment.vgeom_placeholders if ph is not None}
        for vgeom in entity.vgeoms:
            if id(vgeom) in placeholder_ids or vgeom.active_envs_mask is None:
                continue
            vgeom.active_envs_mask[envs] = False  # hide the base morph visual where a pooled object is shown
            (vgeom.active_envs_idx,) = np.where(tensor_to_array(vgeom.active_envs_mask))
        for other_slot, placeholder in enumerate(segment.vgeom_placeholders):
            if placeholder is None:
                continue
            placeholder.active_envs_mask[envs] = other_slot == slot
            (placeholder.active_envs_idx,) = np.where(tensor_to_array(placeholder.active_envs_mask))

    def _pool_set_scale(self, entity, segment, scale, envs_idx):
        """Scale each selected env's ACTIVE pool geometry (bound slot, else the entity's base geoms) in place.

        Writes the per-env geom/vgeom scale fields (collision + visual rescale automatically) and rescales the
        per-env link inertial from that env's active baseline via the same covariance transform set_scale uses.
        """
        base_link = entity.base_link
        envs_idx_np = tensor_to_array(envs_idx)
        n_sel = len(envs_idx_np)

        scale = np.asarray(scale, dtype=gs.np_float)
        if scale.ndim == 0:
            scale = np.broadcast_to(scale.reshape(1, 1), (n_sel, 3))
        elif scale.shape == (3,):
            scale = np.broadcast_to(scale.reshape(1, 3), (n_sel, 3))
        scale = np.ascontiguousarray(np.broadcast_to(scale, (n_sel, 3)))
        if (scale <= 0.0).any():
            gs.raise_exception("scale must be strictly positive.")

        # Per env, resolve its active geom range + placeholder vgeom + inertial baseline, then covariance-scale.
        geom_lo = np.empty(n_sel, dtype=gs.np_int)
        geom_hi = np.empty(n_sel, dtype=gs.np_int)
        vgeom_idx = np.full(n_sel, -1, dtype=gs.np_int)
        out_mass = np.empty((n_sel, 1), dtype=gs.np_float)
        out_pos = np.empty((n_sel, 1, 3), dtype=gs.np_float)
        out_quat = np.zeros((n_sel, 1, 4), dtype=gs.np_float)
        out_quat[..., 0] = 1.0
        out_i = np.empty((n_sel, 1, 3, 3), dtype=gs.np_float)
        eye = np.eye(3, dtype=gs.np_float)
        for i, env in enumerate(envs_idx_np):
            slot = segment.env_slot.get(int(env))
            if slot is not None:
                ranges = segment.slots[slot]
                geom_lo[i], geom_hi[i] = ranges.geom[0], ranges.geom[0] + segment.n_live_geoms[slot]
                placeholder = segment.vgeom_placeholders[slot]
                vgeom_idx[i] = placeholder.idx if placeholder is not None else -1
                mass0, com0, quat0, i0 = segment.inertial[slot]
                rot0 = gu.quat_to_R(np.asarray(quat0, dtype=gs.np_float))
                i_link = rot0 @ np.asarray(i0, dtype=gs.np_float) @ rot0.T
                com0 = np.asarray(com0, dtype=gs.np_float)
            else:
                geom_lo[i], geom_hi[i] = entity.geom_start, entity.geom_end
                vgeom_idx[i] = entity.vgeoms[0].idx if len(entity.vgeoms) else -1
                rot0 = gu.quat_to_R(np.asarray(base_link._inertial_quat, dtype=gs.np_float))
                i_link = rot0 @ np.asarray(base_link._inertial_i, dtype=gs.np_float) @ rot0.T
                com0 = np.asarray(base_link._inertial_pos, dtype=gs.np_float)
                mass0 = float(base_link._inertial_mass)
            s = scale[i]
            det = float(s[0] * s[1] * s[2])
            s_mat = np.diag(s)
            c0 = 0.5 * np.trace(i_link) * eye - i_link
            cov = det * (s_mat @ c0 @ s_mat)
            out_i[i, 0] = np.trace(cov) * eye - cov
            out_pos[i, 0] = s * com0
            out_mass[i, 0] = mass0 * det

        kernel_set_pool_scale(
            envs_idx_np,
            geom_lo,
            geom_hi,
            vgeom_idx,
            scale,
            self.geoms_state,
            self.vgeoms_state,
            self._static_rigid_sim_config,
        )
        kernel_set_links_inertial(
            out_mass,
            out_pos,
            out_quat,
            out_i,
            np.array([base_link.idx], dtype=gs.np_int),
            envs_idx_np,
            self.links_info,
            self._static_rigid_sim_config,
        )

        # Refresh neutral-rest invweight/meaninertia (preserving joint config); the per-substep pool-FK pass
        # rescales the slot AABBs from the new geoms_state.scale.
        if self._n_dofs > 0:
            restore_envs = envs_idx if self.n_envs > 0 else None
            qpos_saved = qd_to_torch(self.qpos, envs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
                qpos_saved = qpos_saved[0]
            self._init_invweight_and_meaninertia(envs_idx=restore_envs)
            self.set_qpos(qpos_saved, envs_idx=restore_envs)
        self._func_update_pool_geoms()

    def _pool_rho(self, entity):
        """Material density for a pooled object, resolved exactly as RigidLink._build does for a variant."""
        from genesis.engine.entities.rigid_entity.rigid_link import RHO_MUJOCO, RHO_OBJECT

        rho = entity.material.rho
        if rho is None:
            # Pooled objects are always non-robot (Mesh/Primitive), so the non-mujoco default is RHO_OBJECT.
            rho = RHO_MUJOCO if self._enable_mujoco_compatibility else RHO_OBJECT
        return rho

    def _pool_write_slot(self, geoms, ranges):
        """Assemble host payloads from transient geoms and upload them into the slot's reserved device rows."""
        n_geoms = len(geoms)
        # Per-vertex payloads (concatenated across the object's sub-geoms), with absolute global indices.
        verts = np.concatenate([g.init_verts for g in geoms], dtype=gs.np_float)
        normals = np.concatenate([g.init_normals for g in geoms], dtype=gs.np_float)
        init_center = np.concatenate([g.init_center_pos for g in geoms], dtype=gs.np_float)
        verts_geom_idx = np.concatenate([np.full(g.n_verts, g.idx, dtype=gs.np_int) for g in geoms])
        n_verts_total = len(verts)
        verts_state_idx = np.arange(ranges.free_vert[0], ranges.free_vert[0] + n_verts_total, dtype=gs.np_int)
        verts_is_fixed = np.zeros(n_verts_total, dtype=gs.np_bool)
        faces = np.concatenate([g.init_faces + g.vert_start for g in geoms], dtype=gs.np_int)
        faces_geom_idx = np.concatenate([np.full(g.n_faces, g.idx, dtype=gs.np_int) for g in geoms])
        edges = np.concatenate([g.init_edges + g.vert_start for g in geoms], dtype=gs.np_int)

        kernel_upload_vert_slot(
            ranges.vert[0],
            ranges.face[0],
            ranges.edge[0],
            verts,
            normals,
            init_center,
            verts_geom_idx,
            verts_state_idx,
            verts_is_fixed,
            faces,
            faces_geom_idx,
            edges,
            self.verts_info,
            self.faces_info,
            self.edges_info,
            self._static_rigid_sim_config,
        )

        # Per-geom payloads.
        def center(g):
            tmesh = g.mesh.trimesh
            return tmesh.center_mass if tmesh.is_watertight else np.mean(tmesh.vertices, axis=0)

        # Use the raw stored sol_params: the geom.sol_params property reads the device (still zero for a
        # freshly-reserved slot) because these transient geoms hang off a built link.
        sol_params = np.array([g._sol_params for g in geoms], dtype=gs.np_float)
        _sanitize_sol_params(sol_params, self._sol_min_timeconst, self._sol_default_timeconst)
        kernel_upload_geom_slot(
            ranges.geom[0],
            np.array([g.init_pos for g in geoms], dtype=gs.np_float),
            np.array([center(g) for g in geoms], dtype=gs.np_float),
            np.array([g.init_quat for g in geoms], dtype=gs.np_float),
            np.full(n_geoms, geoms[0].link.idx, dtype=gs.np_int),
            np.array([g.type for g in geoms], dtype=gs.np_int),
            np.array([g.friction for g in geoms], dtype=gs.np_float),
            sol_params,
            np.array([g.vert_start for g in geoms], dtype=gs.np_int),
            np.array([g.face_start for g in geoms], dtype=gs.np_int),
            np.array([g.edge_start for g in geoms], dtype=gs.np_int),
            np.array([g.verts_state_start for g in geoms], dtype=gs.np_int),
            np.array([g.vert_end for g in geoms], dtype=gs.np_int),
            np.array([g.face_end for g in geoms], dtype=gs.np_int),
            np.array([g.edge_end for g in geoms], dtype=gs.np_int),
            np.array([g.verts_state_start + g.n_verts for g in geoms], dtype=gs.np_int),
            np.array([g.data for g in geoms], dtype=gs.np_float),
            np.array([g.is_convex for g in geoms], dtype=gs.np_bool),
            np.array([g.needs_coup for g in geoms], dtype=gs.np_int),
            np.array([g.contype for g in geoms], dtype=np.int32),
            np.array([g.conaffinity for g in geoms], dtype=np.int32),
            np.array([g.coup_softness for g in geoms], dtype=gs.np_float),
            np.array([g.coup_friction for g in geoms], dtype=gs.np_float),
            np.array([g.coup_restitution for g in geoms], dtype=gs.np_float),
            np.zeros(n_geoms, dtype=gs.np_bool),
            np.array([g.metadata.get("decomposed", False) for g in geoms], dtype=gs.np_bool),
            self.geoms_info,
            self.geoms_state,
            self.verts_info,
            self.geoms_init_AABB,
            self._static_rigid_sim_config,
        )

        # Nonconvex sub-geoms collide via the vertex-vs-SDF narrowphase, so upload their baked SDF grids into
        # the slot's reserved cell range (convex/primitive geoms use analytic/MPR support and need no grid).
        sdf_geoms = [g for g in geoms if not g.is_convex]
        if sdf_geoms:
            kernel_upload_sdf_slot(
                sdf_geoms[0].cell_start,
                np.array([g.idx for g in sdf_geoms], dtype=gs.np_int),
                np.array([g.T_mesh_to_sdf for g in sdf_geoms], dtype=gs.np_float),
                np.array([g.sdf_res for g in sdf_geoms], dtype=gs.np_int),
                np.array([g.cell_start for g in sdf_geoms], dtype=gs.np_int),
                np.array([g.sdf_max for g in sdf_geoms], dtype=gs.np_float),
                np.array(
                    [np.broadcast_to(np.asarray(g.sdf_cell_size, dtype=gs.np_float), (3,)) for g in sdf_geoms],
                    dtype=gs.np_float,
                ),
                np.concatenate([g.sdf_val_flattened for g in sdf_geoms], dtype=gs.np_float),
                np.concatenate([g.sdf_grad_flattened for g in sdf_geoms], dtype=gs.np_float),
                np.concatenate([g.sdf_closest_vert_flattened for g in sdf_geoms], dtype=gs.np_int),
                self._static_rigid_sim_config,
                self.collider._sdf._sdf_info,
            )

    def _pool_bind_slot(self, entity, segment, slot, envs_idx_np):
        """Point each selected env's base link at the slot's live geom range + inertial (device rebind)."""
        ranges = segment.slots[slot]
        link = entity.base_link
        n = len(envs_idx_np)
        mass, com, quat, inertia = segment.inertial[slot]
        # Bind only the live sub-range (the uploaded sub-geoms); the rest of the slot stays reserved/inert.
        geom_start = ranges.geom[0]
        geom_end = geom_start + segment.n_live_geoms[slot]
        # Keep the base morph's visual geometry (visual pooling is a follow-up); rebind collision + inertial.
        kernel_update_heterogeneous_link_info(
            link.idx,
            envs_idx_np,
            np.full(n, geom_start, dtype=gs.np_int),
            np.full(n, geom_end, dtype=gs.np_int),
            np.full(n, link.vgeom_start, dtype=gs.np_int),
            np.full(n, link.vgeom_end, dtype=gs.np_int),
            np.full(n, mass, dtype=gs.np_float),
            np.tile(com, (n, 1)),
            np.tile(quat, (n, 1)),
            np.tile(inertia, (n, 1, 1)),
            self.links_info,
        )

    @mutates(StateChange.GEOMETRY)
    def set_entity_scale(self, entity, scale, envs_idx=None):
        """Set a per-environment geometry scale (diagonal, about each geom's frame origin) for an entity.

        Scales the entity's collision geometry, AABBs and per-link inertial for the selected environments.
        'scale' is a scalar (isotropic), a length-3 vector, or a per-environment (n_envs, 3) array. Requires
        the scene built with 'enable_geom_scaling=True'. Radial primitives (capsule/cylinder) require an
        isotropic radial scale (sx == sy). The joint configuration is preserved.
        """
        if not self._enable_geom_scaling:
            gs.raise_exception("set_scale requires the scene built with RigidOptions(enable_geom_scaling=True).")
        if self._requires_grad:
            gs.raise_exception("set_scale is not supported under requires_grad.")

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        n_sel = len(envs_idx)

        # A geometry-pool entity's per-env active geoms live in the reserved block (a bound slot), not in the
        # entity's static geom range, so scale the active slot per env instead of the base geoms.
        pool_segment = self._geom_pool.segment_for_entity(entity._idx_in_solver) if self._geom_pool else None
        if pool_segment is not None:
            self._pool_set_scale(entity, pool_segment, scale, envs_idx)
            return

        # Resolve scale to a per-selected-environment (n_sel, 3) array.
        scale = np.asarray(scale, dtype=gs.np_float)
        if scale.ndim == 0:
            scale = np.broadcast_to(scale.reshape(1, 1), (n_sel, 3))
        elif scale.shape == (3,):
            scale = np.broadcast_to(scale.reshape(1, 3), (n_sel, 3))
        scale = np.ascontiguousarray(np.broadcast_to(scale, (n_sel, 3)))
        if (scale <= 0.0).any():
            gs.raise_exception("scale must be strictly positive.")

        geoms = entity.geoms
        for geom in geoms:
            if geom.is_fixed and not entity._batch_fixed_verts:
                gs.raise_exception(
                    "set_scale is not supported for fixed-vertex geoms; build the entity with batch_fixed_verts=True."
                )
            if geom.type in (gs.GEOM_TYPE.CAPSULE, gs.GEOM_TYPE.CYLINDER) and not np.allclose(scale[:, 0], scale[:, 1]):
                gs.raise_exception(f"Radial primitive {geom.type} requires an isotropic radial scale (sx == sy).")

        envs_idx_np = tensor_to_array(envs_idx)

        # 1. Write the per-env geom scale field (the same scale for every geom of the entity).
        geoms_idx = np.array([geom.idx for geom in geoms], dtype=gs.np_int)
        geoms_scale = np.ascontiguousarray(np.broadcast_to(scale[:, None, :], (n_sel, len(geoms_idx), 3)))
        kernel_set_geoms_scale(geoms_scale, geoms_idx, envs_idx_np, self.geoms_state, self._static_rigid_sim_config)

        # 1b. Mirror the scale onto the entity's visual geometry so vverts/vAABB and rendering rescale too.
        vgeoms = entity.vgeoms
        if vgeoms:
            vgeoms_idx = np.array([vgeom.idx for vgeom in vgeoms], dtype=gs.np_int)
            vgeoms_scale = np.ascontiguousarray(np.broadcast_to(scale[:, None, :], (n_sel, len(vgeoms_idx), 3)))
            kernel_set_vgeoms_scale(
                vgeoms_scale, vgeoms_idx, envs_idx_np, self.vgeoms_state, self._static_rigid_sim_config
            )

        # 2. Recompute per-link inertial from the baseline (unit-scale) inertial via the covariance transform:
        #    mass *= det(S), com -> S com, and (with the link-frame tensor I0) C0 = 0.5 tr(I0) Id - I0,
        #    C = det(S) S C0 S, I = tr(C) Id - C. The scaled tensor is stored in the link frame (quat = identity).
        links = entity.links
        links_idx = np.array([link.idx for link in links], dtype=gs.np_int)
        n_l = len(links_idx)
        out_mass = np.empty((n_sel, n_l), dtype=gs.np_float)
        out_pos = np.empty((n_sel, n_l, 3), dtype=gs.np_float)
        out_quat = np.zeros((n_sel, n_l, 4), dtype=gs.np_float)
        out_quat[..., 0] = 1.0
        out_i = np.empty((n_sel, n_l, 3, 3), dtype=gs.np_float)
        eye = np.eye(3, dtype=gs.np_float)
        for i_l_, link in enumerate(links):
            rot = gu.quat_to_R(np.array(link._inertial_quat, dtype=gs.np_float))
            i_link = rot @ np.array(link._inertial_i, dtype=gs.np_float) @ rot.T
            com0 = np.array(link._inertial_pos, dtype=gs.np_float)
            mass0 = float(link._inertial_mass)
            c0 = 0.5 * np.trace(i_link) * eye - i_link
            for i_b_ in range(n_sel):
                s = scale[i_b_]
                det = float(s[0] * s[1] * s[2])
                s_mat = np.diag(s)
                cov = det * (s_mat @ c0 @ s_mat)
                out_i[i_b_, i_l_] = np.trace(cov) * eye - cov
                out_pos[i_b_, i_l_] = s * com0
                out_mass[i_b_, i_l_] = mass0 * det
        kernel_set_links_inertial(
            out_mass, out_pos, out_quat, out_i, links_idx, envs_idx_np, self.links_info, self._static_rigid_sim_config
        )

        # 3. Refresh neutral-rest invweight/meaninertia (preserving the current joint configuration; the restoring
        #    set_qpos also drops stale contact caches and re-runs forward kinematics so geom poses/AABBs rescale).
        if self._n_dofs > 0:
            restore_envs = envs_idx if self.n_envs > 0 else None
            qpos_saved = qd_to_torch(self.qpos, envs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
                qpos_saved = qpos_saved[0]
            self._init_invweight_and_meaninertia(envs_idx=restore_envs)
            self.set_qpos(qpos_saved, envs_idx=restore_envs)
        else:
            if self.collider is not None:
                self.collider.reset(envs_idx)
            if self.constraint_solver is not None:
                self.constraint_solver.reset(envs_idx)

    def set_global_sol_params(self, sol_params):
        """
        Set constraint solver parameters.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: Tuple[float] | List[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        """
        sol_params_ = broadcast_tensor(sol_params, gs.tc_float, (7,), ("",))
        sol_params_ = _sanitize_sol_params(sol_params_.clone(), self._sol_min_timeconst)
        kernel_set_global_sol_params(
            sol_params_, self.geoms_info, self.joints_info, self.equalities_info, self._static_rigid_sim_config
        )

    def set_sol_params(self, sol_params, geoms_idx=None, envs_idx=None, *, joints_idx=None, eqs_idx=None):
        """
        Set constraint solver parameters.

        See :func:`genesis.utils.geom.default_solver_params` for the parameter semantics, in particular the
        relationship between ``dampratio``, spring stiffness, and velocity damping.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: Tuple[float] | List[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        """
        # Make sure that a single constraint type has been selected at once
        if sum(inputs_idx is not None for inputs_idx in (geoms_idx, joints_idx, eqs_idx)) > 1:
            gs.raise_exception("Cannot set more than one constraint type at once.")

        # Select the right input type
        if eqs_idx is not None:
            constraint_type = 2
            idx_name = "eqs_idx"
            inputs_idx = eqs_idx
            inputs_length = self.n_equalities
            batched = True
        elif joints_idx is not None:
            constraint_type = 1
            idx_name = "joints_idx"
            inputs_idx = joints_idx
            inputs_length = self.n_joints
            batched = self._options.batch_joints_info
        else:
            constraint_type = 0
            idx_name = "geoms_idx"
            inputs_idx = geoms_idx
            inputs_length = self.n_geoms
            batched = False

        # Sanitize input arguments
        sol_params_, inputs_idx, envs_idx = self._sanitize_io_variables(
            sol_params, inputs_idx, inputs_length, idx_name, envs_idx, (7,), batched=batched, skip_allocation=True
        )
        sol_params_ = _sanitize_sol_params(sol_params_.clone(), self._sol_min_timeconst)
        if self.n_envs == 0 and batched:
            sol_params_ = sol_params_[None]

        kernel_set_sol_params(
            constraint_type,
            sol_params_,
            inputs_idx,
            envs_idx,
            geoms_info=self.geoms_info,
            joints_info=self.joints_info,
            equalities_info=self.equalities_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def _set_dofs_info(self, tensor_list, dofs_idx, name, envs_idx=None):
        if gs.use_zerocopy and name in {
            "kp",
            "kv",
            "act_gain",
            "act_bias",
            "force_range",
            "stiffness",
            "damping",
            "frictionloss",
            "limit",
        }:
            mask = indices_to_mask(*((envs_idx, dofs_idx) if self._options.batch_dofs_info else (dofs_idx,)))
            if name == "kp":
                # kp sets act_gain, act_bias[0] = 0, act_bias[1] = -kp (full PD reset)
                kp = torch.as_tensor(tensor_list[0], dtype=gs.tc_float, device=gs.device)
                gain = qd_to_torch(self.dofs_info.act_gain, transpose=True, copy=False)
                assign_indexed_tensor(gain, mask, kp)
                bias = qd_to_torch(self.dofs_info.act_bias, transpose=True, copy=False)
                bias[(*mask, ..., 0)] = 0.0
                assign_indexed_tensor(bias, (*mask, ..., 1), -kp)
            elif name == "kv":
                # kv sets act_bias[..., 2] = -kv
                kv = torch.as_tensor(tensor_list[0], dtype=gs.tc_float, device=gs.device)
                bias = qd_to_torch(self.dofs_info.act_bias, transpose=True, copy=False)
                assign_indexed_tensor(bias, (*mask, ..., 2), -kv)
            else:
                data = qd_to_torch(getattr(self.dofs_info, name), transpose=True, copy=False)
                num_values = len(tensor_list)
                for j, mask_j in enumerate(((*mask, ..., j) for j in range(num_values)) if num_values > 1 else (mask,)):
                    assign_indexed_tensor(data, mask_j, tensor_list[j])
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        tensor_list = list(tensor_list)
        for j, tensor in enumerate(tensor_list):
            tensor, dofs_idx, envs_idx_ = self._sanitize_io_variables(
                tensor,
                dofs_idx,
                self.n_dofs,
                "dofs_idx",
                envs_idx,
                batched=self._options.batch_dofs_info,
                skip_allocation=True,
            )
            if self.n_envs == 0 and self._options.batch_dofs_info:
                tensor = tensor[None]
            tensor_list[j] = tensor
        if name == "kp":
            kernel_set_dofs_kp(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "kv":
            kernel_set_dofs_kv(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "force_range":
            kernel_set_dofs_force_range(
                *tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "stiffness":
            kernel_set_dofs_stiffness(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "armature":
            kernel_set_dofs_armature(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
            qs_idx = torch.arange(self.n_qs, dtype=gs.tc_int, device=gs.device)
            qpos_cur = self.get_qpos(qs_idx=qs_idx, envs_idx=envs_idx)
            self._init_invweight_and_meaninertia(envs_idx=envs_idx, force_update=True)
            self.set_qpos(qpos_cur, qs_idx=qs_idx, envs_idx=envs_idx)
        elif name == "damping":
            kernel_set_dofs_damping(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "frictionloss":
            kernel_set_dofs_frictionloss(
                *tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "limit":
            kernel_set_dofs_limit(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "act_gain":
            kernel_set_dofs_act_gain(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "act_bias":
            kernel_set_dofs_act_bias(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def set_dofs_kp(self, kp, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kp], dofs_idx, "kp", envs_idx)

    def set_dofs_kv(self, kv, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kv], dofs_idx, "kv", envs_idx)

    def set_dofs_act_gain(self, act_gain, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([act_gain], dofs_idx, "act_gain", envs_idx)

    def set_dofs_act_bias(self, bias0, bias1, bias2, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([bias0, bias1, bias2], dofs_idx, "act_bias", envs_idx)

    def set_dofs_force_range(self, lower, upper, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "force_range", envs_idx)

    def set_dofs_stiffness(self, stiffness, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([stiffness], dofs_idx, "stiffness", envs_idx)

    def set_dofs_armature(self, armature, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([armature], dofs_idx, "armature", envs_idx)

    def set_dofs_damping(self, damping, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([damping], dofs_idx, "damping", envs_idx)

    def set_dofs_frictionloss(self, frictionloss, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([frictionloss], dofs_idx, "frictionloss", envs_idx)

    def set_dofs_limit(self, lower, upper, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx)

    @mutates(StateChange.GEOMETRY)
    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        self.collider.reset(envs_idx)
        self.constraint_solver.reset(envs_idx)

        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]

        self._wake_dofs(dofs_idx, envs_idx)

        kernel_set_dofs_position(
            position,
            dofs_idx,
            envs_idx,
            self.dofs_state,
            self.links_info,
            self.joints_info,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

        if gs.use_zerocopy:
            errno = qd_to_torch(self._errno, copy=False)
            errno[envs_idx] = 0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_set_zero(envs_idx, self._errno)

        kernel_forward_kinematics_links_geoms(
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def _wake_dofs(self, dofs_idx, envs_idx):
        # Revive any hibernated entity owning these (already sanitized) dofs before an input is written to or
        # targeted at them; forward dynamics and integration act only on awake dofs, so an input applied to a
        # sleeping body would otherwise be silently dropped until it is woken by some other means.
        if self._use_hibernation:
            kernel_wake_up_entities_by_dofs(
                dofs_idx,
                envs_idx,
                links_info=self.links_info,
                links_state=self.links_state,
                entities_state=self.entities_state,
                entities_info=self.entities_info,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                island_state=self.constraint_solver.island_state,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False):
        # Wake the owning entities before delegating to the base setter, which re-sanitizes and applies the write.
        if self._use_hibernation:
            _, wake_dofs_idx, wake_envs_idx = self._sanitize_io_variables(
                velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            self._wake_dofs(wake_dofs_idx, wake_envs_idx)
        super().set_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=skip_forward)

    def control_dofs_force(self, force, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy and not self._use_hibernation:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.FORCE
            ctrl_force = qd_to_torch(self.dofs_state.ctrl_force, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_force, mask, force)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        force, dofs_idx, envs_idx = self._sanitize_io_variables(
            force, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            force = force[None]

        self._wake_dofs(dofs_idx, envs_idx)
        kernel_control_dofs_force(force, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def control_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy and not self._use_hibernation:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.VELOCITY
            ctrl_pos = qd_to_torch(self.dofs_state.ctrl_pos, transpose=True, copy=False)
            ctrl_pos[mask] = 0.0
            ctrl_vel = qd_to_torch(self.dofs_state.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity = velocity[None]

        self._wake_dofs(dofs_idx, envs_idx)
        kernel_control_dofs_velocity(velocity, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy and not self._use_hibernation:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(self.dofs_state.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(self.dofs_state.ctrl_vel, transpose=True, copy=False)
            ctrl_vel[mask] = 0.0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]

        self._wake_dofs(dofs_idx, envs_idx)
        kernel_control_dofs_position(position, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def control_dofs_position_velocity(self, position, velocity, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy and not self._use_hibernation:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(self.dofs_state.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(self.dofs_state.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        position, dofs_idx, _ = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]
            velocity = velocity[None]

        self._wake_dofs(dofs_idx, envs_idx)
        kernel_control_dofs_position_velocity(
            position, velocity, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config
        )

    def get_sol_params(self, geoms_idx=None, envs_idx=None, *, joints_idx=None, eqs_idx=None):
        """
        Get constraint solver parameters.
        """
        if eqs_idx is not None:
            # Always batched
            tensor = qd_to_torch(self.equalities_info.sol_params, envs_idx, eqs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
                tensor = tensor[0]
        elif joints_idx is not None:
            # Conditionally batched
            assert envs_idx is None
            # batch_shape = (envs_idx, joints_idx) if self._options.batch_joints_info else (joints_idx,)
            # tensor = qd_to_torch(self.joints_info.sol_params, *batch_shape, transpose=True)
            tensor = qd_to_torch(self.joints_info.sol_params, envs_idx, joints_idx, transpose=True, copy=True)
            if self.n_envs == 0 and self._options.batch_joints_info:
                tensor = tensor[0]
        else:  # geoms_idx is not None
            # Never batched
            assert envs_idx is None
            tensor = qd_to_torch(self.geoms_info.sol_params, geoms_idx, transpose=True, copy=True)
        return tensor

    @staticmethod
    def _convert_ref_to_idx(ref: Literal["link_origin", "link_com", "root_com"]):
        if ref == "root_com":
            return 0
        elif ref == "link_com":
            return 1
        elif ref == "link_origin":
            return 2
        else:
            gs.raise_exception("'ref' must be either 'link_origin', 'link_com', or 'root_com'.")

    def get_links_pos(
        self,
        links_idx=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        relative=False,
    ):
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = self._sanitize_io_variables(
                None, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )

        ref_idx = self._convert_ref_to_idx(ref)
        if ref_idx == 0:
            tensor = qd_to_torch(self.links_state.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        elif ref_idx == 1:
            i_pos = qd_to_torch(self.links_state.i_pos, envs_idx, links_idx, transpose=True)
            root_COM = qd_to_torch(self.links_state.root_COM, envs_idx, links_idx, transpose=True)
            tensor = i_pos + root_COM
        elif ref_idx == 2:
            tensor = qd_to_torch(self.links_state.pos, envs_idx, links_idx, transpose=True, copy=True)
        else:
            gs.raise_exception("'ref' must be either 'link_origin', 'link_com', or 'root_com'.")

        # The pose offset is defined on the link origin, so it is only stripped for the 'link_origin' reference.
        if relative and ref_idx == 2 and self._links_offset_pos is not None:
            quat = qd_to_torch(self.links_state.quat, envs_idx, links_idx, transpose=True, copy=True)
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)

        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_vel(
        self, links_idx=None, envs_idx=None, *, ref: Literal["link_origin", "link_com", "root_com"] = "link_origin"
    ):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(links_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, links_idx)
            cd_vel = qd_to_torch(self.links_state.cd_vel, transpose=True)
            if ref == "root_com":
                return cd_vel[mask]
            cd_ang = qd_to_torch(self.links_state.cd_ang, transpose=True)
            if ref == "link_com":
                i_pos = qd_to_torch(self.links_state.i_pos, transpose=True)
                delta = i_pos[mask]
            else:
                pos = qd_to_torch(self.links_state.pos, transpose=True)
                root_COM = qd_to_torch(self.links_state.root_COM, transpose=True)
                delta = pos[mask] - root_COM[mask]
            return cd_vel[mask] + cd_ang[mask].cross(delta, dim=-1)

        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        assert _tensor is not None
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        ref_idx = self._convert_ref_to_idx(ref)
        kernel_get_links_vel(tensor, links_idx, envs_idx, ref_idx, self.links_state, self._static_rigid_sim_config)
        return _tensor

    def get_links_acc(self, links_idx=None, envs_idx=None):
        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_links_acc(tensor, links_idx, envs_idx, self.links_state, self._static_rigid_sim_config)
        return _tensor

    def get_links_acc_ang(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.cacc_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_root_COM(self, links_idx=None, envs_idx=None):
        """
        Returns the center of mass (COM) of the entire kinematic tree to which the specified links belong.

        This corresponds to the global COM of each entity, assuming a single-rooted structure - that is, as long as no
        two successive links are connected by a free-floating joint (ie a joint that allows all 6 degrees of freedom).
        """
        tensor = qd_to_torch(self.links_state.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_mass_shift(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.mass_shift, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_COM_shift(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.i_pos_shift, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_inertial_mass(self, links_idx=None, envs_idx=None):
        if self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.links_info.inertial_mass, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_links_invweight(self, links_idx=None, envs_idx=None):
        if self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.links_info.invweight, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_geoms_friction_ratio(self, geoms_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.geoms_state.friction_ratio, envs_idx, geoms_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_geoms_pos(self, geoms_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.geoms_state.pos, envs_idx, geoms_idx, transpose=True, copy=True)
        if relative and self._geoms_offset_pos is not None:
            quat = qd_to_torch(self.geoms_state.quat, envs_idx, geoms_idx, transpose=True, copy=True)
            offset_pos = self._geoms_offset_pos if geoms_idx is None else self._geoms_offset_pos[geoms_idx]
            offset_quat = self._geoms_offset_quat if geoms_idx is None else self._geoms_offset_quat[geoms_idx]
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_geoms_quat(self, geoms_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.geoms_state.quat, envs_idx, geoms_idx, transpose=True, copy=True)
        if relative and self._geoms_offset_quat is not None:
            offset_quat = self._geoms_offset_quat if geoms_idx is None else self._geoms_offset_quat[geoms_idx]
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_control_force(self, dofs_idx=None, envs_idx=None):
        _tensor, dofs_idx, envs_idx = self._sanitize_io_variables(None, dofs_idx, self.n_dofs, "dofs_idx", envs_idx)
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_dofs_control_force(
            tensor, dofs_idx, envs_idx, self.dofs_state, self.dofs_info, self._static_rigid_sim_config
        )
        return _tensor

    def get_dofs_actuator_force(self, dofs_idx=None, envs_idx=None):
        """
        Generalized effort transmitted to each DOF at the actuator output (torque for revolute DOFs, force for
        prismatic DOFs), accounting for the gearbox losses between the motor and the joint.

        Computed as qf_applied - armature * qacc + qf_frictionloss + qf_passive: the commanded effort from
        get_dofs_control_force minus the armature-inertia load, plus the dissipative frictionloss and passive damping
        efforts. Contact, Coriolis and gravity loads are captured implicitly through the constraint-solved acceleration.
        """
        qf_applied = qd_to_torch(self.dofs_state.qf_applied, envs_idx, transpose=True)
        qacc = qd_to_torch(self.constraint_solver.qacc, envs_idx, transpose=True)
        qf_passive = qd_to_torch(self.dofs_state.qf_passive, envs_idx, transpose=True)
        if self._options.batch_dofs_info:
            armature = qd_to_torch(self.dofs_info.armature, envs_idx, transpose=True)
            frictionloss = qd_to_torch(self.dofs_info.frictionloss, envs_idx, transpose=True)
        else:
            armature = qd_to_torch(self.dofs_info.armature, transpose=True)
            frictionloss = qd_to_torch(self.dofs_info.frictionloss, transpose=True)

        # Frictionloss constraint forces mapped back to DOF space. Frictionloss constraints occupy the contiguous block
        # [n_constraints_equality, n_constraints_equality + n_constraints_frictionloss) of the constraint list and have
        # an identity Jacobian, so `efc_force` at a frictionloss row is exactly the DOF-space frictionloss effort. The
        # assembly loop appends them in ascending DOF order (it iterates links -> joints -> DOFs serially within each
        # env, matching the global DOF numbering), so the k-th frictionloss row is the k-th DOF with nonzero
        # frictionloss. Its row index is therefore `n_constraints_equality + rank`, with `rank` the running count of
        # frictionloss-enabled DOFs (-1 for DOFs without frictionloss, which contribute zero).
        efc_force = qd_to_torch(self.constraint_solver.efc_force, envs_idx, transpose=True)
        n_constraints_equality = qd_to_torch(self.constraint_solver.n_constraints_equality, envs_idx)
        has_frictionloss = frictionloss > gs.EPS
        rank = torch.cumsum(has_frictionloss, dim=-1) - 1
        gather_idx = (n_constraints_equality[:, None] + rank).clamp_(min=0)
        qf_frictionloss = torch.gather(efc_force, 1, gather_idx) * has_frictionloss

        actuator_force = qf_applied - armature * qacc + qf_frictionloss + qf_passive
        if dofs_idx is not None:
            actuator_force = actuator_force[indices_to_mask(None, dofs_idx)]
        return actuator_force[0] if self.n_envs == 0 else actuator_force

    def get_dofs_force(self, dofs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dofs_state.force, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_kp(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        gain = qd_to_torch(self.dofs_info.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        bias = qd_to_torch(self.dofs_info.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            gain, bias = gain[0], bias[0]
        if not (torch.abs(gain + bias[..., 1]) < gs.EPS * torch.clamp(torch.abs(gain), min=1.0)).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_gain != -act_bias[1]). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        if not (torch.abs(bias[..., 0]) < gs.EPS).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_bias[0] != 0). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        return gain

    def get_dofs_kv(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        gain = qd_to_torch(self.dofs_info.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        bias = qd_to_torch(self.dofs_info.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            gain, bias = gain[0], bias[0]
        if not (torch.abs(gain + bias[..., 1]) < gs.EPS * torch.clamp(torch.abs(gain), min=1.0)).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_gain != -act_bias[1]). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        if not (torch.abs(bias[..., 0]) < gs.EPS).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_bias[0] != 0). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        return -bias[..., 2]

    def get_dofs_act_gain(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_act_bias(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1], tensor[..., 2]

    def get_dofs_force_range(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.force_range, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_stiffness(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.stiffness, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_invweight(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.invweight, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_armature(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.armature, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_damping(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.damping, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_frictionloss(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.frictionloss, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_mass_mat(self, dofs_idx=None, envs_idx=None, decompose=False):
        tensor = qd_to_torch(self.mass_mat_L if decompose else self.mass_mat, envs_idx, transpose=True, copy=True)
        if dofs_idx is not None:
            tensor = tensor[indices_to_mask(None, dofs_idx, dofs_idx)]
        if self.n_envs == 0:
            tensor = tensor[0]

        if decompose:
            mass_mat_D_inv = qd_to_torch(
                self._rigid_global_info.mass_mat_D_inv, envs_idx, dofs_idx, transpose=True, copy=True
            )
            if self.n_envs == 0:
                mass_mat_D_inv = mass_mat_D_inv[0]
            return tensor, mass_mat_D_inv

        return tensor

    def get_geoms_friction(self, geoms_idx=None):
        return qd_to_torch(self.geoms_info.friction, geoms_idx, copy=True)

    def get_AABB(self, entities_idx=None, envs_idx=None):
        from genesis.engine.couplers import LegacyCoupler

        if not isinstance(self.sim.coupler, LegacyCoupler):
            gs.raise_exception("Method only supported when using 'LegacyCoupler' coupler type.")

        aabb_min = qd_to_torch(self.geoms_state.aabb_min, envs_idx, transpose=True)
        aabb_max = qd_to_torch(self.geoms_state.aabb_max, envs_idx, transpose=True)

        aabb = torch.stack([aabb_min, aabb_max], dim=-2)

        if entities_idx is not None:
            entity_geom_starts = []
            entity_geom_ends = []
            for entity_idx in entities_idx:
                entity = self._entities[entity_idx]
                entity_geom_starts.append(entity._geom_start)
                entity_geom_ends.append(entity._geom_start + entity.n_geoms)

            entity_aabbs = []
            for start, end in zip(entity_geom_starts, entity_geom_ends):
                if start < end:
                    entity_geoms_aabb = aabb[..., start:end, :, :]
                    entity_min = entity_geoms_aabb[..., :, 0, :].min(dim=-2)[0]
                    entity_max = entity_geoms_aabb[..., :, 1, :].max(dim=-2)[0]
                    entity_aabb = torch.stack([entity_min, entity_max], dim=-2)
                else:
                    entity_aabb = torch.zeros_like(aabb[..., 0:1, :, :])
                entity_aabbs.append(entity_aabb)

            aabb = torch.stack(entity_aabbs, dim=-2)

        return aabb[0] if self.n_envs == 0 else aabb

    def set_geom_friction(self, friction, geoms_idx):
        kernel_set_geom_friction(geoms_idx, friction, self.geoms_info)

    def set_geoms_friction(self, friction, geoms_idx=None):
        friction, geoms_idx, _ = self._sanitize_io_variables(
            friction, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, batched=False, skip_allocation=True
        )
        kernel_set_geoms_friction(friction, geoms_idx, self.geoms_info, self._static_rigid_sim_config)

    def add_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        return self.constraint_solver.add_weld_constraint(link1_idx, link2_idx, envs_idx)

    def delete_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        return self.constraint_solver.delete_weld_constraint(link1_idx, link2_idx, envs_idx)

    def get_weld_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_weld_constraints(as_tensor, to_torch)

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_equality_constraints(as_tensor, to_torch)

    def clear_external_force(self):
        if gs.use_zerocopy:
            for tensor in (self.links_state.cfrc_applied_ang, self.links_state.cfrc_applied_vel):
                out = qd_to_torch(tensor, copy=False)
                out.zero_()
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        kernel_clear_external_force(self.links_state, self._rigid_global_info, self._static_rigid_sim_config)

    @gs.assert_built
    def set_gravity(self, gravity, envs_idx=None):
        super().set_gravity(gravity, envs_idx)
        if hasattr(self, "_rigid_global_info"):
            self._rigid_global_info.gravity.copy_from(self._gravity)

    def update_drone_propeller_vgeoms(self, propellers_vgeom_idxs, propellers_revs, propellers_spin):
        kernel_update_drone_propeller_vgeoms(
            propellers_vgeom_idxs,
            propellers_revs,
            propellers_spin,
            self.vgeoms_state,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

    def set_drone_rpm(self, propellers_link_idx, propellers_rpm, propellers_spin, KF, KM, invert):
        kernel_set_drone_rpm(
            propellers_link_idx,
            propellers_rpm,
            propellers_spin,
            KF,
            KM,
            invert,
            self.links_state,
            self._static_rigid_sim_config,
        )

    def update_verts_for_geoms(self, geoms_idx):
        _, geoms_idx, _ = self._sanitize_io_variables(
            None, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, skip_allocation=True
        )
        kernel_update_verts_for_geoms(
            geoms_idx,
            self.geoms_state,
            self.geoms_info,
            self.verts_info,
            self.free_verts_state,
            self.fixed_verts_state,
            self._static_rigid_sim_config,
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_geoms(self):
        if self.is_built:
            return self._n_geoms
        return len(self.geoms)

    @property
    def n_cells(self):
        if self.is_built:
            return self._n_cells
        return sum(entity.n_cells for entity in self._entities)

    @property
    def n_verts(self):
        if self.is_built:
            return self._n_verts
        return sum(entity.n_verts for entity in self._entities)

    @property
    def n_free_verts(self):
        if self.is_built:
            return self._n_free_verts
        return sum(link.n_verts if not link.is_fixed or link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_fixed_verts(self):
        if self.is_built:
            return self._n_fixed_verts
        return sum(link.n_verts if link.is_fixed and not link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_faces(self):
        if self.is_built:
            return self._n_faces
        return sum(entity.n_faces for entity in self._entities)

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        return sum(entity.n_edges for entity in self._entities)

    @property
    def max_collision_pairs(self):
        return self._max_collision_pairs

    @property
    def n_equalities(self):
        if self.is_built:
            return self._n_equalities
        return sum(entity.n_equalities for entity in self._entities)

    @property
    def equalities(self):
        if self.is_built:
            return self._equalities
        return gs.List(equality for entity in self._entities for equality in entity.equalities)


@qd.kernel(fastcache=True)
def kernel_step_1(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    island_state: array_class.IslandState,
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
    is_backward: qd.template(),
):
    if qd.static(not is_forward_pos_updated):
        func_update_cartesian_space(
            links_state=links_state,
            links_info=links_info,
            joints_state=joints_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            force_update_fixed_geoms=False,
            is_backward=is_backward,
        )

    if qd.static(not is_forward_vel_updated):
        func_forward_velocity(
            entities_info=entities_info,
            links_info=links_info,
            links_state=links_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )

    func_forward_dynamics(
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        joints_info=joints_info,
        entities_state=entities_state,
        entities_info=entities_info,
        geoms_state=geoms_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        island_state=island_state,
        is_backward=is_backward,
    )


@qd.kernel(fastcache=True)
def kernel_step_2(
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    joints_state: array_class.JointsState,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    island_state: array_class.IslandState,
    is_backward: qd.template(),
    errno: qd.Tensor,
):
    # Position, Velocity and Acceleration data must be consistent when computing links acceleration, otherwise it
    # would not corresponds to anyting physical. There is no other way than doing this right before integration,
    # because the acceleration at the end of the step is unknown for now as it may change discontinuous between
    # before and after integration under the effect of external forces and constraints. This means that
    # acceleration data will be shifted one timestep in the past, but there isn't really any way around.
    func_update_acc(
        update_cacc=True,
        dofs_state=dofs_state,
        links_info=links_info,
        links_state=links_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )

    if qd.static(static_rigid_sim_config.integrator != gs.integrator.approximate_implicitfast):
        func_implicit_damping(
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )

    func_integrate(
        dofs_state=dofs_state,
        links_info=links_info,
        joints_info=joints_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )

    if qd.static(static_rigid_sim_config.use_hibernation):
        func_hibernate__for_all_awake_islands_either_hiberanate_or_update_aabb_sort_buffer(
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            entities_state=entities_state,
            entities_info=entities_info,
            links_info=links_info,
            links_state=links_state,
            geoms_state=geoms_state,
            collider_state=collider_state,
            unused__rigid_global_info=rigid_global_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            island_state=island_state,
            errno=errno,
        )
        func_aggregate_awake_entities(
            entities_state=entities_state,
            entities_info=entities_info,
            links_info=links_info,
            links_state=links_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

    if qd.static(not is_backward):
        func_copy_next_to_curr(
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            errno=errno,
        )

        if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
            func_update_cartesian_space(
                links_state=links_state,
                links_info=links_info,
                joints_state=joints_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                dofs_info=dofs_info,
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                force_update_fixed_geoms=False,
                is_backward=is_backward,
            )
            func_forward_velocity(
                entities_info=entities_info,
                links_info=links_info,
                links_state=links_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=is_backward,
            )
