import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

import numpy as np
import quadrants as qd
import torch
import torch.nn.functional as F

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.solvers.rigid.collider.utils import func_point_in_geom_aabb
from genesis.options.sensors import ContactDepthProbe as ContactDepthProbeOptions
from genesis.options.sensors import ContactProbe as ContactProbeOptions
from genesis.options.sensors import KinematicTaxel as KinematicTaxelOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensor, SimpleSensorMetadata
from .probe import (
    ProbeSensorMetadataMixin,
    ProbeSensorMixin,
    ProbesWithNormalSensorMetadataMixin,
    ProbesWithNormalSensorMixin,
    ProbesWithNormalSensorSharedMetadataT,
    func_noised_probe_radius,
    get_measured_bufs,
)
from .tactile_shared import (
    ContactPrefilterMetadataMixin,
    ViscoelasticHysteresisMetadataMixin,
    ViscoelasticHysteresisMixin,
    normalize_grid_probe_layout,
)

if TYPE_CHECKING:
    from genesis.options.sensors import SensorOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@qd.func
def _func_query_contact_depth_penetration(
    i_b: int,
    i_s: int,
    probe_pos: qd.types.vector(3),
    probe_radius_gt: float,
    probe_radius_m: float,
    sensor_link_idx: int,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    collider_state: array_class.ColliderState,
    sensor_contacts_idx: qd.types.ndarray(),
    sensor_n_contacts: qd.types.ndarray(),
    sdf_info: array_class.SDFInfo,
):
    """
    Max probe penetration from SDF for contacts involving the sensor link, dual-radius.
    """
    max_pen_gt = gs.qd_float(0.0)
    max_pen_m = gs.qd_float(0.0)

    n_c = sensor_n_contacts[i_b, i_s]
    for k in range(n_c):
        i_c = sensor_contacts_idx[i_b, i_s, k]
        c_link_a = collider_state.contact_data.link_a[i_c, i_b]
        c_link_b = collider_state.contact_data.link_b[i_c, i_b]
        c_geom_a = collider_state.contact_data.geom_a[i_c, i_b]
        c_geom_b = collider_state.contact_data.geom_b[i_c, i_b]

        for side in qd.static(range(2)):
            c_link = c_link_a if side == 0 else c_link_b
            i_g = c_geom_b if side == 0 else c_geom_a

            if c_link == sensor_link_idx:
                g_pos = geoms_state.pos[i_g, i_b]
                g_quat = geoms_state.quat[i_g, i_b]
                sd = sdf.sdf_func_world_local(geoms_info, sdf_info, probe_pos, i_g, g_pos, g_quat)
                pen_gt = probe_radius_gt - sd
                if pen_gt > max_pen_gt:
                    max_pen_gt = pen_gt
                pen_m = probe_radius_m - sd
                if pen_m > max_pen_m:
                    max_pen_m = pen_m

    return max_pen_gt, max_pen_m


# Per-(env, sensor) cap on the prefiltered contact list consumed by ``_func_query_contact_depth``
# and ``_func_query_contact_depth_penetration``. Sensors track a single rigid link; even with multicontact
# and many neighbouring geoms, the count of contacts touching one link rarely exceeds a few hundred.
_MAX_CONTACTS_PER_SENSOR = 1024


@qd.kernel
def _kernel_build_sensor_contact_idx(
    sensor_link_idx: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
    sensor_contacts_idx: qd.types.ndarray(),
    sensor_n_contacts: qd.types.ndarray(),
):
    """
    Per-(env, sensor) compact contact index for the KinematicTaxel pre-pass.

    Parallelizes over ``(n_batches, n_sensors)``: each thread scans the collider's contact list once and writes
    the indices of contacts whose ``link_a`` or ``link_b`` equals the sensor's tracked link. Drops the main
    kernel's per-probe contact-list scan from O(n_probes * n_contacts) to O(n_probes * sensor_n_contacts).
    Cap-overflows (count >= last dim of ``sensor_contacts_idx``) silently truncate; see the module-level
    ``_MAX_CONTACTS_PER_SENSOR`` comment.
    """
    n_sensors = sensor_link_idx.shape[0]
    n_batches = sensor_n_contacts.shape[0]
    max_per_sensor = sensor_contacts_idx.shape[2]
    for i_b, i_s in qd.ndrange(n_batches, n_sensors):
        link = sensor_link_idx[i_s]
        count = gs.qd_int(0)
        n_c = collider_state.n_contacts[i_b]
        for i_c in range(n_c):
            if count >= max_per_sensor:
                break
            la = collider_state.contact_data.link_a[i_c, i_b]
            lb = collider_state.contact_data.link_b[i_c, i_b]
            if la == link or lb == link:
                sensor_contacts_idx[i_b, i_s, count] = i_c
                count = count + 1
        sensor_n_contacts[i_b, i_s] = count


@qd.func
def _func_query_contact_depth(
    i_b: int,
    i_s: int,
    probe_pos: qd.types.vector(3),
    probe_radius_gt: float,
    probe_radius_m: float,
    sensor_link_idx: int,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    collider_state: array_class.ColliderState,
    sensor_contacts_idx: qd.types.ndarray(),
    sensor_n_contacts: qd.types.ndarray(),
    sdf_info: array_class.SDFInfo,
    eps: float,
):
    """
    Dual-radius probe query: single SDF + normal pass yielding both GT and noised-radius results.

    Iterates only the per-(env, sensor) prefiltered contact list built by ``_kernel_build_sensor_contact_idx``;
    every contact in that list has ``link_a`` or ``link_b`` equal to ``sensor_link_idx``. AABB pre-filter
    expands by ``max(probe_radius_gt, probe_radius_m)`` so neither branch is silently skipped. Callers without
    a noised radius pass ``probe_radius_m == probe_radius_gt``.
    """
    max_pen_gt = gs.qd_float(0.0)
    contact_link_gt = gs.qd_int(-1)
    contact_normal_gt = qd.Vector.zero(gs.qd_float, 3)
    max_pen_m = gs.qd_float(0.0)
    contact_link_m = gs.qd_int(-1)
    contact_normal_m = qd.Vector.zero(gs.qd_float, 3)

    aabb_expansion = qd.max(probe_radius_gt, probe_radius_m)
    n_c = sensor_n_contacts[i_b, i_s]
    for k in range(n_c):
        i_c = sensor_contacts_idx[i_b, i_s, k]
        c_link_a = collider_state.contact_data.link_a[i_c, i_b]
        c_link_b = collider_state.contact_data.link_b[i_c, i_b]
        c_geom_a = collider_state.contact_data.geom_a[i_c, i_b]
        c_geom_b = collider_state.contact_data.geom_b[i_c, i_b]

        # Each prefiltered contact touches the sensor link on at least one side; check both since the link
        # may appear as link_a, link_b, or (degenerately) both.
        for side in qd.static(range(2)):
            c_link = c_link_a if side == 0 else c_link_b
            i_g = c_geom_b if side == 0 else c_geom_a

            if c_link == sensor_link_idx and func_point_in_geom_aabb(geoms_state, i_g, i_b, probe_pos, aabb_expansion):
                g_pos = geoms_state.pos[i_g, i_b]
                g_quat = geoms_state.quat[i_g, i_b]
                sd = sdf.sdf_func_world_local(geoms_info, sdf_info, probe_pos, i_g, g_pos, g_quat)
                pen_gt = probe_radius_gt - sd
                pen_m = probe_radius_m - sd
                # Compute the SDF normal at most once across both branches.
                need_normal = (pen_gt > max_pen_gt and pen_gt > eps) or (pen_m > max_pen_m and pen_m > eps)
                if need_normal:
                    normal = sdf.sdf_func_normal_world_local(
                        geoms_info, rigid_global_info, collider_static_config, sdf_info, probe_pos, i_g, g_pos, g_quat
                    )
                    if pen_gt > max_pen_gt and pen_gt > eps:
                        max_pen_gt = pen_gt
                        contact_link_gt = c_link_b if side == 0 else c_link_a
                        contact_normal_gt = normal
                    if pen_m > max_pen_m and pen_m > eps:
                        max_pen_m = pen_m
                        contact_link_m = c_link_b if side == 0 else c_link_a
                        contact_normal_m = normal

    return max_pen_gt, contact_link_gt, contact_normal_gt, max_pen_m, contact_link_m, contact_normal_m


@qd.func
def _func_kinematic_spring_damper(
    i_b: int,
    max_penetration: float,
    contact_link: int,
    contact_normal: qd.types.vector(3),
    sensor_link_idx: int,
    probe_pos: qd.types.vector(3),
    probe_pos_local: qd.types.vector(3),
    link_quat: qd.types.vector(4),
    normal_stiffness: float,
    normal_damping: float,
    normal_exponent: float,
    shear_scalar: float,
    twist_scalar: float,
    links_state: array_class.LinksState,
):
    """
    Kinematic spring-damper force / torque in the sensor link frame from a single probe's contact query.

    Shared by the GT and measured branches of ``_kernel_kinematic_taxel`` (they differ only in which dual-radius
    query result is fed in). Returns ``(force_local, torque_local)``; both zero when ``max_penetration <= 0``.
    """
    force_local = qd.Vector.zero(gs.qd_float, 3)
    torque_local = qd.Vector.zero(gs.qd_float, 3)
    if max_penetration > 0:
        contact_normal_local = gu.qd_inv_transform_by_quat(contact_normal, link_quat)
        s = qd.pow(max_penetration, normal_exponent)
        force_local = contact_normal_local * (normal_stiffness * s)

        if contact_link >= 0:
            contact_vel = links_state.cd_vel[contact_link, i_b] + links_state.cd_ang[contact_link, i_b].cross(
                probe_pos - links_state.root_COM[contact_link, i_b]
            )
            sensor_vel = links_state.cd_vel[sensor_link_idx, i_b] + links_state.cd_ang[sensor_link_idx, i_b].cross(
                probe_pos - links_state.root_COM[sensor_link_idx, i_b]
            )
            rel_vel_world = contact_vel - sensor_vel
            rel_vel_local = gu.qd_inv_transform_by_quat(rel_vel_world, link_quat)

            vn_dot = rel_vel_local.dot(contact_normal_local)
            v_t_local = rel_vel_local - contact_normal_local * vn_dot
            force_local += contact_normal_local * (normal_damping * s * vn_dot) - shear_scalar * v_t_local

            rel_ang_world = links_state.cd_ang[contact_link, i_b] - links_state.cd_ang[sensor_link_idx, i_b]
            omega_n = rel_ang_world.dot(contact_normal)
            torque_local = probe_pos_local.cross(force_local) - contact_normal_local * (twist_scalar * omega_n)
        else:
            torque_local = probe_pos_local.cross(force_local)

    return force_local, torque_local


@qd.kernel
def _kernel_kinematic_taxel(
    probe_positions_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    probe_gains: qd.types.ndarray(),
    normal_stiffness: qd.types.ndarray(),
    normal_damping: qd.types.ndarray(),
    normal_exponent: qd.types.ndarray(),
    shear_scalar: qd.types.ndarray(),
    twist_scalar: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    sensor_contacts_idx: qd.types.ndarray(),
    sensor_n_contacts: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
    collider_static_config: qd.template(),
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    sdf_info: array_class.SDFInfo,
    eps: float,
    measured_equals_gt: int,
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]
        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        force_start = cache_start + probe_idx_in_sensor * 3
        torque_start = cache_start + n_probes * 3 + probe_idx_in_sensor * 3

        # Inactive filler probe (probe_radius == 0): reads zero force/torque, no contact query.
        if probe_radii[i_p] <= gs.qd_float(0.0):
            for j in qd.static(range(3)):
                output_gt[force_start + j, i_b] = gs.qd_float(0.0)
                output_gt[torque_start + j, i_b] = gs.qd_float(0.0)
                output_measured[force_start + j, i_b] = gs.qd_float(0.0)
                output_measured[torque_start + j, i_b] = gs.qd_float(0.0)
            continue

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )

        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        probe_radius = probe_radii[i_p]
        probe_radius_noise = probe_radii_noise[i_p]
        use_noised_radius = probe_radius_noise > eps
        probe_radius_m = (
            func_noised_probe_radius(probe_radius, probe_radius_noise) if use_noised_radius else probe_radius
        )

        (
            max_penetration_gt,
            contact_link_gt,
            contact_normal_gt,
            max_penetration_m,
            contact_link_m,
            contact_normal_m,
        ) = _func_query_contact_depth(
            i_b,
            i_s,
            probe_pos,
            probe_radius,
            probe_radius_m,
            sensor_link_idx,
            geoms_info,
            geoms_state,
            rigid_global_info,
            collider_static_config,
            collider_state,
            sensor_contacts_idx,
            sensor_n_contacts,
            sdf_info,
            eps,
        )

        force_local_gt, torque_local_gt = _func_kinematic_spring_damper(
            i_b,
            max_penetration_gt,
            contact_link_gt,
            contact_normal_gt,
            sensor_link_idx,
            probe_pos,
            probe_pos_local,
            link_quat,
            normal_stiffness[i_s],
            normal_damping[i_s],
            normal_exponent[i_s],
            shear_scalar[i_s],
            twist_scalar[i_s],
            links_state,
        )

        force_local_m = force_local_gt
        torque_local_m = torque_local_gt
        if measured_equals_gt == 0:
            # The measured branch differs from GT: either some probe has a noised sensing radius or a non-unit
            # per-(env, probe) gain. Gain scales the measured penetration only; force / torque then scale as
            # ``gain ** normal_exponent`` since they derive from ``s = max_penetration_m ** normal_exponent``.
            max_penetration_m = max_penetration_m * probe_gains[i_b, i_p]
            force_local_m, torque_local_m = _func_kinematic_spring_damper(
                i_b,
                max_penetration_m,
                contact_link_m,
                contact_normal_m,
                sensor_link_idx,
                probe_pos,
                probe_pos_local,
                link_quat,
                normal_stiffness[i_s],
                normal_damping[i_s],
                normal_exponent[i_s],
                shear_scalar[i_s],
                twist_scalar[i_s],
                links_state,
            )

        for j in qd.static(range(3)):
            output_gt[force_start + j, i_b] = force_local_gt[j]
            output_gt[torque_start + j, i_b] = torque_local_gt[j]
            output_measured[force_start + j, i_b] = force_local_m[j]
            output_measured[torque_start + j, i_b] = torque_local_m[j]


@qd.kernel
def _kernel_contact_depth_probe(
    probe_positions_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    probe_gains: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    sensor_contacts_idx: qd.types.ndarray(),
    sensor_n_contacts: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]

        # Inactive filler probe (probe_radius == 0): reads zero depth (which contact-probe interprets as no contact).
        if probe_radii[i_p] <= gs.qd_float(0.0):
            cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
            output_gt[cache_idx, i_b] = gs.qd_float(0.0)
            output_measured[cache_idx, i_b] = gs.qd_float(0.0)
            continue

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )

        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        probe_radius = probe_radii[i_p]
        probe_radius_noise = probe_radii_noise[i_p]
        probe_radius_m = (
            func_noised_probe_radius(probe_radius, probe_radius_noise) if probe_radius_noise > gs.EPS else probe_radius
        )

        max_penetration_gt, max_penetration_m = _func_query_contact_depth_penetration(
            i_b,
            i_s,
            probe_pos,
            probe_radius,
            probe_radius_m,
            sensor_link_idx,
            geoms_info,
            geoms_state,
            collider_state,
            sensor_contacts_idx,
            sensor_n_contacts,
            sdf_info,
        )
        # Per-(env, probe) gain on the measured-branch depth only.
        max_penetration_m = max_penetration_m * probe_gains[i_b, i_p]
        cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
        output_gt[cache_idx, i_b] = max_penetration_gt
        output_measured[cache_idx, i_b] = max_penetration_m


class KinematicTactileSensorMixin(ProbeSensorMixin[ProbesWithNormalSensorSharedMetadataT]):
    def build(self):
        super().build()
        self._shared_metadata.solver.collider.activate_sdf()


@dataclass
class ContactDepthProbeMetadata(
    ViscoelasticHysteresisMetadataMixin,
    ProbeSensorMetadataMixin,
    ContactPrefilterMetadataMixin,
    RigidSensorMetadataMixin,
    SimpleSensorMetadata,
):
    pass


class ContactDepthProbeSensor(
    ViscoelasticHysteresisMixin[ContactDepthProbeMetadata],
    KinematicTactileSensorMixin[ContactDepthProbeMetadata],
    RigidSensorMixin[ContactDepthProbeMetadata],
    SimpleSensor[ContactDepthProbeOptions, None, ContactDepthProbeMetadata, tuple],
):
    """
    Returns contact depth in meters per probe.
    """

    def build(self):
        super().build()
        # Re-allocate the per-(env, sensor) contact prefilter buffers to absorb the newly-registered sensor.
        B = self._manager._sim._B
        n_sensors_built = self._shared_metadata.n_probes_per_sensor.shape[0]
        self._shared_metadata.sensor_contacts_idx = torch.zeros(
            (B, n_sensors_built, _MAX_CONTACTS_PER_SENSOR), dtype=gs.tc_int, device=gs.device
        )
        self._shared_metadata.sensor_n_contacts = torch.zeros((B, n_sensors_built), dtype=gs.tc_int, device=gs.device)

    def _get_return_format(self) -> tuple[int, ...]:
        return self._probe_layout_shape

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_context: None,
        shared_metadata: ContactDepthProbeMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        measured, measured_cols_b = get_measured_bufs(
            shared_metadata, current_ground_truth_data_T, measured_data_timeline
        )
        _kernel_build_sensor_contact_idx(
            shared_metadata.links_idx,
            solver.collider._collider_state,
            shared_metadata.sensor_contacts_idx,
            shared_metadata.sensor_n_contacts,
        )
        _kernel_contact_depth_probe(
            shared_metadata.probe_positions,
            shared_metadata.probe_sensor_idx,
            shared_metadata.probe_radii,
            shared_metadata.probe_radii_noise,
            shared_metadata.probe_gains,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.sensor_contacts_idx,
            shared_metadata.sensor_n_contacts,
            solver.collider._collider_state,
            solver.links_state,
            solver.geoms_state,
            solver.geoms_info,
            solver.collider._sdf._sdf_info,
            current_ground_truth_data_T,
            measured_cols_b,
        )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext"):
        def mask(envs_idx):
            depth = self.read_ground_truth(envs_idx)
            if self._options.history_length > 0:
                depth = depth.select(1 if self._manager._sim.n_envs > 0 else 0, -1)
            return depth >= gs.EPS

        self._draw_debug_probes(context, self._tactile_color_groups_fn(mask))


@dataclass
class ContactProbeMetadata(ContactDepthProbeMetadata):
    contact_threshold: torch.Tensor = make_tensor_field((0,))
    release_threshold: torch.Tensor = make_tensor_field((0,))
    # Per-probe thresholds scattered into intermediate-cache layout, computed lazily on first `_post_process`.
    threshold_row: torch.Tensor = make_tensor_field((0,))
    release_threshold_row: torch.Tensor = make_tensor_field((0,))


class ContactProbeSensor(ContactDepthProbeSensor, SimpleSensor[ContactProbeOptions, None, ContactProbeMetadata, tuple]):
    """
    Returns boolean contact per probe with optional Schmitt-trigger hysteresis. Shares the depth-probe kernel.

    The contact bit latches on when depth exceeds ``contact_threshold`` and releases when depth drops to or below
    ``release_threshold``. When ``release_threshold`` is left unset (the default; it then falls back to
    ``contact_threshold``), the latch is degenerate and behavior matches a stateless threshold. Latch state is read
    from the per-branch return-space ring, so GT and measured branches latch independently and reset cleanly with
    the env (the manager zeros the ring on reset).
    """

    def build(self):
        super().build()
        self._shared_metadata.contact_threshold = concat_with_tensor(
            self._shared_metadata.contact_threshold, self._options.contact_threshold, expand=(1,)
        )
        release = (
            self._options.contact_threshold
            if self._options.release_threshold is None
            else self._options.release_threshold
        )
        self._shared_metadata.release_threshold = concat_with_tensor(
            self._shared_metadata.release_threshold, release, expand=(1,)
        )

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_bool

    @classmethod
    def _get_intermediate_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _post_process(
        cls,
        shared_metadata: ContactProbeMetadata,
        tensor: torch.Tensor,
        timeline: "TensorRingBuffer",
        *,
        is_measured: bool,
    ) -> torch.Tensor:
        if (
            shared_metadata.threshold_row.shape != (tensor.shape[1],)
            or shared_metadata.threshold_row.dtype != tensor.dtype
        ):
            i_p = torch.arange(shared_metadata.total_n_probes, device=gs.device, dtype=gs.tc_int)
            i_s = shared_metadata.probe_sensor_idx
            cache_idx = shared_metadata.sensor_cache_start[i_s] + i_p - shared_metadata.sensor_probe_start[i_s]
            cache_idx_64 = cache_idx.to(dtype=torch.int64)
            enter_row = torch.zeros((tensor.shape[1],), dtype=tensor.dtype, device=gs.device)
            enter_row.scatter_(0, cache_idx_64, shared_metadata.contact_threshold[i_s].to(dtype=tensor.dtype))
            release_row = torch.zeros((tensor.shape[1],), dtype=tensor.dtype, device=gs.device)
            release_row.scatter_(0, cache_idx_64, shared_metadata.release_threshold[i_s].to(dtype=tensor.dtype))
            shared_metadata.threshold_row = enter_row
            shared_metadata.release_threshold_row = release_row
        above_enter = tensor > shared_metadata.threshold_row.unsqueeze(0)
        above_release = tensor > shared_metadata.release_threshold_row.unsqueeze(0)
        prev_state = timeline.at(0, copy=False)
        return above_enter | (prev_state & above_release)

    def _draw_debug(self, context: "RasterizerContext"):
        def mask(envs_idx):
            contact = self.read_ground_truth(envs_idx)
            if self._options.history_length > 0:
                contact = contact.select(1 if self._manager._sim.n_envs > 0 else 0, -1)
            return contact

        self._draw_debug_probes(context, self._tactile_color_groups_fn(mask))


class KinematicTaxelReturnType(NamedTuple):
    """
    Parameters
    ----------
    force: torch.Tensor, shape ([n_envs,] n_probes, 3)
        Estimated contact force in the link frame from the kinematic spring-damper model.
    torque: torch.Tensor, shape ([n_envs,] n_probes, 3)
    """

    force: torch.Tensor
    torque: torch.Tensor


@dataclass
class KinematicTaxelMetadata(
    ViscoelasticHysteresisMetadataMixin,
    ProbesWithNormalSensorMetadataMixin,
    ContactPrefilterMetadataMixin,
    RigidSensorMetadataMixin,
    SimpleSensorMetadata,
):
    normal_stiffness: torch.Tensor = make_tensor_field((0,))
    normal_damping: torch.Tensor = make_tensor_field((0,))
    normal_exponent: torch.Tensor = make_tensor_field((0,))
    shear_scalar: torch.Tensor = make_tensor_field((0,))
    twist_scalar: torch.Tensor = make_tensor_field((0,))

    # Per-sensor spatial crosstalk state. ``crosstalk_meta[i]`` is
    # ``(g_ny, g_nx, probe_start, cache_start, strength, r_v, r_u)``; ``crosstalk_kernels_{v,u}`` hold the matching
    # depthwise 1D Gaussian weights pre-shaped for ``F.conv2d`` with ``groups=6``. Applied as two separable
    # passes (kv then ku) with an identity blend ``(1 - strength) * x + strength * conv(x)``.
    crosstalk_meta: list[tuple] = field(default_factory=list)
    crosstalk_kernels_v: list[torch.Tensor] = field(default_factory=list)
    crosstalk_kernels_u: list[torch.Tensor] = field(default_factory=list)
    any_crosstalk: bool = False


def _build_separable_crosstalk_kernels(
    sigma: float,
    spacing_u: float,
    spacing_v: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build two L1-normalized depthwise 1D Gaussian kernels for separable crosstalk convolution.

    A 2D isotropic Gaussian is the outer product of two 1D Gaussians, so it is applied as two sequential 1D
    convolutions (one per axis) on the unpadded active grid. Each 1D kernel has half-radius
    ``r = ceil(3 * sigma / spacing)`` (3-sigma truncation; tail leakage below ~0.3%) and is L1-normalized after
    truncation so a uniform field passes through unchanged.

    Returns ``(kernel_v, kernel_u, r_v, r_u)``. The kernels are pre-shaped for ``F.conv2d`` with ``groups=6``:
    ``kernel_v`` has shape ``(6, 1, 2*r_v + 1, 1)`` (axis 0 of the field, the v / ny axis) and ``kernel_u`` has
    shape ``(6, 1, 1, 2*r_u + 1)`` (axis 1 of the field, the u / nx axis). The kernel is replicated across all 6
    channels (force xyz + torque xyz) so a single ``groups=6`` call covers everything.
    """
    r_v = max(1, int(math.ceil(3.0 * sigma / spacing_v)))
    r_u = max(1, int(math.ceil(3.0 * sigma / spacing_u)))
    iv = torch.arange(2 * r_v + 1, dtype=dtype, device=device) - r_v
    iu = torch.arange(2 * r_u + 1, dtype=dtype, device=device) - r_u
    gv = torch.exp(-((iv * spacing_v) ** 2) / (2.0 * sigma * sigma))
    gu = torch.exp(-((iu * spacing_u) ** 2) / (2.0 * sigma * sigma))
    gv = gv / gv.sum()
    gu = gu / gu.sum()
    kernel_v = gv.view(1, 1, -1, 1).repeat(6, 1, 1, 1)
    kernel_u = gu.view(1, 1, 1, -1).repeat(6, 1, 1, 1)
    return kernel_v, kernel_u, r_v, r_u


def _kinematic_taxel_grid_separable_crosstalk(
    crosstalk_meta: list[tuple],
    crosstalk_kernels_v: list[torch.Tensor],
    crosstalk_kernels_u: list[torch.Tensor],
    cache_data: torch.Tensor,
    probe_radii: torch.Tensor,
) -> None:
    """
    Apply per-sensor 2D Gaussian spatial crosstalk to all 6 channels (force xyz + torque xyz) of every registered
    grid-crosstalk KinematicTaxel sensor as two depthwise 1D convolutions (separable). Mutates ``cache_data`` in
    place.

    ``cache_data`` is the per-class intermediate cache in ``(B, total_cols)`` layout. Each sensor's slice spans
    ``2 * n_probes * 3`` columns: 3 force xyz cols per probe, then 3 torque xyz cols per probe, both probe-major
    (probe flat index ``iy * nx + ix``). Peak per-sensor working memory is ~3 * (B * 6 * g_ny * g_nx * sizeof(float)):
    the input field, the intermediate after the first pass, and the blurred output. No persistent buffer is held.
    """
    if not crosstalk_meta:
        return
    B = cache_data.shape[0]
    for (g_ny, g_nx, probe_start, cache_start, strength, r_v, r_u), kv, ku in zip(
        crosstalk_meta, crosstalk_kernels_v, crosstalk_kernels_u
    ):
        n_probes = g_ny * g_nx
        # Build a contiguous (B, 6, g_ny, g_nx) field: force xyz then torque xyz stacked on the channel axis. The
        # cache holds probe-major ``iy * nx + ix`` ordering, so reshape (B, n_probes, 3) -> (B, ny, nx, 3) and
        # permute -> (B, 3, ny, nx) per group, then concat along channels. ``.contiguous()`` materializes the
        # permuted layout that ``F.conv2d`` needs.
        force_block = cache_data[:, cache_start : cache_start + n_probes * 3]
        torque_block = cache_data[:, cache_start + n_probes * 3 : cache_start + 2 * n_probes * 3]
        force = force_block.view(B, g_ny, g_nx, 3).permute(0, 3, 1, 2)
        torque = torque_block.view(B, g_ny, g_nx, 3).permute(0, 3, 1, 2)
        field_in = torch.cat((force, torque), dim=1).contiguous()  # (B, 6, g_ny, g_nx)

        # Depthwise separable convolution: one Gaussian per channel via groups=6. ``padding='zeros'`` (the default)
        # means out-of-grid taps contribute zero, so probes near the edge see no spurious mass from beyond the
        # sensor footprint.
        tmp = F.conv2d(field_in, kv, groups=6, padding=(r_v, 0))
        blurred = F.conv2d(tmp, ku, groups=6, padding=(0, r_u))

        # Identity blend: out = (1 - strength) * field_in + strength * blurred. Mathematically equivalent to
        # convolving with ``(1 - strength) * delta + strength * Gaussian``.
        out = field_in.mul_(1.0 - strength).add_(blurred, alpha=strength)

        # Zero inactive filler probes (probe_radius == 0): the blur leaks neighbour force/torque into their cells.
        active = (probe_radii[probe_start : probe_start + n_probes] > 0.0).to(out.dtype).view(1, 1, g_ny, g_nx)
        out.mul_(active)

        # Inverse of the build permute: (B, 3, ny, nx) -> (B, ny, nx, 3) -> flat (B, ny*nx*3).
        cache_data[:, cache_start : cache_start + n_probes * 3] = (
            out[:, 0:3].permute(0, 2, 3, 1).reshape(B, n_probes * 3)
        )
        cache_data[:, cache_start + n_probes * 3 : cache_start + 2 * n_probes * 3] = (
            out[:, 3:6].permute(0, 2, 3, 1).reshape(B, n_probes * 3)
        )


CrosstalkSharedMetadataT = TypeVar("CrosstalkSharedMetadataT", bound=KinematicTaxelMetadata)


class KinematicTaxelCrosstalkMixin(Generic[CrosstalkSharedMetadataT]):
    """
    Adds Gaussian spatial crosstalk (optionally mixed with identity) to KinematicTaxel on the measured branch.
    Operates on all 6 channels (force xyz + torque xyz) of every grid-shaped sensor with ``crosstalk_strength > 0``.
    Must come BEFORE ``SimpleSensor`` and AFTER ``ViscoelasticHysteresisMixin`` in MRO so the data flow is:
    kernel output -> crosstalk -> hysteresis -> hardware imperfections.
    """

    _shared_metadata: CrosstalkSharedMetadataT

    def _register_crosstalk(self):
        """Build this sensor's separable Gaussian crosstalk kernels and append them to the shared metadata lists.

        Called only when this sensor has a validated grid layout AND ``crosstalk_strength > 0``. Stores two
        L1-normalized 1D Gaussians (truncated at 3 sigma on each axis, pre-shaped for depthwise ``F.conv2d`` with
        ``groups=6``); no persistent per-step buffer is allocated.
        """
        sm = self._shared_metadata
        sensor_idx = sm.n_probes_per_sensor.shape[0] - 1  # this sensor was just registered
        probe_start = int(sm.sensor_probe_start[sensor_idx].item())
        cache_start = int(sm.sensor_cache_start[sensor_idx].item())
        g_ny, g_nx = int(self._probe_layout_shape[0]), int(self._probe_layout_shape[1])
        sigma = float(self._options.crosstalk_sigma)
        strength = float(self._options.crosstalk_strength)
        spacing_u = float(self._grid_spacing[0].item())
        spacing_v = float(self._grid_spacing[1].item())
        kernel_v, kernel_u, r_v, r_u = _build_separable_crosstalk_kernels(
            sigma, spacing_u, spacing_v, gs.device, gs.tc_float
        )
        sm.crosstalk_meta.append((g_ny, g_nx, probe_start, cache_start, strength, r_v, r_u))
        sm.crosstalk_kernels_v.append(kernel_v)
        sm.crosstalk_kernels_u.append(kernel_u)
        sm.any_crosstalk = True

    @classmethod
    def _apply_transform(
        cls,
        shared_metadata: CrosstalkSharedMetadataT,
        data: torch.Tensor,
        timeline: "TensorRingBuffer",
        *,
        is_measured: bool,
    ):
        super()._apply_transform(shared_metadata, data, timeline, is_measured=is_measured)
        if not is_measured or not shared_metadata.any_crosstalk:
            return
        _kinematic_taxel_grid_separable_crosstalk(
            shared_metadata.crosstalk_meta,
            shared_metadata.crosstalk_kernels_v,
            shared_metadata.crosstalk_kernels_u,
            data,
            shared_metadata.probe_radii,
        )


class KinematicTaxelSensor(
    ViscoelasticHysteresisMixin[KinematicTaxelMetadata],
    KinematicTaxelCrosstalkMixin[KinematicTaxelMetadata],
    KinematicTactileSensorMixin[KinematicTaxelMetadata],
    ProbesWithNormalSensorMixin[KinematicTaxelMetadata],
    RigidSensorMixin[KinematicTaxelMetadata],
    SimpleSensor[KinematicTaxelOptions, None, KinematicTaxelMetadata, KinematicTaxelReturnType],
):
    """Kinematic taxels: spring-damper force and torque per probe from contact geometry and relative motion."""

    # Two channel groups: force xyz followed by torque xyz (probe-major within each group). See
    # ``ProbeSensorMixin._taxel_channel_groups`` for how this drives dead-taxel cache-col -> probe mapping.
    _taxel_channel_groups: int = 2

    def __init__(
        self,
        options: KinematicTaxelOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        # Grid eligibility for spatial crosstalk: requires a 2D probe layout with non-degenerate spacing. Strict
        # regularity (uniform normals, orthogonal tangents, exact rectangle) is reported separately as a warning.
        # Flat pos/normals are already populated by ProbeSensorMixin / ProbesWithNormalSensorMixin.
        is_grid = len(self._probe_layout_shape) == 2
        _, _, self._use_grid_crosstalk, is_grid_regular, grid_normal, grid_tangent_u, grid_tangent_v, grid_spacing = (
            normalize_grid_probe_layout(
                np.asarray(options.probe_local_pos, dtype=gs.np_float),
                np.asarray(options.probe_local_normal, dtype=gs.np_float),
                is_grid,
            )
        )
        self._grid_normal = torch.tensor(grid_normal, dtype=gs.tc_float, device=gs.device)
        self._grid_tangent_u = torch.tensor(grid_tangent_u, dtype=gs.tc_float, device=gs.device)
        self._grid_tangent_v = torch.tensor(grid_tangent_v, dtype=gs.tc_float, device=gs.device)
        self._grid_spacing = torch.tensor(grid_spacing, dtype=gs.tc_float, device=gs.device)

        if self._options.crosstalk_strength > 0.0:
            if not self._use_grid_crosstalk:
                gs.raise_exception(
                    "KinematicTaxel crosstalk requires a 2D grid-shaped probe_local_pos (shape (ny, nx, 3) with "
                    f"ny, nx >= 2 and non-degenerate spacing); got shape {tuple(self._probe_layout_shape)}."
                )
            if not is_grid_regular:
                gs.logger.warning(
                    "KinematicTaxel crosstalk grid is not strictly regular (uniform spacing, uniform normals, "
                    "orthogonal tangents); crosstalk will use averaged spacing and normal as a best-fit "
                    "approximation."
                )

    def build(self):
        super().build()

        self._shared_metadata.normal_stiffness = concat_with_tensor(
            self._shared_metadata.normal_stiffness, float(self._options.normal_stiffness), expand=(1,)
        )
        self._shared_metadata.normal_damping = concat_with_tensor(
            self._shared_metadata.normal_damping, float(self._options.normal_damping), expand=(1,)
        )
        self._shared_metadata.normal_exponent = concat_with_tensor(
            self._shared_metadata.normal_exponent, float(self._options.normal_exponent), expand=(1,)
        )
        self._shared_metadata.shear_scalar = concat_with_tensor(
            self._shared_metadata.shear_scalar, float(self._options.shear_scalar), expand=(1,)
        )
        self._shared_metadata.twist_scalar = concat_with_tensor(
            self._shared_metadata.twist_scalar, float(self._options.twist_scalar), expand=(1,)
        )

        if self._options.crosstalk_strength > 0.0 and self._use_grid_crosstalk:
            self._register_crosstalk()

        # Re-allocate the per-(env, sensor) contact prefilter buffers to absorb the newly-registered sensor.
        # Sized at build time; the per-step kernel writes into the same buffers without further allocation.
        B = self._manager._sim._B
        n_sensors_built = self._shared_metadata.n_probes_per_sensor.shape[0]
        self._shared_metadata.sensor_contacts_idx = torch.zeros(
            (B, n_sensors_built, _MAX_CONTACTS_PER_SENSOR), dtype=gs.tc_int, device=gs.device
        )
        self._shared_metadata.sensor_n_contacts = torch.zeros((B, n_sensors_built), dtype=gs.tc_int, device=gs.device)

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        shape = (*self._probe_layout_shape, 3)
        return shape, shape

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_context: None,
        shared_metadata: KinematicTaxelMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        measured, measured_cols_b = get_measured_bufs(
            shared_metadata, current_ground_truth_data_T, measured_data_timeline
        )
        # The measured branch is provably identical to GT (and the kernel can skip recomputing it) when no probe
        # has a noised sensing radius and no probe has a non-unit measured-branch gain.
        measured_equals_gt = int(
            not shared_metadata.has_any_probe_radius_noise and not shared_metadata.has_any_probe_gain
        )
        _kernel_build_sensor_contact_idx(
            shared_metadata.links_idx,
            solver.collider._collider_state,
            shared_metadata.sensor_contacts_idx,
            shared_metadata.sensor_n_contacts,
        )
        _kernel_kinematic_taxel(
            shared_metadata.probe_positions,
            shared_metadata.probe_sensor_idx,
            shared_metadata.probe_radii,
            shared_metadata.probe_radii_noise,
            shared_metadata.probe_gains,
            shared_metadata.normal_stiffness,
            shared_metadata.normal_damping,
            shared_metadata.normal_exponent,
            shared_metadata.shear_scalar,
            shared_metadata.twist_scalar,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            shared_metadata.sensor_contacts_idx,
            shared_metadata.sensor_n_contacts,
            solver.collider._collider_state,
            solver.collider._collider_static_config,
            solver.links_state,
            solver.geoms_state,
            solver.geoms_info,
            solver._rigid_global_info,
            solver.collider._sdf._sdf_info,
            gs.EPS,
            measured_equals_gt,
            current_ground_truth_data_T,
            measured_cols_b,
        )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext"):
        def mask(envs_idx):
            force = self.read_ground_truth(envs_idx).force
            if self._options.history_length > 0:
                force = force.select(1 if self._manager._sim.n_envs > 0 else 0, -1)
            return torch.linalg.norm(force, dim=-1) >= gs.EPS

        self._draw_debug_probes(context, self._tactile_color_groups_fn(mask))
