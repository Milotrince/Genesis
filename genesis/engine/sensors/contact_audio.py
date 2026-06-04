from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.engine.audio import PublishedSource
from genesis.options.sensors import ContactAudio as ContactAudioOptions
from genesis.options.sensors import ContactAudioProperties
from genesis.utils.geom import inv_transform_by_trans_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensor, SimpleSensorMetadata

if TYPE_CHECKING:
    from genesis.engine.solvers import RigidSolver
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager

# Fraction of the sliding-texture drive injected into the modal bank (vs. the direct broadband scrape resonator). The
# modal formants carry the material identity (a metal/glass scrape is bright because its high modes are excited), but
# the broadband resonator must dominate so the slide reads as noise, not a tone -- so keep this below 1.
_MODAL_NOISE_FRAC = 0.7


@dataclass
class ContactAudioSensorMetadata(RigidSensorMetadataMixin, SimpleSensorMetadata):
    """
    Shared metadata for all contact audio sensors.

    Holds the packed per-material vibroacoustic tables (resolved from ``properties_dict`` and keyed by the *struck*
    link) and the persistent synthesis state of the modal / texture resonators. ``n_modes`` and ``audio_substeps``
    are class-uniform (asserted at build) so every state tensor is rectangular over all sensors of the class.
    """

    n_modes: int = 0
    audio_substeps: int = 0
    properties_dict: dict[int, ContactAudioProperties] = field(default_factory=dict)

    # Active-acoustic excitation (Lu & Culbertson). has_excitation gates the per-substep drive work off when no sensor
    # in the class uses it. exc_kind codes: -1 none, 0 impulse, 1 linear_sweep, 2 exp_sweep.
    has_excitation: bool = False
    exc_kind: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    exc_flo: torch.Tensor = make_tensor_field((0,))
    exc_fhi: torch.Tensor = make_tensor_field((0,))
    exc_dur: torch.Tensor = make_tensor_field((0,))
    exc_amp: torch.Tensor = make_tensor_field((0,))
    exc_period: torch.Tensor = make_tensor_field((0,))
    exc_t: torch.Tensor = make_tensor_field((0, 0))  # time within the excitation period, per (B, n_sensors)
    exc_phase: torch.Tensor = make_tensor_field((0, 0))  # accumulated sweep phase, per (B, n_sensors)

    # Material tables, shape (n_materials, n_modes) for modal terms and (n_materials,) for scalar terms. Material 0 is
    # the default (key -1); link_to_material_idx maps a struck link index to a row (-1 = no material -> silent).
    link_to_material_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    mat_freqs: torch.Tensor = make_tensor_field((0, 0))
    mat_decays: torch.Tensor = make_tensor_field((0, 0))
    mat_gains: torch.Tensor = make_tensor_field((0, 0))
    mat_rough_gain: torch.Tensor = make_tensor_field((0,))
    mat_rough_sf: torch.Tensor = make_tensor_field((0,))
    mat_rough_bw: torch.Tensor = make_tensor_field((0,))
    mat_impact_gain: torch.Tensor = make_tensor_field((0,))
    mat_impact_thresh: torch.Tensor = make_tensor_field((0,))
    mat_contact_damping: torch.Tensor = make_tensor_field((0,))
    mat_damp_per_force: torch.Tensor = make_tensor_field((0,))
    # Acceleration-noise "click" resonator (Hertzian impact transient), one fast-decay high-freq mode per material.
    mat_accel_gain: torch.Tensor = make_tensor_field((0,))
    mat_accel_freq: torch.Tensor = make_tensor_field((0,))
    mat_accel_decay: torch.Tensor = make_tensor_field((0,))
    # Position-dependent (strike-location) timbre: per-material surface sample points (in the struck link's local
    # frame) and their per-mode normalized mode shapes. has_surface gates the per-step nearest-vertex lookup off when
    # no material provides them. Padded to max_surf over materials; mat_n_surf is the real count per material.
    has_surface: bool = False
    mat_n_surf: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    mat_surf_pts: torch.Tensor = make_tensor_field((0, 0, 3))
    mat_surf_shape: torch.Tensor = make_tensor_field((0, 0, 0))

    # Persistent synthesis state, shape (B, n_sensors, n_modes) for modal banks and (B, n_sensors) for texture/force.
    # cur_a1 / cur_a2 / cur_gains latch the active material's resonator coefficients so a mode keeps ringing down with
    # the last-touched material's decay after the contact is released.
    modal_y1: torch.Tensor = make_tensor_field((0, 0, 0))
    modal_y2: torch.Tensor = make_tensor_field((0, 0, 0))
    cur_a1: torch.Tensor = make_tensor_field((0, 0, 0))
    cur_a2: torch.Tensor = make_tensor_field((0, 0, 0))
    cur_gains: torch.Tensor = make_tensor_field((0, 0, 0))
    tex_y1: torch.Tensor = make_tensor_field((0, 0))
    tex_y2: torch.Tensor = make_tensor_field((0, 0))
    acc_y1: torch.Tensor = make_tensor_field((0, 0))
    acc_y2: torch.Tensor = make_tensor_field((0, 0))
    prev_force: torch.Tensor = make_tensor_field((0, 0))
    # Most recent synthesized block, shape (B, n_sensors, audio_substeps). Exposed so the airborne SpatialAudio
    # sensor can pull each contact mic's structure-borne output as a radiation source (read one step late).
    last_block: torch.Tensor = make_tensor_field((0, 0, 0))


class ContactAudioSensor(
    RigidSensorMixin[ContactAudioSensorMetadata],
    SimpleSensor[ContactAudioOptions, None, ContactAudioSensorMetadata],
):
    """
    Link-level contact vibration / audio sensor (source-filter modal synthesis driven by solver contacts).

    Each step reads the contact forces on the attached link and the relative velocity at the dominant contact, then
    emits a block of ``audio_substeps`` synthesized samples: a modal oscillator bank excited by the contact-force
    onset (impact ring-down) plus a velocity-scaled noise source driving a texture resonator (sliding roughness). The
    timbre is keyed by the material of the struck link.
    """

    def __init__(
        self,
        options: ContactAudioOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self.debug_object: "Mesh | None" = None

    def build(self):
        super().build()

        sm = self._shared_metadata
        solver: "RigidSolver" = sm.solver
        batch_size = self._manager._sim._B

        # n_modes / audio_substeps are class-uniform so the shared state tensors stay rectangular.
        if sm.n_modes == 0:
            sm.n_modes = self._options.n_modes
            sm.audio_substeps = self._options.audio_substeps
        elif sm.n_modes != self._options.n_modes or sm.audio_substeps != self._options.audio_substeps:
            gs.raise_exception(
                "All ContactAudio sensors must share the same n_modes and audio_substeps. "
                f"Got n_modes={self._options.n_modes}, audio_substeps={self._options.audio_substeps} vs existing "
                f"n_modes={sm.n_modes}, audio_substeps={sm.audio_substeps}."
            )
        n_modes = sm.n_modes

        # Material table, rebuilt (like TemperatureGrid) whenever the merged properties_dict grows.
        if sm.link_to_material_idx.shape[0] == 0:
            sm.link_to_material_idx = torch.full((solver.n_links,), -1, dtype=gs.tc_int, device=gs.device)
        sm.properties_dict.update(self._options.properties_dict)
        n_mat = len(sm.properties_dict)
        if n_mat > sm.mat_freqs.shape[0]:
            sm.mat_freqs = torch.zeros((n_mat, n_modes), dtype=gs.tc_float, device=gs.device)
            sm.mat_decays = torch.zeros((n_mat, n_modes), dtype=gs.tc_float, device=gs.device)
            sm.mat_gains = torch.zeros((n_mat, n_modes), dtype=gs.tc_float, device=gs.device)
            sm.mat_rough_gain = torch.zeros((n_mat,), dtype=gs.tc_float, device=gs.device)
            sm.mat_rough_sf = torch.zeros((n_mat,), dtype=gs.tc_float, device=gs.device)
            sm.mat_rough_bw = torch.ones((n_mat,), dtype=gs.tc_float, device=gs.device)
            sm.mat_impact_gain = torch.zeros((n_mat,), dtype=gs.tc_float, device=gs.device)
            sm.mat_impact_thresh = torch.zeros((n_mat,), dtype=gs.tc_float, device=gs.device)
            sm.mat_contact_damping = torch.zeros((n_mat,), dtype=gs.tc_float, device=gs.device)
            sm.mat_damp_per_force = torch.zeros((n_mat,), dtype=gs.tc_float, device=gs.device)
            sm.mat_accel_gain = torch.zeros((n_mat,), dtype=gs.tc_float, device=gs.device)
            sm.mat_accel_freq = torch.zeros((n_mat,), dtype=gs.tc_float, device=gs.device)
            sm.mat_accel_decay = torch.ones((n_mat,), dtype=gs.tc_float, device=gs.device)
            max_surf = max((len(p.surface_points) for p in sm.properties_dict.values()), default=0)
            sm.has_surface = max_surf > 0
            sm.mat_n_surf = torch.zeros((n_mat,), dtype=gs.tc_int, device=gs.device)
            sm.mat_surf_pts = torch.zeros((n_mat, max_surf, 3), dtype=gs.tc_float, device=gs.device)
            sm.mat_surf_shape = torch.zeros((n_mat, max_surf, n_modes), dtype=gs.tc_float, device=gs.device)
            # -1 in link_to_material_idx means "no material" (silent); 0 uses the default (key -1) properties.
            sm.link_to_material_idx[:] = 0 if -1 in sm.properties_dict else -1
            # Sort by link index so the default key -1 lands at row 0.
            for i, (link_idx, props) in enumerate(sorted(sm.properties_dict.items(), key=lambda x: x[0])):
                m = len(props.modal_freqs)
                sm.mat_freqs[i, :m] = torch.tensor(props.modal_freqs, dtype=gs.tc_float, device=gs.device)
                sm.mat_decays[i, :m] = torch.tensor(props.modal_decays, dtype=gs.tc_float, device=gs.device)
                sm.mat_gains[i, :m] = torch.tensor(props.modal_gains, dtype=gs.tc_float, device=gs.device)
                sm.mat_rough_gain[i] = props.roughness_gain
                sm.mat_rough_sf[i] = props.roughness_spatial_freq
                sm.mat_rough_bw[i] = max(props.roughness_bandwidth, gs.EPS)
                sm.mat_impact_gain[i] = props.impact_gain
                sm.mat_impact_thresh[i] = props.impact_threshold
                sm.mat_contact_damping[i] = props.contact_damping
                sm.mat_damp_per_force[i] = props.contact_damping_per_force
                sm.mat_accel_gain[i] = props.accel_noise_gain
                sm.mat_accel_freq[i] = props.accel_noise_freq
                sm.mat_accel_decay[i] = max(props.accel_noise_decay, gs.EPS)
                ns = len(props.surface_points)
                sm.mat_n_surf[i] = ns
                if ns > 0:
                    sm.mat_surf_pts[i, :ns] = torch.tensor(props.surface_points, dtype=gs.tc_float, device=gs.device)
                    shp = torch.tensor(props.surface_mode_shapes, dtype=gs.tc_float, device=gs.device)
                    sm.mat_surf_shape[i, :ns, : shp.shape[1]] = shp
                if link_idx >= 0:
                    sm.link_to_material_idx[link_idx] = i

            # Anti-aliasing / stability guard: silence any mode at or above the carrier band edge so a mode above the
            # block Nyquist cannot alias to a phantom low tone. nyquist = 0.5 * sample_rate = 0.5 * audio_substeps / dt.
            nyquist = 0.5 * sm.audio_substeps / solver._sim.dt
            sm.mat_gains[sm.mat_freqs >= 0.45 * nyquist] = 0.0
            sm.mat_accel_freq.clamp_(max=0.45 * nyquist)

        if self._link is not None and self._link.idx not in sm.properties_dict and -1 not in sm.properties_dict:
            gs.logger.warning(
                f"ContactAudio sensor on link {self._link.idx} has no default (-1) material in properties_dict; "
                "contacts with unlisted links will be silent."
            )

        # Grow the per-sensor synthesis state by one column (this sensor). concat_with_tensor returns the broadcast
        # `expand` view as-is on the first (empty) call, so `.contiguous()` materializes real storage that the
        # per-step in-place `copy_` write-back can target.
        for name in ("modal_y1", "modal_y2", "cur_a1", "cur_a2", "cur_gains"):
            grown = concat_with_tensor(getattr(sm, name), 0.0, expand=(batch_size, 1, n_modes), dim=1)
            setattr(sm, name, grown.contiguous())
        for name in ("tex_y1", "tex_y2", "acc_y1", "acc_y2", "prev_force"):
            grown = concat_with_tensor(getattr(sm, name), 0.0, expand=(batch_size, 1), dim=1)
            setattr(sm, name, grown.contiguous())
        grown = concat_with_tensor(sm.last_block, 0.0, expand=(batch_size, 1, self._options.audio_substeps), dim=1)
        sm.last_block = grown.contiguous()

        # Active-acoustic excitation parameters (per sensor) + phase/time state (per env, per sensor).
        exc = self._options.excitation
        kind_code = {"impulse": 0, "linear_sweep": 1, "exp_sweep": 2}.get(exc.kind, -1) if exc is not None else -1
        sm.exc_kind = concat_with_tensor(sm.exc_kind, kind_code)
        sm.exc_flo = concat_with_tensor(sm.exc_flo, exc.f_lo if exc is not None else 0.0)
        sm.exc_fhi = concat_with_tensor(sm.exc_fhi, exc.f_hi if exc is not None else 0.0)
        sm.exc_dur = concat_with_tensor(sm.exc_dur, max(exc.duration, gs.EPS) if exc is not None else 1.0)
        sm.exc_amp = concat_with_tensor(sm.exc_amp, exc.amplitude if exc is not None else 0.0)
        period = (exc.period if exc.period > 0.0 else exc.duration) if exc is not None else 1.0
        sm.exc_period = concat_with_tensor(sm.exc_period, max(period, gs.EPS))
        for name in ("exc_t", "exc_phase"):
            grown = concat_with_tensor(getattr(sm, name), 0.0, expand=(batch_size, 1), dim=1)
            setattr(sm, name, grown.contiguous())
        if exc is not None:
            sm.has_excitation = True

        # Publish this contact mic's structure-borne output as a radiation source so the airborne SpatialAudio mic can
        # render it through the AudioManager registry. One entry per class (covers all ContactAudio sensors via the
        # shared `last_block`); the callables read the latest tensors after build-time growth.
        if self._idx == 0:
            self._manager._sim._audio_manager.register_published(
                PublishedSource(lambda: sm.last_block, lambda: sm.links_idx, lambda: sm.offsets_pos)
            )

    def _get_return_format(self) -> tuple[int, ...]:
        # One block of `audio_substeps` synthesized samples per step.
        return (self._options.audio_substeps,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: ContactAudioSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        for name in (
            "modal_y1",
            "modal_y2",
            "cur_a1",
            "cur_a2",
            "cur_gains",
            "tex_y1",
            "tex_y2",
            "acc_y1",
            "acc_y2",
            "prev_force",
            "last_block",
            "exc_t",
            "exc_phase",
        ):
            getattr(shared_metadata, name)[envs_idx] = 0.0

    @classmethod
    def _update_raw_data(
        cls, shared_context: None, shared_metadata: ContactAudioSensorMetadata, raw_data_T: torch.Tensor
    ):
        sm = shared_metadata
        solver = sm.solver
        assert solver is not None

        B, n_sensors, n_modes = sm.modal_y1.shape
        K = sm.audio_substeps
        dt_sub = solver._sim.dt / K
        sensor_links = sm.links_idx.long()  # (n_sensors,)

        # --- 1) Slowly-varying contact drivers, aggregated per sensor link ---------------------------------------
        contacts = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        force, link_a, link_b = contacts["force"], contacts["link_a"], contacts["link_b"]
        normal, position = contacts["normal"], contacts["position"]
        if solver.n_envs == 0:
            force, link_a, link_b = force[None], link_a[None], link_b[None]
            normal, position = normal[None], position[None]

        zeros_bs = torch.zeros((B, n_sensors), dtype=gs.tc_float, device=gs.device)
        # Per-mode strike-location factor in [-1, 1] (1 at a mode's antinode); 1 everywhere when no surface data, so it
        # is a no-op for hand-tuned materials. Overwritten in the contact branch when the struck material has a surface
        # mode-shape table.
        loc = torch.ones((B, n_sensors, n_modes), dtype=gs.tc_float, device=gs.device)
        n_contacts = link_a.shape[-1]
        if n_contacts == 0:
            # No contacts this step: no excitation, but the modal banks keep ringing down with their latched
            # coefficients (cur_a1 / cur_a2 / cur_gains).
            f_normal = zeros_bs
            slip = zeros_bs
            valid = torch.zeros((B, n_sensors), dtype=torch.bool, device=gs.device)
            # Released: the modal bank rings down freely with its latched (contact-damping-free) coefficients.
            a1, a2, gains = sm.cur_a1, sm.cur_a2, sm.cur_gains
            impulse = torch.zeros((B, n_sensors, 1), dtype=gs.tc_float, device=gs.device)
            modal_drive = zeros_bs  # no surface excitation into the modes when not in contact
            tex_amp = zeros_bs
            tex_freq = zeros_bs
            bw = zeros_bs + 1.0  # texture is silent (tex_amp=0); any positive bandwidth keeps the resonator stable
            # Click resonator fully damped when released; its sub-millisecond burst has already decayed by now.
            aa1 = zeros_bs
            aa2 = zeros_bs
            accel_kick = zeros_bs
        else:
            la = link_a.unsqueeze(1)  # (B, 1, C)
            lb = link_b.unsqueeze(1)
            sl = sensor_links.view(1, -1, 1)  # (1, n_sensors, 1)
            is_a = la == sl  # (B, n_sensors, C)
            is_b = lb == sl
            involved = is_a | is_b
            any_contact = involved.any(dim=2)  # (B, n_sensors)

            # Total contact force on the sensor link (Newton's third law: +force when the sensor is link_b).
            orient = is_b.to(gs.tc_float) - is_a.to(gs.tc_float)  # (B, n_sensors, C)
            total_force = (orient.unsqueeze(-1) * force.unsqueeze(1)).sum(dim=2)  # (B, n_sensors, 3)
            f_normal = total_force.norm(dim=-1)  # (B, n_sensors)

            # Dominant contact per sensor (largest force magnitude among involved contacts).
            fmag = force.norm(dim=-1)  # (B, C)
            fmag_involved = torch.where(involved, fmag.unsqueeze(1), zeros_bs.new_zeros((B, n_sensors, n_contacts)))
            primary = fmag_involved.argmax(dim=2)  # (B, n_sensors), index into C

            other_all = torch.where(is_a, lb.expand(B, n_sensors, n_contacts), la.expand(B, n_sensors, n_contacts))
            struck = other_all.gather(2, primary.unsqueeze(-1)).squeeze(-1).long()  # (B, n_sensors)
            prim_n = (
                normal.unsqueeze(1)
                .expand(B, n_sensors, n_contacts, 3)
                .gather(2, primary.view(B, n_sensors, 1, 1).expand(B, n_sensors, 1, 3))
                .squeeze(2)
            )  # (B, n_sensors, 3)

            # Relative tangential velocity at the contact (link-origin velocities; ignores the angular lever arm,
            # adequate for a first-pass texture pitch).
            vel_all = solver.get_links_vel()
            if solver.n_envs == 0:
                vel_all = vel_all[None]  # (B, n_links, 3)
            v_sensor = vel_all[:, sensor_links, :]  # (B, n_sensors, 3)
            v_struck = vel_all.gather(1, struck.clamp(min=0).unsqueeze(-1).expand(B, n_sensors, 3))
            v_rel = v_sensor - v_struck
            v_tan = v_rel - (v_rel * prim_n).sum(dim=-1, keepdim=True) * prim_n
            slip = v_tan.norm(dim=-1) * any_contact.to(gs.tc_float)  # (B, n_sensors)

            # Material of the struck link.
            struck_safe = struck.clamp(min=0)
            mat_idx = sm.link_to_material_idx[struck_safe]  # (B, n_sensors)
            valid = any_contact & (mat_idx >= 0)
            validf = valid.to(gs.tc_float)
            mat_safe = mat_idx.clamp(min=0).long()

            # Strike-location modal weighting: map the dominant contact point into the struck link's local frame, find
            # the nearest precomputed surface sample, and read its per-mode mode shape. A mode struck at its node is
            # silent; at its antinode, maximal (van den Doel; Zheng & James). loc stays all-ones for materials with no
            # surface table, recovering the flat-gain behavior.
            if sm.has_surface:
                prim_pos = (
                    position.unsqueeze(1)
                    .expand(B, n_sensors, n_contacts, 3)
                    .gather(2, primary.view(B, n_sensors, 1, 1).expand(B, n_sensors, 1, 3))
                    .squeeze(2)
                )  # (B, n_sensors, 3) world-frame contact point
                all_pos = solver.get_links_pos()
                all_quat = solver.get_links_quat()
                if solver.n_envs == 0:
                    all_pos, all_quat = all_pos[None], all_quat[None]
                struck_pos = all_pos.gather(1, struck_safe.unsqueeze(-1).expand(B, n_sensors, 3))
                struck_quat = all_quat.gather(1, struck_safe.unsqueeze(-1).expand(B, n_sensors, 4))
                local_c = inv_transform_by_trans_quat(prim_pos, struck_pos, struck_quat)  # (B, n_sensors, 3)

                max_surf = sm.mat_surf_pts.shape[1]
                pts = sm.mat_surf_pts[mat_safe]  # (B, n_sensors, max_surf, 3)
                nsurf = sm.mat_n_surf[mat_safe]  # (B, n_sensors)
                d2 = (pts - local_c.unsqueeze(2)).square().sum(dim=-1)  # (B, n_sensors, max_surf)
                pad = torch.arange(max_surf, device=gs.device).view(1, 1, -1) >= nsurf.unsqueeze(-1)
                d2 = d2.masked_fill(pad, float("inf"))
                nearest = d2.argmin(dim=2)  # (B, n_sensors)
                shp = sm.mat_surf_shape[mat_safe]  # (B, n_sensors, max_surf, n_modes)
                pos_shape = shp.gather(2, nearest.view(B, n_sensors, 1, 1).expand(B, n_sensors, 1, n_modes)).squeeze(2)
                has_surf = ((nsurf > 0) & valid).unsqueeze(-1)  # only where the material actually has a table
                loc = torch.where(has_surf, pos_shape, loc)

            freqs = sm.mat_freqs[mat_safe]  # (B, n_sensors, n_modes)
            decays = sm.mat_decays[mat_safe]
            new_gains = sm.mat_gains[mat_safe] * loc  # readout weight modulated by strike location
            theta = 2.0 * torch.pi * freqs * dt_sub

            # Two coefficient sets per mode: the FREE decay (modal_decays only) used for the post-release ring-down,
            # and the in-contact decay (modal_decays + contact_damping) used while the finger is pressing. A finger
            # mass-loads and damps the object's modes, so during a slide they must not ring freely into a tone; the
            # long free ring-down only appears after release.
            free_r = torch.exp(-decays * dt_sub)
            free_a1 = 2.0 * free_r * torch.cos(theta)
            free_a2 = free_r * free_r
            # Force-coupled contact damping (Zheng & James 2011, per-mode form): a constant floor plus a term
            # proportional to the normal contact force AND to the squared mode shape at the contact, so damping a mode
            # at its antinode kills the ring while a contact at its node barely touches it (the coffee-mug effect).
            damp_force = (sm.mat_damp_per_force[mat_safe] * f_normal).unsqueeze(-1) * loc.square()
            cont_decays = decays + sm.mat_contact_damping[mat_safe].unsqueeze(-1) + damp_force
            cont_r = torch.exp(-cont_decays * dt_sub)
            cont_a1 = 2.0 * cont_r * torch.cos(theta)
            cont_a2 = cont_r * cont_r

            # Coefficients used for THIS step's synthesis: contact-damped where in contact, latched-free where
            # released. The latch (`cur_*`) stores the FREE coefficients so a released mode rings down naturally.
            vmask = valid.unsqueeze(-1)
            a1 = torch.where(vmask, cont_a1, sm.cur_a1)
            a2 = torch.where(vmask, cont_a2, sm.cur_a2)
            gains = torch.where(vmask, new_gains, sm.cur_gains)
            sm.cur_a1.copy_(torch.where(vmask, free_a1, sm.cur_a1))
            sm.cur_a2.copy_(torch.where(vmask, free_a2, sm.cur_a2))
            sm.cur_gains.copy_(gains)

            # Impact excitation, transient-gated: only a sharp force onset above the material threshold counts as a
            # tap and pings the modes. Steady-sliding force ripple stays below threshold, so sliding no longer
            # re-excites the modes into a sustained tone.
            d_force = (f_normal - sm.prev_force).clamp(min=0.0)
            is_tap = (d_force > sm.mat_impact_thresh[mat_safe]).to(gs.tc_float)
            # Excite each mode in proportion to its mode shape at the strike point (loc), so a tap on a node does not
            # ring that mode (loc=1 reproduces the uniform-impulse flat model).
            impulse = (sm.mat_impact_gain[mat_safe] * d_force * is_tap * validf).unsqueeze(
                -1
            ) * loc  # (B, n_s, n_modes)

            # Acceleration-noise "click": a sharp, fast-decaying high-frequency burst kicked by the same tap, modeling
            # the Hertzian contact transient that the slow modal ring-down misses (Wang et al. 2018 / Chadwick et al.).
            # The kick uses the raw gated force jump (independent of impact_gain) scaled by the material's accel gain.
            aa_r = torch.exp(-sm.mat_accel_decay[mat_safe] * dt_sub)
            aa1 = 2.0 * aa_r * torch.cos(2.0 * torch.pi * sm.mat_accel_freq[mat_safe] * dt_sub)
            aa2 = aa_r * aa_r
            accel_kick = sm.mat_accel_gain[mat_safe] * d_force * is_tap * validf  # (B, n_sensors)

            # Sliding texture: a velocity- and force-scaled broadband noise source. `tex_amp` drives the direct
            # scrape (a wide band-pass at the slip pitch) and `modal_drive` injects the same surface noise into the
            # (now contact-damped) modal bank so the object's resonances appear as formants coloring the scrape,
            # rather than as a pure tone.
            tex_amp = sm.mat_rough_gain[mat_safe] * f_normal * slip * validf
            modal_drive = _MODAL_NOISE_FRAC * tex_amp
            tex_freq = sm.mat_rough_sf[mat_safe] * slip
            bw = sm.mat_rough_bw[mat_safe]  # (B, n_sensors)

        # --- 2) High-rate block synthesis (carrier above the physics Nyquist) ------------------------------------
        # Texture resonator: a 2-pole resonator tuned to the slip-dependent pitch, damped by the material bandwidth.
        # Clamp the slip pitch below the carrier band edge so a fast slide cannot push the resonator past Nyquist
        # (sample_rate = 1/dt_sub, so the band edge is 0.45 * 0.5 / dt_sub).
        tex_freq = tex_freq.clamp(max=0.45 * 0.5 / dt_sub)
        rt = torch.exp(-torch.pi * bw * dt_sub)
        at1 = 2.0 * rt * torch.cos(2.0 * torch.pi * tex_freq * dt_sub)
        at2 = rt * rt

        # Clone the persistent state into loop-local working copies so the in-place `copy_` write-back below cannot
        # alias the storage being read (the rolling swap would otherwise share storage for tiny K).
        my1, my2 = sm.modal_y1.clone(), sm.modal_y2.clone()
        ty1, ty2 = sm.tex_y1.clone(), sm.tex_y2.clone()
        ay1, ay2 = sm.acc_y1.clone(), sm.acc_y2.clone()
        # Independent noise streams: `tex_noise` is the direct scrape source, `modal_noise` colors the modal bank.
        tex_noise = torch.randn((B, n_sensors, K), dtype=gs.tc_float, device=gs.device)
        modal_noise = torch.randn((B, n_sensors, n_modes, K), dtype=gs.tc_float, device=gs.device)
        md = modal_drive.unsqueeze(-1) * loc  # (B, n_sensors, n_modes), surface noise weighted by strike location

        # Active-acoustic emitter drive (Lu & Culbertson): while the sensor link is in contact, inject the excitation
        # signal into the (contact-damped) modal bank weighted by each mode's coupling, so the synthesized output is
        # the swept modal response. Gated off entirely when no sensor in the class uses excitation.
        if sm.has_excitation:
            et, eph = sm.exc_t.clone(), sm.exc_phase.clone()
            gate = valid.to(gs.tc_float)  # (B, n_sensors): emitter couples only while grasping
            ekind = sm.exc_kind.view(1, -1)
            is_imp, is_exp = ekind == 0, ekind == 2
            active_f = (ekind >= 0).to(gs.tc_float)
            eflo, efhi = sm.exc_flo.view(1, -1), sm.exc_fhi.view(1, -1)
            edur, eamp, eperiod = sm.exc_dur.view(1, -1), sm.exc_amp.view(1, -1), sm.exc_period.view(1, -1)
            eratio = (efhi / eflo.clamp(min=gs.EPS)).clamp(min=gs.EPS)

        out = torch.empty((B, n_sensors, K), dtype=gs.tc_float, device=gs.device)
        for k in range(K):
            # Modal bank: continuous surface-noise excitation (formants while sliding) + transient impulse on a tap.
            y = a1 * my1 - a2 * my2 + md * modal_noise[:, :, :, k]
            if k == 0:
                y = y + impulse
            if sm.has_excitation:
                frac = (et / edur).clamp(0.0, 1.0)
                f_inst = torch.where(is_exp, eflo * torch.pow(eratio, frac), eflo + (efhi - eflo) * frac)
                eph = eph + 2.0 * torch.pi * f_inst * dt_sub
                drive = torch.where(is_imp, torch.where(et < dt_sub, eamp, eamp * 0.0), eamp * torch.sin(eph))
                drive = drive * active_f * gate  # (B, n_sensors)
                y = y + gains * drive.unsqueeze(-1)
                et = (et + dt_sub) % eperiod
            my2, my1 = my1, y
            modal_sample = (gains * y).sum(dim=-1)  # (B, n_sensors)

            ty = at1 * ty1 - at2 * ty2 + tex_amp * tex_noise[:, :, k]
            ty2, ty1 = ty1, ty

            # Acceleration-noise click resonator: kicked once at block start, rings down within the block.
            ay = aa1 * ay1 - aa2 * ay2
            if k == 0:
                ay = ay + accel_kick
            ay2, ay1 = ay1, ay

            out[:, :, k] = modal_sample + ty + ay

        if sm.has_excitation:
            sm.exc_t.copy_(et)
            sm.exc_phase.copy_(eph)
        sm.acc_y1.copy_(ay1)
        sm.acc_y2.copy_(ay2)
        sm.modal_y1.copy_(my1)
        sm.modal_y2.copy_(my2)
        sm.tex_y1.copy_(ty1)
        sm.tex_y2.copy_(ty2)
        sm.prev_force.copy_(f_normal)
        # Expose the raw synthesized block (pre hardware-imperfection) as a radiation source for SpatialAudio.
        sm.last_block.copy_(out)

        # Cache layout is (n_sensors * K, B): sensor 0's K samples, then sensor 1's, ...
        raw_data_T[:] = out.permute(1, 2, 0).reshape(n_sensors * K, B)

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw a sphere at the link whose radius grows with the loudness of the most recent synthesized block.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        pos = tensor_to_array(self._link.get_pos(env_idx).reshape((3,)))
        block = self.read(env_idx).reshape(-1)
        amp = float(block.abs().max())

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None
        radius = 0.01 + min(amp, 1.0) * 0.05
        self.debug_object = context.draw_debug_sphere(pos=pos, radius=radius, color=(1.0, 0.3, 0.0, 0.6))
