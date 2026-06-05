from typing import TYPE_CHECKING, Annotated, Any, Generic, NamedTuple, Sequence, TypeVar

import numpy as np
from pydantic import BeforeValidator, Field, StrictBool, StrictInt, field_validator

import genesis as gs
from genesis.typing import (
    FArrayType,
    Grid3DFloatType,
    IArrayType,
    LaxVec3FType,
    NonNegativeFloat,
    NonNegativeInt,
    OptionalIArrayType,
    PositiveFArrayType,
    PositiveFGridType,
    PositiveFloat,
    PositiveInt,
    PositiveVec3IType,
    RotationMatrixType,
    UnitInterval,
    UnitIntervalVec3Type,
    UnitIntervalVec4Type,
    UnitVec3FArrayType,
    UnitVec3FGridType,
    UnitVec3FType,
    Vec2FType,
    Vec3FArrayType,
    Vec3FGridType,
    Vec3FType,
    Vec4FType,
    is_sequence,
)

from ..options import Options
from .raycaster import DepthCameraPattern, RaycastPattern

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.sensors.base_sensor import Sensor
    from genesis.engine.sensors.contact_audio import ContactAudioSensor
    from genesis.engine.sensors.contact_force import ContactForceSensor, ContactSensor
    from genesis.engine.sensors.imu import IMUSensor
    from genesis.engine.sensors.raycaster import RaycasterSensor
    from genesis.engine.sensors.spatial_audio import SpatialAudioSensor
    from genesis.engine.sensors.surface_distance_probe import SurfaceDistanceProbeSensor

    NonNegativeUnboundedFloat = float
    LaxNonNegativeUnboundedVec3FType = Vec3FType | float
else:
    NonNegativeUnboundedFloat = Annotated[float, Field(ge=0, strict=False)]
    LaxNonNegativeUnboundedVec3FType = Annotated[
        tuple[NonNegativeUnboundedFloat, NonNegativeUnboundedFloat, NonNegativeUnboundedFloat],
        BeforeValidator(lambda v: v if is_sequence(v) else (v,) * 3),
        Field(strict=False),
    ]
CrossCouplingAxisType = RotationMatrixType | UnitIntervalVec3Type | float


SensorT = TypeVar("SensorT", bound="Sensor")


def _check_len_match(value, expected_len: int, name: str, ref_name: str):
    if isinstance(value, Sequence) and len(value) != expected_len:
        gs.raise_exception(
            f"{name} must have the same length as {ref_name} when {name} is array-like. "
            f"Got {len(value)} {name} and {expected_len} {ref_name}."
        )


class SensorOptions(Options, Generic[SensorT]):
    """
    Base class for all sensor options.

    Each sensor should have their own options class that inherits from this class.
    The associated sensor class registers itself via ``Sensor.__init_subclass__`` when parameterized
    with this options class, e.g. ``class MySensor(Sensor[MyOptions, MyMetadata, MyData]): ...``

    Parameters
    ----------
    history_length : NonNegativeInt
        The length of the history to store. Defaults to 0 (no history).
    delay : float, optional
        The read delay time in seconds. Data read will be outdated by this amount. Defaults to 0.0 (no delay).
    jitter : float, optional
        The jitter in seconds modeled as a random additive delay sampled uniformly in ``[0, jitter)`` each step.
        Jitter cannot be greater than delay.
    draw_debug : bool
        If True and visualizer is active, the sensor will draw debug shapes in the scene. Defaults to False.
    """

    history_length: NonNegativeInt = 0
    delay: NonNegativeFloat = 0.0
    jitter: NonNegativeFloat = 0.0
    draw_debug: StrictBool = False
    # -1 means not link-attached. None is accepted from users and normalized to -1 so SensorManager can sort uniformly.
    entity_idx: StrictInt = Field(default=-1, ge=-1)

    @field_validator("entity_idx", mode="before")
    @classmethod
    def _normalize_entity_idx(cls, value):
        return -1 if value is None else value

    def model_post_init(self, context: Any) -> None:
        if self.jitter > self.delay:
            gs.raise_exception(f"{type(self).__name__}: Jitter must be less than or equal to read delay.")

    def validate_scene(self, scene: "Scene"):
        """
        Validate the sensor options values before the sensor is added to the scene.

        Use pydantic's model_post_init() for validation that does not require scene context.
        """
        assert scene.sim is not None
        if self.delay > 0:
            delay_hz = self.delay / scene.sim.dt
            if not np.isclose(delay_hz, round(delay_hz), atol=gs.EPS):
                gs.logger.warning(
                    f"{type(self).__name__}: Read delay should be a multiple of the simulation time step. Got "
                    f"{self.delay} and {scene.sim.dt}. Actual read delay will be {1 / round(delay_hz)}."
                )


class KinematicSensorOptionsMixin(SensorOptions[SensorT]):
    """
    Base options class for sensors attached to a KinematicEntity (or any subclass, including RigidEntity). Use this
    base for sensors whose output is purely kinematic and does not depend on physics-derived quantities like contact
    forces or inertial dynamics.

    Parameters
    ----------
    entity_idx : int
        The global entity index of the entity to which this sensor is attached. -1 or None for static sensors.
    link_idx_local : int, optional
        The local index of the link of the entity to which this sensor is attached.
    pos_offset : array-like[float, float, float], optional
        The positional offset of the sensor from the link.
    euler_offset : array-like[float, float, float], optional
        The rotational offset of the sensor from the link in degrees.
    """

    link_idx_local: NonNegativeInt = 0
    pos_offset: Vec3FType = (0.0, 0.0, 0.0)
    euler_offset: Vec3FType = (0.0, 0.0, 0.0)

    def validate_scene(self, scene: "Scene"):
        from genesis.engine.entities import KinematicEntity

        super().validate_scene(scene)
        if self.entity_idx >= 0:
            if self.entity_idx >= len(scene.entities):
                gs.raise_exception(f"Invalid entity index {self.entity_idx}.")
            entity = scene.entities[self.entity_idx]
            if not isinstance(entity, KinematicEntity):
                gs.raise_exception(f"Entity at index {self.entity_idx} is not a KinematicEntity.")
            if self.link_idx_local >= entity.n_links:
                gs.raise_exception(f"Invalid link index {self.link_idx_local} for entity {self.entity_idx}.")


class RigidSensorOptionsMixin(KinematicSensorOptionsMixin[SensorT]):
    """
    Options for sensors that require a RigidEntity specifically (e.g. contact, contact force, IMU, tactile).

    Any sensor whose output depends on physics quantities (contact pairs, friction, inertial dynamics) belongs
    here.
    """

    def validate_scene(self, scene: "Scene"):
        from genesis.engine.entities import RigidEntity

        super().validate_scene(scene)
        if self.entity_idx >= 0:
            entity = scene.entities[self.entity_idx]
            if not isinstance(entity, RigidEntity):
                gs.raise_exception(f"Entity at index {self.entity_idx} is not a RigidEntity.")


class SimpleSensorOptions(SensorOptions[SensorT]):
    """
    Options carrying SimpleSensor's imperfection parameters.

    Interpreted by ``_apply_hardware_imperfections`` as perturbations introduced by the embedded sampler when it
    snapshots the sensor into shared memory. Inherited by every ``SimpleSensor``-derived options class; Camera
    (deriving from ``Sensor`` directly) stays on plain ``SensorOptions``.

    Parameters
    ----------
    resolution : float | array-like[float, ...], optional
        The measurement resolution of the sensor (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    bias : float | array-like[float, ...], optional
        The constant additive bias of the sensor.
    noise : float | array-like[float, ...], optional
        The standard deviation of the additive white noise.
    random_walk : float | array-like[float, ...], optional
        The standard deviation of the random walk, which acts as accumulated bias drift.
    """

    resolution: FArrayType | float = 0.0
    bias: FArrayType | float = 0.0
    noise: FArrayType | float = 0.0
    random_walk: FArrayType | float = 0.0


class ProbeSensorOptionsMixin(SensorOptions[SensorT]):
    """
    Base options class for sensors that use local probe points.

    Parameters
    ----------
    probe_local_pos : array-like[array-like[float, float, float]] or shape ``(M, N, 3)`` grid
        Probe positions in link-local frame. Either a flat ``(N, 3)`` set or a 2D grid ``(M, N, 3)``; the
        ``read()`` output is reshaped back to match this layout.
    probe_radius : float | array-like[float] or shape ``(M, N)`` grid
        Probe sensing radius in meters. A scalar is shared by every probe; an array (or grid) must match the
        layout of ``probe_local_pos``.
    probe_radius_noise : float
        Additive radius noise in meters used by kernels whose measured branch depends on effective probe radius.
    debug_probe_color : array-like[float, float, float]
        RGB color for debug probe spheres (no alpha; the center sphere is drawn opaque and the outer sphere uses
        ``debug_probe_sphere_opacity``).
    debug_probe_center_radius : float
        Radius in meters of the small opaque marker sphere drawn at each probe position.
    debug_probe_sphere_opacity : float
        Alpha (0..1) of the outer translucent sphere drawn at each probe's sensing radius. Set to ``0.0`` to skip.
    """

    probe_local_pos: Vec3FArrayType | Vec3FGridType = ((0.0, 0.0, 0.0),)
    probe_radius: PositiveFloat | PositiveFArrayType | PositiveFGridType = 0.01
    probe_radius_noise: NonNegativeFloat = 0.0
    debug_probe_color: UnitIntervalVec3Type = (0.2, 0.4, 1.0)
    debug_probe_center_radius: PositiveFloat = 0.0008
    debug_probe_sphere_opacity: UnitInterval = 0.3

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        n_probes = int(np.prod(np.asarray(self.probe_local_pos).shape[:-1]))
        if isinstance(self.probe_radius, Sequence):
            if np.asarray(self.probe_radius).size != n_probes:
                gs.raise_exception(
                    f"probe_radius shape {np.asarray(self.probe_radius).shape} must contain "
                    f"{n_probes} entries to match probe_local_pos."
                )


class ProbesWithNormalSensorOptionsMixin(ProbeSensorOptionsMixin[SensorT]):
    """
    Probe options for sensors that also define one normal per probe, or one shared normal.
    """

    probe_local_normal: UnitVec3FType | UnitVec3FArrayType | UnitVec3FGridType = (0.0, 0.0, 1.0)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        n_probes = int(np.prod(np.asarray(self.probe_local_pos).shape[:-1]))
        normals = np.asarray(self.probe_local_normal)
        if normals.ndim > 1 and normals.size // 3 != n_probes:
            gs.raise_exception(
                "probe_local_normal must be one normal or contain one normal per probe. "
                f"Got normal shape {normals.shape} for {n_probes} probes."
            )


class Contact(RigidSensorOptionsMixin["ContactSensor"], SimpleSensorOptions["ContactSensor"]):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.

    Parameters
    ----------
    filter_link_idx : array-like[int], optional
        Global rigid link indices (solver link space). Contacts with the sensor link where the other
        participant is one of these links are ignored. Default is empty (no filtering).
    threshold : float, optional
        The bool-conversion threshold applied at read time to the underlying float contact magnitude
        (kernel produces float). A bin reads ``True`` iff its magnitude exceeds this value. Default
        ``0.0`` so any positive magnitude registers as contact.
    debug_sphere_radius : float, optional
        The radius of the debug sphere. Defaults to 0.05.
    debug_color : array-like[float, float, float, float], optional
        The rgba color of the debug sphere. Defaults to (1.0, 0.0, 1.0, 0.5).
    """

    filter_link_idx: OptionalIArrayType = Field(default_factory=tuple)
    threshold: NonNegativeFloat = 0.0
    debug_sphere_radius: PositiveFloat = 0.05
    debug_color: UnitIntervalVec4Type = (1.0, 0.0, 1.0, 0.5)

    def validate_scene(self, scene: "Scene"):
        super().validate_scene(scene)
        if self.filter_link_idx:
            n_links = scene.sim.rigid_solver.n_links
            if np.any(np.array(self.filter_link_idx) < 0) or np.any(np.array(self.filter_link_idx) >= n_links):
                gs.raise_exception(
                    f"Contact sensor filter_link_idx should be in range [0, {n_links}). Got {self.filter_link_idx}"
                )


class ContactForce(RigidSensorOptionsMixin["ContactForceSensor"], SimpleSensorOptions["ContactForceSensor"]):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.

    Parameters
    ----------
    min_force : float | array-like[float, float, float], optional
        The minimum detectable absolute force per each axis. Values below this will be treated as 0. Default is 0.
    max_force : float | array-like[float, float, float], optional
        The maximum output absolute force per each axis. Values above this will be clipped. Default is infinity.
    debug_color : array-like[float, float, float, float], optional
        The rgba color of the debug arrow. Defaults to (1.0, 0.0, 1.0, 0.5).
    debug_scale : float, optional
        The scale factor for the debug force arrow. Defaults to 0.01.
    """

    resolution: LaxVec3FType = 0.0

    min_force: LaxNonNegativeUnboundedVec3FType = 0.0
    max_force: LaxNonNegativeUnboundedVec3FType = np.inf

    debug_color: UnitIntervalVec4Type = (1.0, 0.0, 1.0, 0.5)
    debug_scale: PositiveFloat = 0.01

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if np.any(np.array(self.max_force) <= np.array(self.min_force)):
            gs.raise_exception(f"min_force should be less than max_force, got: {self.min_force} and {self.max_force}")


class ContactAudioProperties(NamedTuple):
    """
    Vibroacoustic material descriptor for an object that is *being contacted*, used by the ``ContactAudio`` sensor to
    synthesize contact vibration. Modal terms are the object's resonant signature (impact ring-down); roughness terms
    are the sliding-texture source.

    The source-filter split: ``ContactAudio`` excites a bank of damped modal oscillators (the filter) with the
    solver contact-force onset (impact) and a velocity-scaled noise source (sliding texture). Metal-like materials use
    high frequencies with slow decay (high-Q ringing); wood-like materials use lower frequencies with fast decay plus
    a stronger roughness source.

    Parameters
    ----------
    modal_freqs : tuple[float, ...]
        Resonant mode frequencies in Hz. One entry per mode; shorter than the sensor's ``n_modes`` is zero-padded.
    modal_decays : tuple[float, ...]
        Per-mode amplitude decay rate in 1/s (large = fast decay = wood-like; small = long ring = metal-like). Must
        match the length of ``modal_freqs``.
    modal_gains : tuple[float, ...]
        Per-mode output weight. Must match the length of ``modal_freqs``.
    roughness_gain : float
        Amplitude of the sliding-texture noise source. Scales with normal force and slip speed at runtime. ``0``
        disables texture (smooth surface). Default ``0``.
    roughness_spatial_freq : float
        Surface roughness spatial frequency in bumps per meter. The texture's temporal pitch is
        ``roughness_spatial_freq * slip_speed`` (Hz), so faster sliding raises the pitch. Default ``0``.
    roughness_bandwidth : float
        Spectral width (Hz) of the sliding-texture noise source. This is the bandwidth of a broadband band-pass
        centered on the slip-dependent pitch; larger values give a noisier "shhh/scrrr" scrape, smaller values an
        unrealistic tone. Use several hundred Hz for a realistic scrape. Default ``600``.
    impact_gain : float
        Scale applied to the (transient-gated) contact-force onset impulse that excites the modal bank on a tap.
        Default ``1``.
    impact_threshold : float
        Minimum positive force jump (Newtons, per physics step) that counts as a tap and injects a modal impulse.
        Steady-sliding force ripple stays below this, so sliding does not re-ping the modes into a sustained tone;
        only a sharp onset (an actual strike) excites the ring-down. Default ``0.5``.
    contact_damping : float
        Extra modal decay (1/s) added to ``modal_decays`` *while the surface is in contact*. A finger pressing on an
        object mass-loads and damps its modes, so they should not ring freely during a slide; the long free ring-down
        appears only after release (when this term is removed). Larger values give a more deadened in-contact sound.
        This is the force-independent floor; see ``contact_damping_per_force`` for the force-coupled term. Default
        ``80``.
    contact_damping_per_force : float
        Force-coupled in-contact modal decay in ``1/(s·N)``: the extra decay while in contact is
        ``contact_damping + contact_damping_per_force * f_normal``. This is the cheap per-mode form of the
        contact-dependent viscous damping of Zheng & James 2011 (damping proportional to contact force), which
        reproduces the coffee-mug effect (a firmer press deadens the ring more than a light touch). Default ``0``
        (back-compatible: in-contact damping is the constant ``contact_damping``).
    accel_noise_gain : float
        Amplitude of the acceleration-noise "click" injected on a tap (a sharp, fast-decaying broadband burst that
        models the Hertzian contact transient of small hard objects, which the slow modal ring-down misses; cf. the
        acceleration-noise shader of Wang et al. 2018 / Chadwick et al.). Scales the impact impulse into a dedicated
        high-frequency, fast-decay resonator. ``0`` disables the click. Default ``0``.
    accel_noise_freq : float
        Center frequency (Hz) of the acceleration-noise click resonator. Default ``5000``.
    accel_noise_decay : float
        Decay rate (1/s) of the acceleration-noise click resonator. Large = a short snappy click. Default ``800``.
    surface_points : tuple[tuple[float, float, float], ...]
        Surface sample positions in the struck object's *link-local* frame (meters), shape ``(n_surface, 3)``. With
        ``surface_mode_shapes`` these make the timbre depend on *where* the object is struck (a mode is silent at its
        node, loud at its antinode; van den Doel, Zheng & James). Populated by ``from_mesh``; empty disables
        position dependence (flat ``modal_gains`` are used everywhere). Default ``()``.
    surface_mode_shapes : tuple[tuple[float, ...], ...]
        Per-surface-point, per-mode normalized mode-shape amplitude in ``[-1, 1]`` (1 at each mode's antinode), shape
        ``(n_surface, n_modes)`` aligned with ``surface_points`` / ``modal_freqs``. Default ``()``.
    """

    modal_freqs: tuple[float, ...] = (250.0,)
    modal_decays: tuple[float, ...] = (40.0,)
    modal_gains: tuple[float, ...] = (1.0,)
    roughness_gain: float = 0.0
    roughness_spatial_freq: float = 0.0
    roughness_bandwidth: float = 600.0
    impact_gain: float = 1.0
    impact_threshold: float = 0.5
    contact_damping: float = 80.0
    contact_damping_per_force: float = 0.0
    accel_noise_gain: float = 0.0
    accel_noise_freq: float = 5000.0
    accel_noise_decay: float = 800.0
    surface_points: tuple = ()
    surface_mode_shapes: tuple = ()

    @classmethod
    def from_mesh(cls, verts, elems, material, n_modes: int = 8, sample_rate: float | None = None, **overrides):
        """
        Build physically-derived modal properties from a tetrahedral mesh and an isotropic material via linear modal
        analysis (see ``genesis.utils.modal_analysis``), instead of hand-tuning ``modal_freqs/decays/gains``.

        Parameters
        ----------
        verts : array-like, shape (N, 3)
            Tetrahedral mesh vertices in meters (e.g. from ``genesis.utils.modal_analysis.tetrahedralize``).
        elems : array-like, shape (T, 4)
            Tetrahedra (vertex indices).
        material : str | genesis.utils.modal_analysis.Material
            A key of ``MATERIAL_PRESETS`` (e.g. ``"steel"``) or an explicit ``Material``.
        n_modes : int
            Number of modes to extract (rigid-body modes are skipped). Default ``8``.
        sample_rate : float, optional
            If given, modes above the carrier band edge are dropped (anti-aliasing).
        overrides
            Any remaining ``ContactAudioProperties`` fields to set (e.g. ``roughness_gain``, ``impact_gain``).
        """
        from genesis.utils.modal_analysis import MATERIAL_PRESETS, compute_modal_model

        if isinstance(material, str):
            material = MATERIAL_PRESETS[material]
        model = compute_modal_model(verts, elems, material, n_modes, sample_rate)
        # Normalize each mode's raw surface amplitude to a [-1, 1] shape (1 at its antinode) so position weighting
        # modulates the flat modal_gains rather than rescaling overall loudness.
        sg = np.asarray(model.surface_gains, dtype=np.float64)
        shapes = sg / np.maximum(np.abs(sg).max(axis=0, keepdims=True), 1e-12)
        props = cls(
            modal_freqs=tuple(float(f) for f in model.freqs),
            modal_decays=tuple(float(d) for d in model.decays),
            modal_gains=tuple(float(g) for g in model.gains),
            contact_damping_per_force=float(material.contact_damping_per_force),
            surface_points=tuple(tuple(float(c) for c in p) for p in model.surface_points.tolist()),
            surface_mode_shapes=tuple(tuple(float(s) for s in row) for row in shapes.tolist()),
        )
        return props._replace(**overrides) if overrides else props


class ExcitationSignal(NamedTuple):
    """
    Active-acoustic excitation injected by an emitter into the grasped object's modal bank (Lu & Culbertson 2023). The
    receiver ``ContactAudio`` sensor records the modal response, whose spectrum encodes the object's resonances and how
    contact formations damp them.

    Parameters
    ----------
    kind : str
        ``"impulse"`` (a click each period), ``"linear_sweep"`` (frequency rises linearly), or ``"exp_sweep"``
        (frequency rises exponentially, emphasizing low frequencies). Default ``"linear_sweep"``.
    f_lo : float
        Sweep start frequency in Hz. Default ``20``.
    f_hi : float
        Sweep end frequency in Hz. Default ``10000``.
    duration : float
        Sweep duration in seconds (one pass low->high). Default ``0.5``.
    amplitude : float
        Drive amplitude. Default ``1``.
    period : float
        Repeat interval in seconds; the excitation restarts every ``period`` (``<= 0`` uses ``duration``, i.e. it loops
        back-to-back). Default ``0`` (loop).
    """

    kind: str = "linear_sweep"
    f_lo: float = 20.0
    f_hi: float = 10000.0
    duration: float = 0.5
    amplitude: float = 1.0
    period: float = 0.0


class ContactAudio(RigidSensorOptionsMixin["ContactAudioSensor"], SimpleSensorOptions["ContactAudioSensor"]):
    """
    Link-level contact vibration / audio sensor.

    Reads the rigid solver's contact forces on the attached link and the relative velocity at the contact, then
    synthesizes a high-rate vibration waveform via source-filter modal synthesis: the contact-force onset excites a
    bank of damped modal oscillators (impact ring-down) and a velocity-scaled noise source drives a texture resonator
    (sliding roughness). The timbre is keyed by the material of the *struck* link via ``properties_dict``.

    Each ``scene.step()`` emits a block of ``audio_substeps`` samples (the physics step is the slow envelope; the
    block synthesis runs above the physics Nyquist), so the effective sample rate is ``audio_substeps / dt`` Hz. The
    ``read()`` output has shape ``(audio_substeps,)`` per environment; concatenating blocks across steps yields a
    continuous waveform suitable for the Pacinian band or for writing to an audio file.

    Note
    ----
    The synthesized signal is a single (mono) normal-acceleration-like channel per sensor. Vibration is a whole-body
    phenomenon, so it is reported per link rather than per taxel.

    Parameters
    ----------
    properties_dict : dict[int, ContactAudioProperties]
        Maps a *struck* link index (the object in contact, not the sensor's own link) to its vibroacoustic material.
        Key ``-1`` is the default for links not present in the dict; if omitted, contacts with unlisted links
        generate no sound. Shared across all ``ContactAudio`` sensors (dicts are merged).
    audio_substeps : int
        Number of synthesized samples emitted per physics step (the carrier upsampling factor K). The effective
        audio sample rate is ``audio_substeps / dt`` Hz. Default ``20``.
    n_modes : int
        Size of the modal oscillator bank. Materials with fewer modes are zero-padded. Shared across all
        ``ContactAudio`` sensors. Default ``8``.
    excitation : ExcitationSignal, optional
        If set, the sensor runs in *active-acoustic* mode (Lu & Culbertson 2023): while the sensor link is in contact,
        the given excitation is injected into the struck object's modal bank and the synthesized output is the modal
        response (the "received" waveform), whose spectrum reveals the object's resonances and how the contact damps
        them. Combine with ``roughness_gain=0`` to isolate the active response from passive scrape noise. Default
        ``None`` (passive contact-mic mode).
    velocity_gate_ref : float
        If ``> 0``, scale the synthesized output by a soft gate that tracks how fast the *sensor's own link* (the
        attached body) is moving, so the sensor goes quiet when that body is nearly still. The contact-force synthesis
        alone clicks on every tap/regrip even when the body barely moves; this gate suppresses those when the body is
        not actually in motion. The gate gain is ``motion / (motion + velocity_gate_ref)`` (0 at rest, 0.5 at
        ``velocity_gate_ref``, ->1 when fast), where ``motion = |linear_vel| + velocity_gate_ang_weight*|angular_vel|``
        of the sensor link (m/s-equivalent). ``0`` (default) disables the gate entirely (unchanged behavior).
    velocity_gate_ang_weight : float
        Weight converting the sensor link's angular speed (rad/s) to a linear-equivalent (m/s) in the gate's motion
        metric, roughly the body's radius. Only used when ``velocity_gate_ref > 0``. Default ``0``.
    velocity_gate_smooth : float
        One-pole smoothing coefficient in ``(0, 1]`` applied to the gate gain across physics steps (1 = no smoothing,
        smaller = slower/heavier) so the gain cannot step abruptly and create its own clicks. Only used when
        ``velocity_gate_ref > 0``. Default ``1.0``.
    """

    properties_dict: dict[int, ContactAudioProperties] = Field(default_factory=dict)
    audio_substeps: PositiveInt = 20
    n_modes: PositiveInt = 8
    excitation: ExcitationSignal | None = None
    velocity_gate_ref: float = Field(default=0.0, ge=0.0)
    velocity_gate_ang_weight: float = Field(default=0.0, ge=0.0)
    velocity_gate_smooth: float = Field(default=1.0, gt=0.0, le=1.0)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.excitation is not None and self.excitation.kind not in ("impulse", "linear_sweep", "exp_sweep"):
            gs.raise_exception(
                f"ExcitationSignal.kind must be 'impulse', 'linear_sweep', or 'exp_sweep', got "
                f"'{self.excitation.kind}'."
            )
        for link_idx, props in self.properties_dict.items():
            if not (len(props.modal_freqs) == len(props.modal_decays) == len(props.modal_gains)):
                gs.raise_exception(
                    f"ContactAudioProperties for link {link_idx}: modal_freqs, modal_decays, and modal_gains must "
                    f"have equal length. Got {len(props.modal_freqs)}, {len(props.modal_decays)}, "
                    f"{len(props.modal_gains)}."
                )
            if len(props.modal_freqs) > self.n_modes:
                gs.raise_exception(
                    f"ContactAudioProperties for link {link_idx} has {len(props.modal_freqs)} modes, exceeding "
                    f"n_modes={self.n_modes}."
                )


class SpatialAudio(KinematicSensorOptionsMixin["SpatialAudioSensor"], SimpleSensorOptions["SpatialAudioSensor"]):
    """
    Airborne point-microphone sensor: a mono listener in world space that renders the airborne sound radiated by the
    scene's ``ContactAudio`` (contact-mic) sensors, with geometric propagation -- distance attenuation plus a
    speed-of-sound delay (and the Doppler shift a changing delay implies).

    The listener is either *static* (``entity_idx < 0``: fixed at ``pos_offset`` in world frame) or *attached* to a
    link (``entity_idx >= 0``: riding at ``link_pos + pos_offset``, e.g. a head). Every ``ContactAudio`` sensor in the
    scene is treated as a point radiation source located at its sensor link, and its synthesized structure-borne block
    is reused as the radiated signal. This is a deliberate batched approximation -- radiated pressure is taken
    proportional to surface normal acceleration (the Neumann boundary intuition of Wang et al. 2018), not a true
    radiation/directivity model. The source block is consumed one physics step late, so the mic is independent of
    sensor step order.

    Each ``scene.step()`` emits a block of ``audio_substeps`` samples (which must equal the ``ContactAudio`` sources'
    ``audio_substeps``); ``read()`` returns shape ``(audio_substeps,)`` per environment, concatenable into a continuous
    waveform exactly like ``ContactAudio``.

    Parameters
    ----------
    audio_substeps : int
        Samples emitted per physics step; must equal the ``ContactAudio`` sources' ``audio_substeps``. Default ``20``.
    speed_of_sound : float
        Propagation speed (m/s) for the source->listener delay. Default ``343``.
    ref_distance : float
        Distance (m) at which attenuation is unity, and the near-field rolloff floor (gain is clamped for
        ``r < ref_distance`` to avoid the ``1/r`` singularity). Default ``0.1``.
    attenuation : str
        Distance rolloff law: ``"inverse"`` (1/r) or ``"inverse_square"`` (1/r^2). Default ``"inverse"``.
    enable_doppler : bool
        If True, ramp the propagation delay across each block from last step's value to this step's, so a moving
        source or listener produces a Doppler pitch shift (and block-boundary discontinuities are avoided). Default
        ``True``.
    max_delay : float
        Maximum modeled propagation delay (s); sizes the internal source-history buffer and clamps larger delays. Set
        to at least ``max_source_distance / speed_of_sound``. Default ``0.03`` (~10 m at 343 m/s).
    enable_occlusion : bool
        Reserved for raycast-based occlusion; not yet implemented (raises if set True). Default ``False``.
    """

    audio_substeps: PositiveInt = 20
    speed_of_sound: PositiveFloat = 343.0
    ref_distance: PositiveFloat = 0.1
    attenuation: str = "inverse"
    enable_doppler: StrictBool = True
    max_delay: PositiveFloat = 0.03
    enable_occlusion: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.attenuation not in ("inverse", "inverse_square"):
            gs.raise_exception(
                f"SpatialAudio attenuation must be 'inverse' or 'inverse_square', got '{self.attenuation}'."
            )
        if self.enable_occlusion:
            gs.raise_exception("SpatialAudio enable_occlusion is not yet implemented.")


class TemperatureProperties(NamedTuple):
    """
    Material properties for temperature sensor.

    Parameters
    ----------
    base_temperature: float
        The base temperature of the material in Celsius.
    conductivity: float
        The conductivity of the material in W/(m·K)
    density: float
        The density of the material in kilograms per cubic meter.
    specific_heat: float
        The specific heat of the material in J/(kg·C).
    emissivity: float
        The emissivity of the material, between 0 and 1.
    """

    base_temperature: float = 21.0
    conductivity: float = 50.0
    density: float = 1000.0
    specific_heat: float = 1.0
    emissivity: float = 0.9


class TemperatureGrid(RigidSensorOptionsMixin["TemperatureGridSensor"], SimpleSensorOptions["TemperatureGridSensor"]):
    """
    Sensor that returns the temperature in Celsius of the associated RigidLink in its local frame.
    Temperature is computed based on object contacts and their material properties provided to these options.

    Parameters
    ----------
    properties_dict: dict[int, TemperatureProperties]
        A dictionary which maps link indices to their temperature-related material properties. Key `-1` is
        used as the default for links not present in the dict; if omitted, unlisted links are ignored in contacts.
        This parameter is shared across all Temperature sensors (dicts will be merged).
    ambient_temperature: float
        The ambient temperature in Celsius. Default is 21°C.
        This parameter is shared across all Temperature sensors (the last one set will be used).
    convection_coefficient: float
        Convection coefficient h in W/(m²·K) for surface cooling. Default 1.0.
        This parameter is shared across all Temperature sensors (the last one set will be used).
    simulate_all_link_temperatures: bool
        If True, the temperatures of all links with temperature properties will be simulated.
        When False, other links are treated as adiabatic (no heat transfer, always at base temperature).
        This parameter is shared across all Temperature sensors (setting True for one sets it for all).
    grid_size: tuple[int, int, int]
        The size of the grid in the x, y, and z directions which determines the sensor resolution by spatially
        discretizing the bounding box of the rigid entity link.
    heat_generation: Grid3DFloatType | None
        The heat generation rate in Watts per square meter for each cell in the grid.
    sensor_time_constant: float
        The time constant of the sensor in seconds.
    contact_depth_weight: float
        The weight of the contact depth in the temperature calculation.
    debug_temperature_range: tuple[float, float], optional
        The range of temperatures to visualize in the debug mode. Defaults to (0.0, 100.0).
    """

    properties_dict: dict[int, TemperatureProperties] = Field(default_factory=dict)
    ambient_temperature: float | None = None
    convection_coefficient: float | None = None
    simulate_all_link_temperatures: bool = False

    grid_size: PositiveVec3IType = (1, 1, 1)
    heat_generation: Grid3DFloatType | None = None
    sensor_time_constant: NonNegativeFloat = 0.0
    contact_depth_weight: NonNegativeFloat = 1.0
    debug_temperature_range: Vec2FType = (0.0, 100.0)


class IMU(RigidSensorOptionsMixin["IMUSensor"], SimpleSensorOptions["IMUSensor"]):
    """
    IMU sensor returns the linear acceleration (accelerometer) and angular velocity (gyroscope)
    of the associated entity link.

    Parameters
    ----------
    acc_resolution : float, optional
        The measurement resolution of the accelerometer (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    acc_cross_axis_coupling : float | array-like[float, float, float] | array-like with shape (3,3)
        Accelerometer axes alignment as a 3x3 rotation matrix, where diagonal elements represent alignment (0.0 to 1.0)
        for each axis, and off-diagonal elements account for cross-axis misalignment effects.
        - If a scalar is provided (float), all off-diagonal elements are set to the scalar value.
        - If a 3-element vector is provided (array-like[float, float, float]), off-diagonal elements are set.
        - If a full 3x3 matrix is provided, it is used directly.
    acc_bias : array-like[float, float, float]
        The constant additive bias for each axis of the accelerometer.
    acc_noise : array-like[float, float, float]
        The standard deviation of the white noise for each axis of the accelerometer.
    acc_random_walk : array-like[float, float, float]
        The standard deviation of the random walk, which acts as accumulated bias drift.
    gyro_resolution : float, optional
        The measurement resolution of the gyroscope (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    gyro_cross_axis_coupling : float | array-like[float, float, float] | array-like with shape (3,3)
        Gyroscope axes alignment as a 3x3 rotation matrix, similar to `acc_cross_axis_coupling`.
    gyro_bias : array-like[float, float, float]
        The constant additive bias for each axis of the gyroscope.
    gyro_noise : array-like[float, float, float]
        The standard deviation of the white noise for each axis of the gyroscope.
    gyro_random_walk : array-like[float, float, float]
        The standard deviation of the bias drift for each axis of the gyroscope.
    mag_resolution : float, optional
        The measurement resolution of the magnetometer (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    mag_cross_axis_coupling : float | array-like[float, float, float] | array-like with shape (3,3)
        Magnetometer axes alignment as a 3x3 rotation matrix, similar to `acc_cross_axis_coupling`.
    mag_bias : array-like[float, float, float]
        The constant additive bias for each axis of the magnetometer.
    mag_noise : array-like[float, float, float]
        The standard deviation of the white noise for each axis of the gyroscope.
    mag_random_walk : array-like[float, float, float]
        The standard deviation of the bias drift for each axis of the magnetometer.
    debug_acc_color : array-like[float, float, float, float], optional
        The rgba color of the debug acceleration arrow. Defaults to (1.0, 0.0, 0.0, 0.6).
    debug_acc_scale: float, optional
        The scale factor for the debug acceleration arrow. Defaults to 0.01.
    debug_gyro_color : array-like[float, float, float, float], optional
        The rgba color of the debug gyroscope arrow. Defaults to (0.0, 1.0, 0.0, 0.6).
    debug_gyro_scale: float, optional
        The scale factor for the debug gyroscope arrow. Defaults to 0.01.
    debug_mag_color : array-like[float, float, float, float], optional
        The rgba color of the debug magnetometer arrow. Defaults to (0.0, 0.0, 1.0, 0.6).
    debug_mag_scale: float, optional
        The scale factor for the debug magnetometer arrow. Defaults to 0.01.
    """

    # Accelerometer
    acc_resolution: LaxVec3FType = 0.0
    acc_cross_axis_coupling: CrossCouplingAxisType = 0.0
    acc_noise: LaxVec3FType = 0.0
    acc_bias: LaxVec3FType = 0.0
    acc_random_walk: LaxVec3FType = 0.0

    # Gyroscope
    gyro_resolution: LaxVec3FType = 0.0
    gyro_cross_axis_coupling: CrossCouplingAxisType = 0.0
    gyro_noise: LaxVec3FType = 0.0
    gyro_bias: LaxVec3FType = 0.0
    gyro_random_walk: LaxVec3FType = 0.0

    # Magnetometer
    mag_resolution: LaxVec3FType = 0.0
    mag_cross_axis_coupling: CrossCouplingAxisType = 0.0
    mag_noise: LaxVec3FType = 0.0
    mag_bias: LaxVec3FType = 0.0
    mag_random_walk: LaxVec3FType = 0.0
    magnetic_field: LaxVec3FType = (0.0, 0.0, 0.5)

    debug_acc_color: UnitIntervalVec4Type = (1.0, 0.0, 0.0, 0.6)
    debug_acc_scale: PositiveFloat = 0.01
    debug_gyro_color: UnitIntervalVec4Type = (0.0, 1.0, 0.0, 0.6)
    debug_gyro_scale: PositiveFloat = 0.01
    debug_mag_color: UnitIntervalVec4Type = (0.0, 0.0, 1.0, 0.6)
    debug_mag_scale: PositiveFloat = 0.5

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        # FIXME: Resolution should be made private or converted to properties in mixin to prevent setting them directly
        self.resolution = self.acc_resolution + self.gyro_resolution + self.mag_resolution
        self.bias = self.acc_bias + self.gyro_bias + self.mag_bias
        self.random_walk = self.acc_random_walk + self.gyro_random_walk + self.mag_random_walk
        self.noise = self.acc_noise + self.gyro_noise + self.mag_noise


class SurfaceDistanceProbe(
    RigidSensorOptionsMixin["SurfaceDistanceProbeSensor"],
    SimpleSensorOptions["SurfaceDistanceProbeSensor"],
    ProbeSensorOptionsMixin["SurfaceDistanceProbeSensor"],
):
    """
    Surface distance probe that reports nearest distances from probe positions to tracked mesh surfaces.
    The read() output will provide the distances, and the nearest points can be accessed with `sensor.nearest_points`.

    Attached to a rigid entity link. Takes a list of local probe positions and a list of global link indices
    to track; for each probe, outputs the distance and nearest point (world frame) to the closest mesh
    surface among the tracked links. If no mesh is within max_range, reports max_range and the probe
    position as nearest point.

    Parameters
    ----------
    probe_local_pos : array-like[array-like[float, float, float]]
        Probe positions in link-local frame. One (x, y, z) per probe.
    probe_radius : float | array-like[float]
        Maximum sensing range in meters. When no mesh is within this distance, distance is clamped to the probe
        radius and nearest points is the probe position. Default: 0.5. Also controls the outer debug sphere.
    track_link_idx : array-like[int]
        Global link indices (solver link space) whose mesh geoms are used for distance queries.
    """

    probe_radius: PositiveFArrayType | PositiveFloat = 0.5
    track_link_idx: IArrayType = Field(default_factory=tuple)

    def validate_scene(self, scene: "Scene"):
        super().validate_scene(scene)
        n_links = scene.sim.rigid_solver.n_links
        for i, link_idx in enumerate(self.track_link_idx):
            if not (0 <= link_idx < n_links):
                gs.raise_exception(
                    f"SurfaceDistanceProbe track_link_idx[{i}]={link_idx} is out of range [0, {n_links})."
                )


class Raycaster(KinematicSensorOptionsMixin["RaycasterSensor"], SimpleSensorOptions["RaycasterSensor"]):
    """
    Raycaster sensor that performs ray casting to get distance measurements and point clouds.

    Parameters
    ----------
    pattern: RaycastPatternOptions
        The raycasting pattern for the sensor.
    min_range : float, optional
        The minimum sensing range in meters. Defaults to 0.0.
    max_range : float, optional
        The maximum sensing range in meters. Defaults to 20.0.
    no_hit_value : float, optional
        The value to return for no hit. Defaults to max_range if not specified.
    return_world_frame : bool, optional
        Whether to return points in the world frame. Defaults to False (local frame).
    debug_sphere_radius: float, optional
        The radius of each debug sphere drawn in the scene. Defaults to 0.02.
    debug_ray_start_color: array-like[float, float, float, float], optional
        The color of each debug ray start sphere drawn in the scene. Defaults to (0.5, 0.5, 1.0, 1.0).
    debug_ray_hit_color: array-like[float, float, float, float], optional
        The color of each debug ray hit point sphere drawn in the scene. Defaults to (1.0, 0.5, 0.5, 1.0).
    """

    pattern: RaycastPattern
    min_range: NonNegativeFloat = 0.0
    max_range: PositiveFloat = 20.0
    no_hit_value: float | None = None
    return_world_frame: StrictBool = False

    debug_sphere_radius: PositiveFloat = 0.02
    debug_ray_start_color: Vec4FType = (0.5, 0.5, 1.0, 1.0)
    debug_ray_hit_color: Vec4FType = (1.0, 0.5, 0.5, 1.0)

    def model_post_init(self, context: Any) -> None:
        if self.no_hit_value is None:
            self.no_hit_value = self.max_range
        if self.max_range <= self.min_range:
            gs.raise_exception(
                f"[{type(self).__name__}] max_range {self.max_range} should be greater than min_range {self.min_range}."
            )


class DepthCamera(Raycaster):
    """
    Depth camera that uses ray casting to obtain depth images.

    Parameters
    ----------
    pattern: DepthCameraPattern
        The raycasting pattern configuration for the sensor.
    """

    pattern: DepthCameraPattern
