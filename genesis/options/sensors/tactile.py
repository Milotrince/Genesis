from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import (
    FArrayType,
    FGridType,
    IArrayType,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFArrayType,
    PositiveFloat,
    PositiveInt,
    PositiveVec2FType,
    UnitIntervalVec3Type,
    UnitIntervalVec4Type,
    Vec2FType,
)

from .options import (
    ProbeSensorOptionsMixin,
    ProbesWithNormalSensorOptionsMixin,
    RigidSensorOptionsMixin,
    SensorOptions,
    SensorT,
    SimpleSensorOptions,
    _check_len_match,
)

if TYPE_CHECKING:
    from genesis.engine.sensors.kinematic_tactile import (
        ContactDepthProbeSensor,
        ContactProbeSensor,
        ElastomerTaxelSensor,
        KinematicTaxelSensor,
        ProximityTaxelSensor,
    )


def _validate_filler_probe_radius(probe_radius, sensor_name: str) -> None:
    """
    Validate a ``probe_radius`` that permits 0-valued (inactive padding for grid) entries.
    """
    radii = np.atleast_1d(np.asarray(probe_radius, dtype=float))
    if np.any(radii < 0.0):
        gs.raise_exception(f"{sensor_name} probe_radius entries must be non-negative. Got {probe_radius}.")
    if not np.any(radii > 0.0):
        gs.raise_exception(f"{sensor_name} requires at least one positive probe_radius. Got {probe_radius}.")


class ViscoelasticHysteresisOptionsMixin(SensorOptions[SensorT]):
    """
    Single-Maxwell viscoelastic hysteresis applied on the measured branch only.

    Output equals ``x + hysteresis_strength * xi``, where ``xi`` is a per-cache-column state with
    ``xi_k = exp(-dt / hysteresis_tau) * xi_{k-1} + (x_k - x_{k-1})``. Equilibrium gain is 1 (steady-state output =
    steady-state input). On a step input, output transiently overshoots by ``strength``, decaying with time constant
    ``tau``. On cyclic input this gives a loading-unloading loop in output-vs-input space.

    Parameters
    ----------
    hysteresis_strength : float, optional
        Dimensionless ratio of the Maxwell branch to the equilibrium branch (``E_1 / E_inf`` with ``E_inf = 1``).
        ``0`` disables hysteresis. Default ``0``.
    hysteresis_tau : float, optional
        Relaxation time constant in seconds. Must be positive when ``hysteresis_strength > 0``.
    """

    hysteresis_strength: NonNegativeFloat = 0.0
    hysteresis_tau: NonNegativeFloat = 0.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.hysteresis_strength > 0.0 and self.hysteresis_tau <= 0.0:
            gs.raise_exception(
                f"hysteresis_tau ({self.hysteresis_tau}) must be > 0 when hysteresis_strength "
                f"({self.hysteresis_strength}) > 0."
            )


class TactileProbeSensorOptionsMixin(ProbeSensorOptionsMixin[SensorT]):
    """
    Tactile probe sensors use SDF contact-depth queries around each probe position instead of physics solver
    contact impulses. This allows fast contact sensing at arbitrary probe locations without affecting simulation.

    Note
    ----
    If this sensor is attached to a fixed entity, it will not detect contacts with other fixed entities.

    Parameters
    ----------
    debug_contact_color: array-like[float, float, float]
        RGB color of the debug probe spheres while in contact.
    probe_gain : float | array-like[float], optional
        Per-taxel multiplicative gain applied to the measured-branch contact depth. Default ``1.0`` (no gain). Accepts
        a scalar (applied to all probes) or an array matching the probe count. Force/torque scale as
        ``gain**normal_exponent`` because the spring-damper sees the gained depth.
    probe_gain_resample_range : (float, float), optional
        If set, the per-probe gain is resampled uniformly in ``(low, high)`` on every ``scene.reset()``. Disables the
        static ``probe_gain`` after the first reset. Default ``None`` (no resampling; gain stays at initial value).
    dead_taxel_probability : float, optional
        Per-probe Bernoulli probability that the taxel becomes dead on each ``scene.reset()``. Default ``0.0``
        (no dead taxels). When set, the intermediate-cache value for dead probes is overwritten by a fresh
        per-channel uniform sample in ``dead_taxel_value_range`` at the hardware-imperfections stage; the GT branch
        is untouched.
    dead_taxel_value_range : (float, float), optional
        Uniform range for the dead value sampled per channel on reset. Default ``(0.0, 0.0)``.
    """

    debug_contact_color: UnitIntervalVec3Type = (1.0, 0.2, 0.0)

    probe_gain: PositiveFArrayType | PositiveFloat = 1.0
    probe_gain_resample_range: PositiveVec2FType | None = None
    dead_taxel_probability: NonNegativeFloat = 0.0
    dead_taxel_value_range: Vec2FType = (0.0, 0.0)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        n_probes = int(np.prod(np.asarray(self.probe_local_pos).shape[:-1]))
        _check_len_match(self.probe_gain, n_probes, "probe_gain", "probe_local_pos")

        if self.probe_gain_resample_range is not None:
            low, high = float(self.probe_gain_resample_range[0]), float(self.probe_gain_resample_range[1])
            if low > high:
                gs.raise_exception(f"probe_gain_resample_range must satisfy low <= high. Got ({low}, {high}).")
        if self.dead_taxel_probability > 1.0:
            gs.raise_exception(f"dead_taxel_probability must be in [0, 1]. Got {self.dead_taxel_probability}.")
        low, high = float(self.dead_taxel_value_range[0]), float(self.dead_taxel_value_range[1])
        if low > high:
            gs.raise_exception(f"dead_taxel_value_range must satisfy low <= high. Got ({low}, {high}).")


class PointCloudTactileSensorMixin(TactileProbeSensorOptionsMixin[SensorT]):
    """
    Parameters
    ----------
    track_link_idx : array-like[int]
        Global link indices whose mesh geometry is used to sample a point cloud from.
    n_sample_points: int | array-like[int]
        Total FPS samples split across ``track_link_idx``, or one count per tracked link.
    use_visual_mesh : bool
        Whether to use the visual mesh when sampling the point cloud.
    debug_point_cloud_color : array-like[float, float, float, float]
        The rgba color of the debug tracked object point cloud spheres.
    debug_point_cloud_radius : float
        The radius of the debug tracked object point cloud spheres.
    """

    track_link_idx: IArrayType = Field(default_factory=tuple)
    n_sample_points: IArrayType | NonNegativeInt = 500
    use_visual_mesh: StrictBool = True

    debug_point_cloud_color: UnitIntervalVec4Type = (1.0, 0.8, 0.0, 1.0)
    debug_point_cloud_radius: PositiveFloat = 0.002


class ContactProbe(
    RigidSensorOptionsMixin["ContactProbeSensor"],
    SimpleSensorOptions["ContactProbeSensor"],
    TactileProbeSensorOptionsMixin["ContactProbeSensor"],
    ViscoelasticHysteresisOptionsMixin["ContactProbeSensor"],
):
    """
    Returns boolean contact per probe based on the contact depth threshold.

    Parameters
    ----------
    probe_radius : float | array-like[float] or shape ``(M, N)`` grid
        Probe sensing radius in meters. A scalar is shared by every probe; an array (or grid) must match the
        layout of ``probe_local_pos``. Array entries of ``0`` mark inactive filler probes -- they always read
        ``False`` and skip the SDF query -- so an irregular taxel set can be padded into a regular grid.
    contact_threshold: float
        Penetration depth (meters) at or above which a probe latches into contact.
    release_threshold: float, optional
        Penetration depth (meters) at or below which a latched probe releases (Schmitt-trigger hysteresis). Must be
        ``<= contact_threshold``. Defaults to ``contact_threshold`` (no hysteresis).
    """

    # Permits 0-valued (inactive filler) entries; see _validate_filler_probe_radius.
    probe_radius: PositiveFloat | FArrayType | FGridType = 0.01

    contact_threshold: NonNegativeFloat = 0.0001
    release_threshold: NonNegativeFloat | None = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        _validate_filler_probe_radius(self.probe_radius, "ContactProbe")
        if self.release_threshold is not None and self.release_threshold > self.contact_threshold:
            gs.raise_exception(
                f"release_threshold ({self.release_threshold}) must be <= contact_threshold ({self.contact_threshold})."
            )


class ContactDepthProbe(
    RigidSensorOptionsMixin["ContactDepthProbeSensor"],
    SimpleSensorOptions["ContactDepthProbeSensor"],
    TactileProbeSensorOptionsMixin["ContactDepthProbeSensor"],
    ViscoelasticHysteresisOptionsMixin["ContactDepthProbeSensor"],
):
    """
    Returns contact depth in meters per probe.
    """


class KinematicTaxel(
    RigidSensorOptionsMixin["KinematicTaxelSensor"],
    SimpleSensorOptions["KinematicTaxelSensor"],
    TactileProbeSensorOptionsMixin["KinematicTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["KinematicTaxelSensor"],
    ViscoelasticHysteresisOptionsMixin["KinematicTaxelSensor"],
):
    """
    A tactile sensor which estimates force and torque per taxel by querying contact depth relative to given probe
    normals and within the radius of the probe positions along a rigid entity link and the relative velocity of the
    probe and the entity in contact.

    The returned force is a spring-damper estimate based on contact depth and relative motion:
        v_n = dot(relative_velocity, probe_normal) * probe_normal
        v_t = relative_velocity - v_n
        s = penetration ** normal_exponent
        F = (-normal_stiffness * s * probe_normal) - (normal_damping * s * v_n) - (shear_scalar * v_t)
        T = cross(probe_local_pos, F) - twist_scalar * dot(relative_angular_velocity, probe_normal) * probe_normal
    as opposed to the actual impulse force on the link from the contact obtained from the physics solver.

    Note
    ----
    If this sensor is attached to a fixed entity, it will not detect contacts with other fixed entities.

    ``probe_local_pos`` may be either an arbitrary set of probes with shape ``(N, 3)`` or a grid-shaped set with shape
    ``(M, N, 3)``. Regular planar grids enable spatial crosstalk on the measured branch (see ``crosstalk_strength``).
    A probe whose ``probe_radius`` is 0 is treated as an inactive filler -- it reads 0 force/torque and is skipped --
    so an irregular taxel set can be padded into a regular grid for crosstalk.

    Parameters
    ----------
    probe_radius : float | array-like[float]
        Probe sensing radius in meters. A scalar is shared by every probe; an array must match the probe count.
        Array entries of 0 mark inactive filler probes (see the grid note above); at least one must be positive.
    normal_stiffness : float
        Stiffness for normal force estimation based on contact penetration depth and spring-damper model.
    normal_damping : float
        Damping for normal force estimation based on contact penetration depth and spring-damper model.
    normal_exponent : float
        Exponent for contact force estimation based on contact penetration depth and nonlinear spring-damper model.
        Default is 1.0, which means linear spring-damper model. Use 1.5 for Hertzian (spherical) contact.
    shear_scalar : float, optional
        Coefficient for shear force estimation based on relative linear velocity of the probe and entity in contact.
    twist_scalar : float, optional
        Coefficient for twist torque estimation based on relative angular velocity of the probe and entity in contact.
    crosstalk_strength : float, optional
        Spatial crosstalk mixing fraction applied on the measured branch. ``0`` (default) disables; ``1`` is pure
        Gaussian blur with sigma ``crosstalk_sigma``. Requires a validated regular grid layout for
        ``probe_local_pos`` and ``crosstalk_sigma > 0``.
    crosstalk_sigma : float, optional
        Gaussian crosstalk standard deviation in meters (same units as ``probe_local_pos`` spacing). Must be > 0
        when ``crosstalk_strength > 0``.
    """

    # Permits 0-valued (inactive filler) entries; see _validate_filler_probe_radius.
    probe_radius: PositiveFloat | FArrayType | FGridType = 0.01

    normal_stiffness: NonNegativeFloat = 1000.0
    normal_damping: NonNegativeFloat = 1.0
    normal_exponent: NonNegativeFloat = 1.0
    shear_scalar: NonNegativeFloat = 1.0
    twist_scalar: NonNegativeFloat = 1.0

    crosstalk_strength: NonNegativeFloat = 0.0
    crosstalk_sigma: NonNegativeFloat = 0.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        _validate_filler_probe_radius(self.probe_radius, "KinematicTaxel")
        if self.normal_exponent < 1.0:
            gs.raise_exception(f"normal_exponent must be greater than or equal to 1.0. Got {self.normal_exponent}.")
        if self.crosstalk_strength > 0.0 and self.crosstalk_sigma <= 0.0:
            gs.raise_exception(
                f"crosstalk_sigma ({self.crosstalk_sigma}) must be > 0 when crosstalk_strength "
                f"({self.crosstalk_strength}) > 0."
            )


class ElastomerTaxel(
    RigidSensorOptionsMixin["ElastomerTaxelSensor"],
    SimpleSensorOptions["ElastomerTaxelSensor"],
    PointCloudTactileSensorMixin["ElastomerTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["ElastomerTaxelSensor"],
    ViscoelasticHysteresisOptionsMixin["ElastomerTaxelSensor"],
):
    """
    An elastomer tactile sensor that implements HydroShear-style marker displacement from Genesis SDF queries.
    The tracked rigid links are sampled into indenter on-surface points for shear history, while marker dilation is
    queried directly from the tracked geometry SDF.

    Note
    ----
    ``probe_local_pos`` may be either an arbitrary set of probes with shape ``(N, 3)`` or a grid-shaped set with shape
    ``(M, N, 3)``. Regular planar grids with one shared normal use FFT acceleration for dilation; other layouts use the
    direct dilation path. Shear is computed directly. A probe whose ``probe_radius`` is 0 is treated as an inactive
    filler -- it reads 0 and is excluded from dilation/shear -- so an irregular taxel set can be padded into a
    regular grid for FFT acceleration.

    Note
    ----
    ``probe_gain`` is applied to ElastomerTaxel as a post-step linear scale of the measured marker displacement
    (the dilation kernel writes a single shared field for both branches). This is exact for the tangential
    dilation and shear components but approximate for the normal dilation term, which scales as
    ``depth**normal_exponent`` and would ideally scale as ``gain**normal_exponent`` rather than ``gain``. For
    gains near 1 the error is small.

    Parameters
    ----------
    probe_local_pos: array-like[array-like[float, float, float]], shape (N, 3) or (M, N, 3)
        Probe positions in link-local frame.
    probe_local_normal : array-like[float, float, float] or array-like[array-like[float, float, float]]
        Unit direction(s) in link-local frame: one normal for all probes, or one normal per probe matching
        ``probe_local_pos``.
    probe_radius : float | array-like[float]
        Probe sensing radius in meters. A scalar is shared by every probe; an array must match the probe count.
        Array entries of 0 mark inactive filler probes (see the grid note above); at least one must be positive.
    track_link_idx : array-like[int]
        Global rigid link indices whose collision geometry is queried by SDF and whose mesh is sampled for shear.
    n_sample_points: int | array-like[int]
        Total surface samples split across ``track_link_idx``, or one count per tracked link.
    lambda_d: float
        Gaussian falloff coefficient (in 1/m^2) for the dilation kernel ``exp(-lambda_d * r^2)`` that smears each
        in-contact probe's normal/tangential bulge across its neighbors. Larger values give sharper, more localized
        markers; smaller values smear the bulge across more probes.
    lambda_s: float
        Gaussian falloff coefficient (in 1/m^2) for the shear kernel ``exp(-lambda_s * r^2)`` that spreads each
        anchored tracked-surface point's tangential displacement to nearby probes. Larger values keep shear tightly
        local to the contact patch; smaller values produce a softer, more diffuse shear response.
    dilate_scale: float
        Scalar gain applied to dilation displacement.
    shear_scale: float
        Scalar gain applied to shear displacement.
    normal_exponent: float
        Exponent of the penetration-depth power law for the normal (out-of-plane) marker dilation: the normal
        bulge scales as ``depth ** normal_exponent``. Must be >= 1.0. Default ``2.0`` (the HydroShear quadratic
        normal response). Tangential dilation and shear stay linear in depth regardless of this value.
    elastomer_contact_sdf_enter: float
        Positive margin on signed distance: a tracked surface point starts anchoring shear when its elastomer SDF
        value is below ``-elastomer_contact_sdf_enter``.
    elastomer_contact_sdf_exit: float
        Positive margin: the anchor clears when the elastomer SDF value rises above ``+elastomer_contact_sdf_exit``
        (hysteresis band between enter and exit reduces chatter).

    Note
    ----
    Genesis reuses rigid-body SDFs for HydroShear queries. For non-analytic tracked meshes, the collision geometry
    should be watertight enough for signed-distance preprocessing, and the attached elastomer link's collision geometry
    should represent the compliant contact surface.
    """

    # Permits 0-valued (inactive filler) entries; see _validate_filler_probe_radius.
    probe_radius: PositiveFloat | FArrayType | FGridType = 0.01

    lambda_d: NonNegativeFloat = 700.0
    lambda_s: NonNegativeFloat = 300.0
    dilate_scale: NonNegativeFloat = 1.0
    shear_scale: NonNegativeFloat = 1.0
    normal_exponent: NonNegativeFloat = 2.0

    elastomer_contact_sdf_enter: NonNegativeFloat = 1e-5
    elastomer_contact_sdf_exit: NonNegativeFloat = 1e-4

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        _validate_filler_probe_radius(self.probe_radius, "ElastomerTaxel")
        if len(self.track_link_idx) == 0:
            gs.raise_exception("ElastomerTaxel requires at least one tracked link in track_link_idx.")
        if self.normal_exponent < 1.0:
            gs.raise_exception(f"normal_exponent must be greater than or equal to 1.0. Got {self.normal_exponent}.")


class ProximityTaxel(
    RigidSensorOptionsMixin["ProximityTaxelSensor"],
    SimpleSensorOptions["ProximityTaxelSensor"],
    PointCloudTactileSensorMixin["ProximityTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["ProximityTaxelSensor"],
    ViscoelasticHysteresisOptionsMixin["ProximityTaxelSensor"],
):
    """
    A tactile sensor which estimates force and torque per taxel from proximity to point clouds sampled on tracked
    meshes within a **spherical** sensing volume of nominal ``probe_radius`` around each taxel.

    For each taxel, every tracked point inside that sphere contributes a penetration depth ``P_i = R_eff - ||p_i - o||``
    where ``R_eff`` is drawn each simulation step when ``probe_radius_noise`` is non-zero (additive uniform noise
    in meters around the sensing radius, clipped nonnegative). Normal force is aligned with ``probe_local_normal``;
    shear uses tangential relative velocity. Generic SimpleSensor imperfections (bias, resolution, etc.) still apply.
    Outputs are in link-local frame.

    Parameters
    ----------
    probe_local_normal : array-like[array-like[float, float, float]]
        Unit direction(s) for the normal force channel in link-local frame: one ``(3,)`` for all taxels, or one row per
        taxel matching ``probe_local_pos``. Default ``(0, 0, 1)``.
    stiffness : float
        Linear spring stiffness (N/m) scaling summed penetration depths into total reported force.
    shear_coupling : float
        Scales penetration-weighted tangential slip ``sum_i P_i * v_{t,i}`` into a shear force contribution (see
        sensor documentation). Set to ``0.0`` to disable shear and use only the normal channel.
    density_scalar : int
        Reference point count for normalizing summed penetrations against tracked cloud size
        (scale is ``density_scalar / max(N_pc, 1)`` for this sensor's tracked samples).
    """

    stiffness: NonNegativeFloat = 100.0
    shear_coupling: NonNegativeFloat = 0.0
    density_scalar: PositiveInt = 100
