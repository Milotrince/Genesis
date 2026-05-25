import math
import time
from typing import Any

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu

from .test_rigid_benchmarks import (
    STEP_DT,
    SceneMeta,
    get_rigid_solver_options,
    run_benchmark,
)

pytestmark = [
    pytest.mark.benchmarks,
    pytest.mark.cache(False),
]


TACTILE_SENSOR_RUNNABLES = (
    "surface_distance_probe",
    "contact_depth_probe",
    "kinematic_taxel",
    "proximity_taxel",
    "elastomer_taxel",
)

PROBE_SENSOR_RUNNABLES = (
    "surface_distance_probe",
    "contact_depth_probe",
    "kinematic_taxel",
    "proximity_taxel",
    "elastomer_taxel",
)
POINTCLOUD_SENSOR_RUNNABLES = (
    "proximity_taxel",
    "elastomer_taxel",
)

N_ENVS_VARIANTS = (512, 1024, 2048, 4096, 8192, 16384)
N_SENSORS_VARIANTS = (1, 5)
PROBE_COUNTS = (10, 100, 1000, 10_000)
SAMPLE_POINT_COUNTS = (60, 600, 6000, 60_000, 600_000)

DEFAULT_N_SENSORS = 1
DEFAULT_N_PROBES = 100
DEFAULT_N_SAMPLE_POINTS = 600
DEFAULT_N_ENVS = 1024

# Sensor imperfections applied when running the noised variant. Each is filtered through
# ``_sensor_has_field`` so it is only set on sensors that expose it (see examples/sensors/tactile_sandbox.py):
#   - ``hysteresis_*``, ``probe_gain``: tactile sensors only (not SurfaceDistanceProbe).
#   - ``probe_radius_noise``: every probe-style sensor (including SurfaceDistanceProbe).
#   - ``noise`` (white noise stddev) and ``random_walk``: every SimpleSensor (all five here).
#   - ``crosstalk_*``: KinematicTaxel only; requires a 2D probe grid (see ``_make_probe_kwargs``).
NOISE_KWARGS = {
    "hysteresis_strength": 0.5,
    "hysteresis_tau": 0.1,
    "probe_radius_noise": 0.001,
    "probe_gain": 1.5,
    "noise": 0.001,
    "random_walk": 0.0001,
    "crosstalk_strength": 0.3,
    "crosstalk_sigma": 0.01,
}


def _sensor_has_field(sensor_cls: type[Any], field_name: str) -> bool:
    return field_name in sensor_cls.model_fields


def _make_probe_kwargs(sensor_cls: type[Any], n_probes: int, half_size: float, noise: bool = False) -> dict[str, Any]:
    nx = math.ceil(math.sqrt(n_probes))
    ny = math.ceil(n_probes / nx)
    n_total = nx * ny
    n_filler = n_total - n_probes

    grid = gu.generate_grid_points_on_plane(
        lo=[-half_size, -half_size, half_size],
        hi=[half_size, half_size, half_size],
        normal=(0.0, 0.0, -1.0),
        nx=nx,
        ny=ny,
    )

    # Only KinematicTaxel and ElastomerTaxel accept probe_radius=0 filler entries; for the rest,
    # trim the flattened grid down to exactly n_probes.
    supports_filler = sensor_cls in (gs.sensors.KinematicTaxel, gs.sensors.ElastomerTaxel)
    if n_filler > 0 and not supports_filler:
        return {"probe_local_pos": grid.reshape(-1, 3)[:n_probes]}

    # ElastomerTaxel always needs the 2D grid; KinematicTaxel needs it to enable FFT crosstalk under noise.
    keep_grid = sensor_cls is gs.sensors.ElastomerTaxel or (sensor_cls is gs.sensors.KinematicTaxel and noise)
    probe_local_pos = grid if keep_grid else grid.reshape(-1, 3)
    kwargs: dict[str, Any] = {"probe_local_pos": probe_local_pos}
    if n_filler > 0:
        probe_radius = np.full(n_total, 0.01, dtype=gs.np_float)
        probe_radius[-n_filler:] = 0.0
        if keep_grid:
            probe_radius = probe_radius.reshape(nx, ny)
        kwargs["probe_radius"] = probe_radius
    return kwargs


def _make_tactile_sensor_options(
    sensor_cls: type[Any],
    *,
    box: Any,
    track_box: Any,
    n_probes: int,
    n_sample_points: int,
    half_size: float,
    noise: bool = False,
):
    sensor_kwargs = {"entity_idx": box.idx}

    if _sensor_has_field(sensor_cls, "track_link_idx"):
        sensor_kwargs["track_link_idx"] = (track_box.base_link_idx,)

    if _sensor_has_field(sensor_cls, "probe_local_pos"):
        sensor_kwargs.update(_make_probe_kwargs(sensor_cls, n_probes, half_size, noise=noise))

    if _sensor_has_field(sensor_cls, "n_sample_points"):
        sensor_kwargs["n_sample_points"] = n_sample_points

    if noise:
        for field, value in NOISE_KWARGS.items():
            if _sensor_has_field(sensor_cls, field):
                sensor_kwargs[field] = value

    return sensor_cls(**sensor_kwargs)


def make_box_pyramid_with_sensors(
    n_envs,
    sensor_cls,
    n_sensors=DEFAULT_N_SENSORS,
    n_probes=DEFAULT_N_PROBES,
    n_sample_points=DEFAULT_N_SAMPLE_POINTS,
    n_cubes=4,
    noise=False,
    **scene_kwargs,
):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                tolerance=1e-5,
            )
        ),
        **{
            "viewer_options": gs.options.ViewerOptions(
                camera_pos=(0.0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            "show_viewer": False,
            "show_FPS": False,
            **scene_kwargs,
        },
    )

    scene.add_entity(gs.morphs.Plane())

    box_size = 0.25
    box_spacing = (1.0 - 1e-3) * box_size
    box_pos_offset = (-0.5, 1.0, 0.0) + 0.5 * np.array([box_size, box_size, box_size])
    boxes = []
    for i in range(n_cubes):
        for j in range(n_cubes - i):
            box = scene.add_entity(
                gs.morphs.Box(
                    size=[box_size, box_size, box_size],
                    pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0.0, j]),
                ),
            )
            boxes.append(box)

    half_size = box_size / 2.0
    for sensor_idx in range(n_sensors):
        box = boxes[sensor_idx % len(boxes)]
        track_box = boxes[(sensor_idx + 1) % len(boxes)]
        scene.add_sensor(
            _make_tactile_sensor_options(
                sensor_cls,
                box=box,
                track_box=track_box,
                n_probes=n_probes,
                n_sample_points=n_sample_points,
                half_size=half_size,
                noise=noise,
            )
        )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    if n_envs > 0:
        for box in boxes:
            box.set_dofs_velocity(0.04 * torch.rand((n_envs, 6), dtype=gs.tc_float, device=gs.device))

    def step():
        scene.step()

    return scene, step, SceneMeta(compile_time=compile_time)


def _run_tactile_sensor_benchmark(n_envs, n_sensors, n_probes, n_sample_points, sensor_cls, noise=False):
    _, step_fn, meta = make_box_pyramid_with_sensors(
        n_envs,
        sensor_cls,
        n_sensors=n_sensors,
        n_probes=n_probes,
        n_sample_points=n_sample_points,
        noise=noise,
    )
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def n_sensors():
    return DEFAULT_N_SENSORS


@pytest.fixture
def n_probes():
    return DEFAULT_N_PROBES


@pytest.fixture
def n_sample_points():
    return DEFAULT_N_SAMPLE_POINTS


@pytest.fixture
def noise(request):
    # Default off; the noised tests flip this to True via ``indirect=["noise"]`` parametrize.
    return getattr(request, "param", False)


@pytest.fixture
def no_sensors(n_envs):
    return _run_tactile_sensor_benchmark(n_envs, 0, 0, 0, None)


@pytest.fixture
def surface_distance_probe(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.SurfaceDistanceProbe, noise=noise
    )


@pytest.fixture
def contact_depth_probe(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.ContactDepthProbe, noise=noise
    )


@pytest.fixture
def kinematic_taxel(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.KinematicTaxel, noise=noise
    )


@pytest.fixture
def elastomer_taxel(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.ElastomerTaxel, noise=noise
    )


@pytest.fixture
def proximity_taxel(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.ProximityTaxel, noise=noise
    )


# ---------------------------------------------------------------------------
# Parametrized benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runnable, n_envs, backend",
    [("no_sensors", n_envs, gs.gpu) for n_envs in N_ENVS_VARIANTS],
)
def test_scene_speed(factory_logger, request, runnable, n_envs):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": 0,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_sensors, backend",
    [
        (runnable, n_envs, n_sensors, gs.gpu)
        for runnable in TACTILE_SENSOR_RUNNABLES
        for n_envs in N_ENVS_VARIANTS
        for n_sensors in N_SENSORS_VARIANTS
    ],
)
def test_tactile_sensor_speed(factory_logger, request, runnable, n_envs, n_sensors):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_sensors, noise, backend",
    [
        (runnable, n_envs, n_sensors, True, gs.gpu)
        for runnable in TACTILE_SENSOR_RUNNABLES
        for n_envs in N_ENVS_VARIANTS
        for n_sensors in N_SENSORS_VARIANTS
    ],
    indirect=["noise"],
)
def test_noised_tactile_sensor_speed(factory_logger, request, runnable, n_envs, n_sensors, noise):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "noise": True,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_probes, backend",
    [(runnable, DEFAULT_N_ENVS, n_probes, gs.gpu) for runnable in PROBE_SENSOR_RUNNABLES for n_probes in PROBE_COUNTS],
)
def test_probe_sensor_speed_per_num_probe(factory_logger, request, runnable, n_envs, n_sensors, n_probes):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "n_probes": n_probes,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_sample_points, backend",
    [
        (runnable, DEFAULT_N_ENVS, n_sample_points, gs.gpu)
        for runnable in POINTCLOUD_SENSOR_RUNNABLES
        for n_sample_points in SAMPLE_POINT_COUNTS
    ],
)
def test_pointcloud_sensor_speed_per_num_samples(
    factory_logger,
    request,
    runnable,
    n_envs,
    n_sensors,
    n_probes,
    n_sample_points,
):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "n_probes": n_probes,
            "n_sample_points": n_sample_points,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))
