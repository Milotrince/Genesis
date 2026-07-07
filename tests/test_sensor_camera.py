import gc
import sys
import weakref

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.utils.geom import pos_lookat_up_to_T, trans_quat_to_T, trans_to_T

from .conftest import SKIP_NO_LUISA, SKIP_NO_MADRONA
from .utils import assert_allclose, assert_equal, rgb_array_to_png_bytes


try:
    import LuisaRenderPy

    ENABLE_RAYTRACER = True
except ImportError:
    ENABLE_RAYTRACER = False
try:
    import gs_madrona

    ENABLE_MADRONA = True
except ImportError:
    ENABLE_MADRONA = False


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 1])
def test_rasterizer_non_batched(n_envs, show_viewer):
    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            color=(0.4, 0.4, 0.4),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.5,
            pos=(0.0, 0.0, 2.0),
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 0.5, 0.5),
        ),
    )

    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.3, 0.3, 0.3),
            pos=(1.0, 1.0, 1.0),
        ),
        surface=gs.surfaces.Rough(
            color=(0.5, 1.0, 0.5),
        ),
    )

    # Kinematic mesh in addition to the rigid entities above, so the camera-mount path is exercised against both
    # entity kinds via raster_cam_attached_kin below.
    kin_box = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.3,
            pos=(-1.0, 1.0, 1.0),
            fixed=True,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Rough(color=(0.5, 0.5, 1.0)),
    )

    raster_cam0 = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(512, 512),
            pos=(3.0, 0.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            near=0.1,
            far=100.0,
            lights=[
                {
                    "pos": (2.0, 2.0, 5.0),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 5.0,
                }
            ],
        )
    )
    raster_cam1 = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(256, 256),
            pos=(0.0, 3.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=45.0,
        )
    )
    raster_cam_attached = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(320, 240),
            pos=(0.0, 0.0, 1.0),  # Relative to link when attached
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=70.0,
            entity_idx=sphere.idx,  # Attach to sphere
            link_idx_local=0,
        )
    )
    offset_T = np.eye(4)
    offset_T[2, 3] = 1.0
    raster_cam_offset_T = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(320, 240),
            pos=(0.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=70.0,
            entity_idx=sphere.idx,
            link_idx_local=0,
            offset_T=offset_T,
        )
    )
    raster_cam_attached_kin = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(320, 240),
            pos=(0.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=70.0,
            entity_idx=kin_box.idx,  # Mount on a kinematic entity to exercise the cross-solver attach path.
            link_idx_local=0,
        )
    )

    scene.build(n_envs=n_envs)
    for _ in range(10):
        scene.step()
    data_cam0 = raster_cam0.read()
    data_cam1 = raster_cam1.read()
    data_attached = raster_cam_attached.read()
    data_offset_T = raster_cam_offset_T.read()
    data_attached_kin = raster_cam_attached_kin.read()

    for _cam_name, data in [
        ("cam0", data_cam0),
        ("cam1", data_cam1),
        ("attached", data_attached),
        ("offset_T", data_offset_T),
        ("attached_kin", data_attached_kin),
    ]:
        rgb_np = tensor_to_array(data.rgb)
        mean = np.mean(rgb_np)
        assert 1.0 < mean < 254.0
        variance = np.var(rgb_np)
        assert variance > 1.0
    data_env0 = raster_cam0.read(envs_idx=0)
    assert data_env0.rgb.shape == (512, 512, 3)

    def _get_camera_world_pos(sensor):
        renderer = sensor._shared_metadata.renderer
        context = sensor._shared_metadata.context
        node = renderer._camera_nodes[sensor.camera.uid]
        pose = context._scene.get_pose(node)
        if pose.ndim == 3:
            pose = pose[0]
        return pose[:3, 3].copy()

    cam_pos_initial = _get_camera_world_pos(raster_cam_attached)
    cam_pos_initial_offset_T = _get_camera_world_pos(raster_cam_offset_T)

    for _ in range(10):  # Test over multiple steps
        scene.step()

    raster_cam_attached.read()
    cam_pos_final = _get_camera_world_pos(raster_cam_attached)
    cam_move_dist = np.linalg.norm(cam_pos_final - cam_pos_initial)
    assert cam_move_dist > 1e-2
    raster_cam_offset_T.read()
    cam_pos_final_offset_T = _get_camera_world_pos(raster_cam_offset_T)
    cam_move_dist_offset_T = np.linalg.norm(cam_pos_final_offset_T - cam_pos_initial_offset_T)
    assert cam_move_dist_offset_T > 1e-2
    assert_allclose(cam_move_dist_offset_T, cam_move_dist, atol=1e-2)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_rasterizer_batched(show_viewer, png_snapshot):
    CAM_RES = (128, 128)

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(-2.0, 0.0, 0.0),
            plane_size=(8.0, 3.0),
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.3,
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 0.5, 0.5),
        ),
    )
    camera = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=CAM_RES,
            pos=(4.0, 0.0, 1.5),
            lookat=(0.0, 0.0, 0.5),
            fov=60.0,
            draw_debug=show_viewer,
        )
    )
    scene.build(n_envs=2)

    # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
    camera._shared_metadata.context.shadow = False
    # Small discrepancy on apple software renderer
    if sys.platform == "darwin" and scene.visualizer.is_software:
        png_snapshot.extension._std_err_threshold = 2.0

    sphere.set_pos([[0.0, 0.0, 2.0], [1.0, 0.5, 0.3]])
    scene.step()

    data = camera.read()

    assert data.rgb.shape == (2, *CAM_RES, 3)
    assert data.rgb.dtype == torch.uint8
    assert (data.rgb[0] != data.rgb[1]).any(), "Frames should be different"

    # Ground-truth per-env identity (independent of any screen-axis assumption): env 1's sphere sits closer to the
    # camera (x=1 vs x=0), so it must cover strictly more pixels at its own batch index. A reversed batched read would
    # place the larger sphere at index 0 and fail this.
    sphere_px = [int(((data.rgb[i][..., 0].int() - data.rgb[i][..., 1].int()) > 40).sum()) for i in range(scene.n_envs)]
    assert sphere_px[1] > sphere_px[0], f"data.rgb index must match env index; got per-env sphere pixels {sphere_px}"

    for i in range(scene.n_envs):
        assert rgb_array_to_png_bytes(data.rgb[i]) == png_snapshot


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_rasterizer_attached_batched(show_viewer, png_snapshot, tol):
    png_snapshot.extension._std_err_threshold = 1.1

    scene = gs.Scene(show_viewer=show_viewer)

    # Add a plane
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    # Add a sphere
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.3,
            pos=(0.0, 0.0, 1.0),
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 0.5, 0.5),
        ),
    )

    cam_pos = (-0.4, 0.1, 2.0)
    cam_lookat = (-0.6, 0.4, 1.0)
    cam_up = (0.0, 0.0, 1.0)
    camera = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(64, 64),
            pos=cam_pos,
            lookat=cam_lookat,
            up=cam_up,
            fov=60.0,
            entity_idx=sphere.idx,
            draw_debug=show_viewer,
        )
    )

    scene.build(n_envs=2)

    # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
    camera._shared_metadata.context.shadow = False

    sphere.set_pos([[0.0, 0.0, 1.0], [0.2, 0.0, 0.5]])
    # 45° around Z for env 0, 30° around X for env 1
    sphere.set_quat([[1.0, 0.0, 0.0, 0.4], [1.0, 0.3, 0.0, 0.0]])
    scene.step()

    data = camera.read()

    assert data.rgb.shape == (2, 64, 64, 3)
    assert data.rgb.dtype == torch.uint8
    try:
        assert (data.rgb[0] != data.rgb[1]).any(), "We should have different frames"
    except AssertionError:
        if sys.platform == "darwin" and scene.visualizer.is_software:
            pytest.xfail("Flaky on MacOS with Apple Software Renderer.")
        raise

    # Verify camera pose matches the analytical formula
    offset_T = pos_lookat_up_to_T(
        np.array(cam_pos, dtype=np.float32),
        np.array(cam_lookat, dtype=np.float32),
        np.array(cam_up, dtype=np.float32),
    )
    sphere_pos = tensor_to_array(sphere.get_pos())
    sphere_quat = tensor_to_array(sphere.get_quat())
    link_T = trans_quat_to_T(sphere_pos, sphere_quat)
    expected_T = link_T @ offset_T

    camera_node = camera._shared_metadata.renderer._camera_nodes[camera.camera.uid]
    actual_pose = camera._shared_metadata.context._scene.get_pose(camera_node)
    assert_allclose(actual_pose, expected_T, tol=tol)

    for i in range(scene.n_envs):
        try:
            assert rgb_array_to_png_bytes(data.rgb[i]) == png_snapshot
        except AssertionError:
            if sys.platform == "darwin" and scene.visualizer.is_software:
                pytest.xfail("Flaky on MacOS with Apple Software Renderer. Nothing but the background was rendered.")
            raise


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.skipif(not ENABLE_MADRONA, reason=SKIP_NO_MADRONA)
def test_batch_renderer(n_envs, png_snapshot):
    CAM_RES = (128, 256)

    scene = gs.Scene(
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.5,
            pos=(0.0, 0.0, 1.0),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.5, 0.5),
        ),
    )

    camera_common_options = dict(
        res=CAM_RES,
        pos=(-2.0, 0.0, 1.5),
        lookat=(0.0, 0.0, 1.0),
        up=(0.0, 0.0, 1.5),
        fov=70.0,
        lights=[
            dict(
                pos=(2.0, 2.0, 5.0),
                color=(1.0, 0.5, 0.25),
                intensity=1.0,
                directional=False,
            )
        ],
        use_rasterizer=True,
    )
    camera_1 = scene.add_sensor(gs.sensors.BatchRendererCameraOptions(**camera_common_options))
    camera_2 = scene.add_sensor(
        gs.sensors.BatchRendererCameraOptions(
            **camera_common_options,
            entity_idx=sphere.idx,
            link_idx_local=0,
            offset_T=trans_to_T(np.array([0.0, 0.0, 3.0])),
        )
    )

    scene.build(n_envs=n_envs)

    scene.step()
    for camera in (camera_1, camera_2):
        data = camera.read()
        if n_envs > 0:
            for i in range(n_envs):
                assert rgb_array_to_png_bytes(data.rgb[i]) == png_snapshot
        else:
            assert rgb_array_to_png_bytes(data.rgb) == png_snapshot


@pytest.mark.required
def test_destroy_unbuilt_scene_with_camera():
    """Test that destroy on an unbuilt scene with cameras doesn't crash."""
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_sensor(gs.sensors.RasterizerCameraOptions(res=(64, 64)))

    scene.destroy()


@pytest.mark.required
def test_destroy_idempotent_with_camera():
    """Test that calling destroy twice on a scene with cameras doesn't crash."""
    scene = gs.Scene(show_viewer=False)
    camera = scene.add_sensor(gs.sensors.RasterizerCameraOptions(res=(64, 64)))

    scene.build()
    camera.read()

    scene.destroy()
    scene.destroy()


@pytest.mark.required
def test_rasterizer_destroy():
    scene = gs.Scene(show_viewer=False)
    cam1 = scene.add_sensor(gs.sensors.RasterizerCameraOptions(res=(64, 64)))
    cam2 = scene.add_sensor(gs.sensors.RasterizerCameraOptions(res=(32, 32)))

    scene.build()
    cam1.read()
    cam2.read()

    offscreen_renderer_ref = weakref.ref(cam1._shared_metadata.renderer._renderer)
    scene.destroy()
    gc.collect()

    assert offscreen_renderer_ref() is None


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.skipif(not ENABLE_MADRONA, reason=SKIP_NO_MADRONA)
def test_batch_renderer_destroy():
    scene = gs.Scene(show_viewer=False)
    # FIXME: This test fails without any entities in the scene.
    scene.add_entity(morph=gs.morphs.Plane())
    cam1 = scene.add_sensor(gs.sensors.BatchRendererCameraOptions(res=(64, 64), use_rasterizer=True))
    cam2 = scene.add_sensor(gs.sensors.BatchRendererCameraOptions(res=(64, 64), use_rasterizer=True))

    scene.build()
    cam1.read()
    cam2.read()

    shared_metadata = cam1._shared_metadata
    assert cam1._shared_metadata is cam2._shared_metadata
    assert len(shared_metadata.sensors) == 2
    assert shared_metadata.renderer is not None

    scene.destroy()

    assert shared_metadata.sensors is None
    assert shared_metadata.renderer is None


@pytest.mark.required
@pytest.mark.skipif(not ENABLE_RAYTRACER, reason=SKIP_NO_LUISA)
def test_raytracer_destroy():
    scene = gs.Scene(
        renderer=gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
            ),
            env_radius=20.0,
        ),
        show_viewer=False,
    )

    cam1 = scene.add_sensor(gs.sensors.RaytracerCameraOptions(res=(64, 64)))
    cam2 = scene.add_sensor(gs.sensors.RaytracerCameraOptions(res=(64, 64)))

    scene.build()
    cam1.read()
    cam2.read()

    shared_metadata = cam1._shared_metadata
    assert cam1._shared_metadata is cam2._shared_metadata
    assert len(shared_metadata.sensors) == 2
    assert shared_metadata.renderer is not None

    scene.destroy()

    assert shared_metadata.sensors is None
    assert shared_metadata.renderer is None


@pytest.mark.required
@pytest.mark.skipif(not ENABLE_RAYTRACER, reason=SKIP_NO_LUISA)
def test_raytracer_attached_without_offset_T():
    """Test that RaytracerCameraSensor works when attached without explicit offset_T.

    Also checks consistency with a scene-level camera (scene.add_camera) using the same
    pose and attachment, to make sure both camera APIs produce matching output.
    """
    CAM_RES = (128, 64)
    CAM_POS = (1.0, 0.5, 2.0)

    scene = gs.Scene(renderer=gs.renderers.RayTracer())
    scene.add_entity(morph=gs.morphs.Plane())
    sphere = scene.add_entity(morph=gs.morphs.Sphere())

    # Sensor camera attached WITHOUT offset_T - should use pos as offset.
    # The off-axis pos/lookat produce a non-identity rotation in the offset transform.
    camera_common_options = dict(
        res=CAM_RES,
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=30.0,
        spp=64,
        denoise=False,
    )
    sensor_camera = scene.add_sensor(
        gs.sensors.RaytracerCameraOptions(
            **camera_common_options,
            pos=CAM_POS,
            entity_idx=sphere.idx,
        )
    )

    # Scene-level camera with the same pose, attached with explicit offset_T
    scene_camera = scene.add_camera(
        **camera_common_options,
    )

    scene.build()

    # Attach scene-level camera with equivalent offset_T
    cam_lookat = np.array(camera_common_options["lookat"], dtype=np.float32)
    cam_up = np.array(camera_common_options["up"], dtype=np.float32)
    scene_camera.attach(
        sphere.base_link,
        offset_T=pos_lookat_up_to_T(np.array(CAM_POS, dtype=np.float32), cam_lookat, cam_up),
    )

    scene.step()

    sensor_data = sensor_camera.read()
    assert sensor_data.rgb.shape == (CAM_RES[1], CAM_RES[0], 3)
    assert sensor_data.rgb.float().std() > 1.0, "Sensor camera RGB std too low, image may be blank"

    scene_camera.move_to_attach()
    scene_rgb, *_ = scene_camera.render(rgb=True, force_render=True)
    scene_rgb = tensor_to_array(scene_rgb, dtype=np.int32)
    sensor_rgb = tensor_to_array(sensor_data.rgb, dtype=np.int32)

    # Both cameras should produce the same image
    assert_equal(sensor_rgb, scene_rgb)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 1])
@pytest.mark.skipif(not ENABLE_RAYTRACER, reason=SKIP_NO_LUISA)
def test_raytracer(n_envs, png_snapshot):
    # Relax pixel matching because RayTracer is not deterministic between different hardware (eg RTX6000 vs H100), even
    # without denoiser.
    png_snapshot.extension._blurred_kernel_size = 3

    scene = gs.Scene(
        renderer=gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(
                    color=(0.2, 0.3, 0.5),
                ),
            ),
            env_radius=20.0,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.5,
            pos=(0.0, 0.0, 1.0),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.5, 0.5),
        ),
    )

    camera_common_options = dict(
        res=(128, 256),
        pos=(-2.0, 0.0, 1.5),
        lookat=(0.0, 0.0, 1.0),
        up=(0.0, 0.0, 1.5),
        fov=70.0,
        model="pinhole",
        spp=64,
        denoise=False,
        lights=[
            dict(
                pos=(2.0, 2.0, 5.0),
                color=(10.0, 10.0, 10.0),
                intensity=1.0,
            )
        ],
    )
    camera_1 = scene.add_sensor(
        gs.sensors.RaytracerCameraOptions(
            **camera_common_options,
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(
                    color=(0.2, 0.3, 0.5),
                ),
            ),
            env_radius=20.0,
        )
    )
    camera_2 = scene.add_sensor(
        gs.sensors.RaytracerCameraOptions(
            **camera_common_options,
            entity_idx=sphere.idx,
            link_idx_local=0,
            offset_T=trans_to_T(np.array([0.0, 0.0, 3.0])),
        )
    )

    scene.build(n_envs=n_envs)

    scene.step()
    for camera in (camera_1, camera_2):
        data = camera.read()
        if n_envs > 0:
            for i in range(n_envs):
                assert rgb_array_to_png_bytes(data.rgb[i]) == png_snapshot
        else:
            assert rgb_array_to_png_bytes(data.rgb) == png_snapshot


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_camera_lookat_entity(show_viewer, png_snapshot):
    scene = gs.Scene(show_viewer=show_viewer)

    scene.add_entity(morph=gs.morphs.Plane())

    # Colored spheres at distinct locations so each camera sees different content
    attach_sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.5,
            pos=(0.0, 0.0, 0.5),
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 0.2, 0.2),
        ),
    )
    for pos, color in (
        ((2.0, 0.0, 0.5), (0.2, 1.0, 0.2)),
        ((0.0, 1.5, 0.5), (0.3, 0.3, 1.0)),
        ((0.0, -1.5, 0.5), (1.0, 1.0, 0.0)),
    ):
        scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=0.5,
                pos=pos,
            ),
            surface=gs.surfaces.Smooth(
                color=color,
            ),
        )

    cameras = []
    for camera_options in (
        # Attached cameras: same offset position, different lookat targets
        dict(pos=(0.0, 0.0, 1.5), lookat=(0.0, 1.5, 0.5), fov=70.0, entity_idx=attach_sphere.idx, link_idx_local=0),
        dict(pos=(0.0, 0.0, 1.5), lookat=(0.0, -1.5, 0.5), fov=70.0, entity_idx=attach_sphere.idx, link_idx_local=0),
        # Detached cameras: same position, different lookat targets
        dict(pos=(0.0, 0.0, 2.5), lookat=(0.0, 0.0, 0.5), fov=60.0),
        dict(pos=(0.0, 0.0, 2.5), lookat=(2.0, 0.0, 0.5), fov=60.0),
    ):
        camera = scene.add_sensor(
            gs.sensors.RasterizerCameraOptions(
                res=(64, 64),
                up=(0.0, 0.0, 1.0),
                **camera_options,
            ),
        )
        cameras.append(camera)

    scene.build()

    # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
    for camera in cameras:
        camera._shared_metadata.context.shadow = False

    # Snapshot check for every camera
    for camera in cameras:
        try:
            assert rgb_array_to_png_bytes(camera.read().rgb) == png_snapshot
        except AssertionError:
            if sys.platform == "darwin" and scene.visualizer.is_software:
                pytest.xfail("Flaky on MacOS with Apple Software Renderer. Nothing but the background was rendered.")
            raise


def _modality_scene(res, **cam_kwargs):
    """A small scene with several distinct entities and one rasterizer camera; returns (scene, camera)."""
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.Sphere(radius=0.5, pos=(0.0, 0.0, 2.0)),
        surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(1.0, 1.0, 1.0)),
        surface=gs.surfaces.Rough(color=(0.5, 1.0, 0.5)),
    )
    camera = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=res,
            pos=(3.0, 0.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            near=0.1,
            far=100.0,
            lights=[{"pos": (2.0, 2.0, 5.0), "color": (1.0, 1.0, 1.0), "intensity": 5.0}],
            **cam_kwargs,
        )
    )
    return scene, camera


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rasterizer_modalities(n_envs):
    """Rasterizer camera sensor returns rgb/depth/segmentation/normal with the right shapes, dtypes, and content."""
    CAM_RES = (128, 96)  # (width, height)
    W, H = CAM_RES
    scene, camera = _modality_scene(
        CAM_RES, render_rgb=True, render_depth=True, render_segmentation=True, render_normal=True
    )
    scene.build(n_envs=n_envs)
    camera._shared_metadata.context.shadow = False
    for _ in range(3):
        scene.step()

    data = camera.read()
    batch = () if n_envs == 0 else (n_envs,)
    assert data.rgb.shape == (*batch, H, W, 3) and data.rgb.dtype == torch.uint8
    assert data.depth.shape == (*batch, H, W) and data.depth.dtype == torch.float32
    assert data.segmentation.shape == (*batch, H, W) and data.segmentation.dtype == torch.int32
    assert data.normal.shape == (*batch, H, W, 3) and data.normal.dtype == torch.float32

    depth = data.depth
    finite = depth[torch.isfinite(depth)]
    assert (finite > 0).all(), "depth must be positive"
    assert (finite < 99.0).any(), "expected geometry closer than the far plane"
    assert torch.unique(data.segmentation).numel() > 1, "expected multiple segmentation ids"


@pytest.mark.required
def test_rasterizer_modality_defaults_rgb_only():
    """Default options request only RGB; the other modalities read back as None."""
    scene, camera = _modality_scene((64, 64))
    scene.build(n_envs=0)
    camera._shared_metadata.context.shadow = False
    scene.step()
    data = camera.read()
    assert data.rgb is not None
    assert data.depth is None and data.segmentation is None and data.normal is None
    # Camera sensors are read via sensor.read(), never through read_sensors().
    assert scene.read_sensors() == {}


@pytest.mark.required
def test_read_cameras():
    # Cameras are excluded from the vector reader and returned by the dedicated camera reader.
    W, H = 64, 48
    scene, camera = _modality_scene((W, H), render_depth=True)
    scene.build(n_envs=0)
    camera._shared_metadata.context.shadow = False
    scene.step()

    assert scene.read_sensors() == {}

    cams = scene.read_cameras()
    assert list(cams.keys()) == [camera]
    data = cams[camera]
    assert data.rgb is not None and data.rgb.shape == (H, W, 3)
    assert data.depth is not None and data.depth.shape == (H, W)


@pytest.mark.slow
@pytest.mark.required
def test_camera_coexists_with_ring_pipeline_sensor():
    # A float32 depth camera (non-ring) shares the float32 buffer with an IMU (ring-pipeline). The camera columns must
    # stay out of the IMU's transform timeline ring, both sensors must read correctly, and read_sensors() must return
    # the IMU but not the camera.
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    box = scene.add_entity(morph=gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 1.0)))
    imu = scene.add_sensor(gs.sensors.IMU(entity_idx=box.idx))
    camera = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(48, 48),
            pos=(2.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            render_rgb=False,
            render_depth=True,
        )
    )
    scene.build(n_envs=0)
    camera._shared_metadata.context.shadow = False
    scene.step()

    depth = camera.read().depth
    assert depth.shape == (48, 48) and depth.dtype == torch.float32
    assert torch.isfinite(depth).any()

    bulk = scene.read_sensors()
    assert gs.sensors.types.IMU in bulk  # ring-pipeline sensor is aggregated
    assert gs.sensors.types.RasterizerCameraOptions not in bulk  # camera is read via camera.read()
    assert imu.read() is not None


@pytest.mark.required
def test_camera_sensor_intrinsics_extrinsics():
    # The sensor delegates camera matrices to its owned vis.Camera; they must match a scene.add_camera of the same
    # pose/params (intrinsics/projection are exact; extrinsics matches the shared static world pose).
    CAM_RES = (128, 96)
    W, H = CAM_RES
    common = dict(res=CAM_RES, pos=(3.0, 0.0, 2.0), lookat=(0.0, 0.0, 1.0), up=(0.0, 0.0, 1.0), fov=60.0)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    sensor = scene.add_sensor(gs.sensors.RasterizerCameraOptions(near=0.1, far=100.0, **common))
    ref = scene.add_camera(near=0.1, far=100.0, **common)
    scene.build(n_envs=0)
    sensor._shared_metadata.context.shadow = False
    scene.step()
    sensor.read()

    assert_allclose(sensor.intrinsics, ref.intrinsics, atol=1e-6)
    assert_allclose(sensor.projection_matrix, ref.projection_matrix, atol=1e-6)
    assert_allclose(sensor.extrinsics, ref.extrinsics, atol=1e-4)
    assert_allclose(sensor.cx, W / 2, atol=1e-6)
    assert_allclose(sensor.cy, H / 2, atol=1e-6)


@pytest.mark.slow
@pytest.mark.required
def test_camera_sensor_set_pose_moves_detached():
    # A detached sensor camera can be re-posed at runtime via the delegated set_pose; the next read re-renders.
    scene, camera = _modality_scene((64, 48))
    scene.build(n_envs=0)
    camera._shared_metadata.context.shadow = False
    scene.step()
    frame_a = camera.read().rgb.clone()
    pos_a = camera.get_pos().clone()

    camera.set_pose(pos=(0.0, 3.0, 2.0), lookat=(0.0, 0.0, 1.0), up=(0.0, 0.0, 1.0))
    assert (camera.get_pos() != pos_a).any()
    frame_b = camera.read().rgb
    assert (frame_a != frame_b).any(), "re-posing a detached camera must change the rendered frame"


@pytest.mark.slow
@pytest.mark.required
def test_camera_read_after_reset_rerenders():
    # Camera storage now lives in the shared manager cache, which reset() zeroes; a read at the same timestep as a
    # preceding reset must re-render rather than return the zeroed cache.
    scene, camera = _modality_scene((64, 48))
    scene.build(n_envs=0)
    camera._shared_metadata.context.shadow = False
    scene.step()
    before = camera.read().rgb.clone()
    assert before.float().std() > 1.0  # a real rendered frame, not a blank buffer

    scene.reset()
    after = camera.read().rgb
    assert after.float().std() > 1.0, "read after reset returned a blank/zeroed frame instead of re-rendering"


@pytest.mark.slow
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_camera_history(n_envs):
    CAM_RES = (64, 48)
    W, H = CAM_RES
    scene, camera = _modality_scene(CAM_RES, render_depth=True, history_length=3)
    scene.build(n_envs=n_envs)
    camera._shared_metadata.context.shadow = False
    sphere = scene.entities[1]

    # Distinct sphere heights per step so consecutive snapshots differ. A read between steps must not disturb the
    # per-step capture, so interleave a read to guard the eager-render dedup.
    for z in (2.0, 1.4, 0.8, 0.6):
        sphere.set_pos([0.0, 0.0, z] if n_envs == 0 else [[0.0, 0.0, z]] * n_envs)
        scene.step()
        camera.read()

    data = camera.read()
    batch = () if n_envs == 0 else (n_envs,)
    # History adds a leading dimension of length 3 in front of the modality shape.
    assert data.rgb.shape == (*batch, 3, H, W, 3) and data.rgb.dtype == torch.uint8
    assert data.depth.shape == (*batch, 3, H, W) and data.depth.dtype == torch.float32

    depth_hist = data.depth if n_envs == 0 else data.depth[0]  # (3, H, W), newest-first
    assert (depth_hist[0] != depth_hist[1]).any()
    assert (depth_hist[1] != depth_hist[2]).any()


@pytest.mark.slow
@pytest.mark.required
def test_camera_delay():
    scene, camera = _modality_scene((64, 48), delay=0.02)  # 2 steps at the default dt=0.01
    scene.build(n_envs=0)
    camera._shared_metadata.context.shadow = False
    assert camera._delay_ts == 2
    sphere = scene.entities[1]

    # Step-1 frame (undelayed ground truth) is what a 2-step delayed read must reproduce two steps later. Reading each
    # step exercises the interleaved read/step path that a naive scene.t dedup would corrupt.
    sphere.set_pos([0.0, 0.0, 2.0])
    scene.step()
    frame_step1 = camera.read_ground_truth().rgb.clone()

    sphere.set_pos([0.0, 0.0, 1.2])
    scene.step()
    camera.read()
    sphere.set_pos([0.0, 0.0, 0.6])
    scene.step()

    delayed = camera.read().rgb  # measured, delayed by 2 steps
    current = camera.read_ground_truth().rgb  # undelayed, current step
    assert_equal(delayed, frame_step1)  # ZOH reproduces the exact frame from 2 steps ago
    assert (delayed != current).any()  # the delay actually shifted the observed frame


@pytest.mark.required
def test_camera_modalities_require_at_least_one():
    with pytest.raises(gs.GenesisException, match="at least one"):
        gs.sensors.RasterizerCameraOptions(
            render_rgb=False, render_depth=False, render_segmentation=False, render_normal=False
        )


@pytest.mark.slow
@pytest.mark.required
def test_rasterizer_modalities_match_add_camera():
    """Rendered depth/segmentation from the sensor match a scene.add_camera render of the same pose."""
    CAM_RES = (128, 96)
    scene, sensor = _modality_scene(CAM_RES, render_rgb=False, render_depth=True, render_segmentation=True)
    ref = scene.add_camera(
        res=CAM_RES, pos=(3.0, 0.0, 2.0), lookat=(0.0, 0.0, 1.0), up=(0.0, 0.0, 1.0), fov=60.0, near=0.1, far=100.0
    )
    scene.build(n_envs=0)
    sensor._shared_metadata.context.shadow = False
    scene.step()

    data = sensor.read()
    _, ref_depth, ref_seg, _ = ref.render(rgb=False, depth=True, segmentation=True, force_render=True)

    sensor_depth = tensor_to_array(data.depth)
    ref_depth = tensor_to_array(ref_depth)
    # Silhouette/edge pixels can differ between two independent rasterizations; require the vast majority to agree.
    close_frac = float((np.abs(sensor_depth - ref_depth) < 0.05).mean())
    assert close_frac > 0.95, f"sensor vs add_camera depth agreement too low: {close_frac:.3f}"

    # Segmentation is a foreground/background structure: the non-background masks should overlap almost perfectly.
    sensor_fg = tensor_to_array(data.segmentation) > 0
    ref_fg = tensor_to_array(ref_seg) > 0
    iou = float((sensor_fg & ref_fg).sum()) / float((sensor_fg | ref_fg).sum())
    assert iou > 0.95, f"sensor vs add_camera segmentation IoU too low: {iou:.3f}"


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.skipif(not ENABLE_MADRONA, reason=SKIP_NO_MADRONA)
def test_batch_renderer_modalities(n_envs):
    """Batch renderer camera sensors support all four modalities, incl. cameras requesting different subsets."""
    CAM_RES = (128, 96)
    W, H = CAM_RES
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.Sphere(radius=0.5, pos=(0.0, 0.0, 1.0)),
        surface=gs.surfaces.Default(color=(1.0, 0.5, 0.5)),
    )
    common = dict(res=CAM_RES, pos=(-2.0, 0.0, 1.5), lookat=(0.0, 0.0, 1.0), up=(0.0, 0.0, 1.0), fov=70.0)
    cam_all = scene.add_sensor(
        gs.sensors.BatchRendererCameraOptions(
            **common, render_rgb=True, render_depth=True, render_segmentation=True, render_normal=True
        )
    )
    # A second camera in the same batch requesting only depth exercises the per-sensor modality union path.
    cam_depth = scene.add_sensor(gs.sensors.BatchRendererCameraOptions(**common, render_rgb=False, render_depth=True))
    scene.build(n_envs=n_envs)
    scene.step()

    batch = () if n_envs == 0 else (n_envs,)
    data_all = cam_all.read()
    assert data_all.rgb.shape == (*batch, H, W, 3) and data_all.rgb.dtype == torch.uint8
    assert data_all.depth.shape == (*batch, H, W) and data_all.depth.dtype == torch.float32
    assert data_all.segmentation.shape == (*batch, H, W) and data_all.segmentation.dtype == torch.int32
    assert data_all.normal.shape == (*batch, H, W, 3) and data_all.normal.dtype == torch.float32

    data_depth = cam_depth.read()
    assert data_depth.rgb is None and data_depth.segmentation is None and data_depth.normal is None
    assert data_depth.depth.shape == (*batch, H, W)
