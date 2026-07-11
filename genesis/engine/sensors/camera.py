"""
Camera sensors for rendering: Rasterizer, Raytracer, and Batch Renderer.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, NamedTuple, Optional

import numpy as np
import torch

import genesis as gs
from genesis.options.renderers import BatchRenderer as BatchRendererOptions
from genesis.options.sensors import (
    BatchRendererCameraOptions,
    RasterizerCameraOptions,
    RaytracerCameraOptions,
    SensorOptions,
)
from genesis.options.vis import VisOptions
from genesis.utils.geom import pos_lookat_up_to_T
from genesis.vis.batch_renderer import BatchRenderer
from genesis.vis.camera import Camera
from genesis.vis.rasterizer import Rasterizer
from genesis.vis.rasterizer_context import RasterizerContext

from .base_sensor import OptionsT, KinematicSensorMetadataMixin, KinematicSensorMixin, Sensor, SharedSensorMetadata

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.batch_renderer import BatchRenderer
    from genesis.vis.rasterizer import Rasterizer
    from genesis.vis.rasterizer_context import RasterizerContext
    from genesis.vis.raytracer import Raytracer

    from .sensor_manager import SensorManager


# ========================== Data Class ==========================


class CameraReturnType(NamedTuple):
    """
    Camera sensor return data.

    Only the modalities enabled on the sensor options are populated; the rest are ``None``. ``rgb`` is kept first for
    backward compatibility with code that unpacks or accesses ``.rgb``.
    """

    rgb: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    segmentation: Optional[torch.Tensor] = None
    normal: Optional[torch.Tensor] = None


# Ordered to match the render backends' return tuple `(rgb, depth, segmentation, normal)` (see
# `Rasterizer.render_camera`, `vis.Camera.render`, and `BatchRenderer.render` which returns in `IMAGE_TYPE` order).
_CAMERA_MODALITIES = ("rgb", "depth", "segmentation", "normal")


def _modality_dtype(name: str) -> torch.dtype:
    """Per-pixel dtype for a modality: uint8 color, float32 depth/normal, int32 segmentation indices."""
    if name == "rgb":
        return torch.uint8
    if name == "segmentation":
        return torch.int32
    return torch.float32


def _modality_shape(name: str, B: int, w: int, h: int) -> tuple[int, ...]:
    """Cache buffer shape for a modality. rgb/normal are 3-channel; depth/segmentation are single-channel."""
    if name in ("rgb", "normal"):
        return (B, h, w, 3)
    return (B, h, w)


def _enabled_modalities(options) -> tuple[str, ...]:
    return tuple(name for name in _CAMERA_MODALITIES if getattr(options, f"render_{name}"))


def _to_cache_tensor(arr, dtype: torch.dtype) -> torch.Tensor:
    """Convert a rendered array (numpy, possibly with negative strides, or torch/CUDA) to a contiguous typed tensor."""
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype=dtype, device=gs.device).contiguous()
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=dtype, device=gs.device)


class StandaloneCameraBackend:
    """
    A minimal ``CameraBackendProvider`` for headless / sensor camera rendering, standing in for the interactive
    ``Visualizer``.

    A camera sensor owns a ``vis.Camera`` and builds/renders it against this provider, which exposes exactly the
    closed set of members a ``Camera`` (and, for the batch path, ``BatchRenderer``) reaches for. It backs both the
    standalone rasterizer path (``rasterizer`` set) and the batch path (``batch_renderer`` set); ``raytracer`` sensors
    use the real ``Visualizer`` instead (they require one). It never creates a viewer/GUI.
    """

    def __init__(self, scene, context, rasterizer=None, batch_renderer=None):
        self.scene = scene
        self._context = context
        self.rasterizer = rasterizer
        self.raytracer = None
        self.batch_renderer = batch_renderer
        self.has_display = False
        # `vis.Camera` instances enumerated by the batch renderer (mirrors `Visualizer._cameras`).
        self._cameras: List["Camera"] = []

    @property
    def context(self):
        return self._context

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        if self.batch_renderer is not None:
            return self.batch_renderer.colorize_seg_idxc_arr(seg_idxc_arr)
        return self._context.colorize_seg_idxc_arr(seg_idxc_arr)


# ========================== Shared Metadata ==========================


@dataclass
class RasterizerCameraSharedMetadata(KinematicSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all Rasterizer cameras."""

    # Rasterizer instance (the visualizer's when a viewer exists, else a standalone one)
    renderer: Optional["Rasterizer"] = None
    # RasterizerContext instance
    context: Optional["RasterizerContext"] = None
    # The `CameraBackendProvider` the owned `vis.Camera`s build/render against (the `Visualizer` when a viewer exists,
    # else a `StandaloneCameraBackend`).
    backend: Optional[Any] = None
    # List of light dictionaries
    lights: Optional[List[Dict[str, Any]]] = None
    # List of RasterizerCameraSensor instances
    sensors: Optional[List["RasterizerCameraSensor"]] = None
    # Track when rasterizer cameras were last updated
    last_render_timestep: int = -1

    def destroy(self):
        super().destroy()

        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
        if self.context is not None:
            self.context.destroy()
            self.context = None
        self.backend = None
        self.lights = None
        self.sensors = None


@dataclass
class RaytracerCameraSharedMetadata(KinematicSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all Raytracer cameras."""

    # Raytracer instance
    renderer: Optional["Raytracer"] = None
    # List of light objects
    lights: Optional[List[Any]] = None
    # List of RaytracerCameraSensor instances
    sensors: Optional[List["RaytracerCameraSensor"]] = None
    # Track when raytracer cameras were last updated
    last_render_timestep: int = -1

    def destroy(self):
        super().destroy()

        self.renderer = None
        self.sensors = None


@dataclass
class BatchRendererCameraSharedMetadata(KinematicSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all Batch Renderer cameras."""

    # BatchRenderer instance
    renderer: Optional["BatchRenderer"] = None
    # gs.List of lights
    lights: Optional[Any] = None
    # List of BatchRendererCameraSensor instances
    sensors: Optional[List["BatchRendererCameraSensor"]] = None
    # Track when batch was last rendered
    last_render_timestep: int = -1
    # StandaloneCameraBackend the owned `vis.Camera`s build against and which the batch renderer enumerates.
    backend: Optional["StandaloneCameraBackend"] = None

    def destroy(self):
        super().destroy()

        self.renderer = None
        self.sensors = None
        if self.backend is not None and self.backend.context is not None:
            self.backend.context.destroy()
        self.backend = None


# ========================== Base Camera Sensor ==========================


class BaseCameraSensor(KinematicSensorMixin, Sensor[OptionsT, None, SharedSensorMetadata, CameraReturnType]):
    """
    Base class for camera sensors that render multi-modality images into the shared sensor cache.

    Cameras are first-class sensors: their enabled modalities (rgb uint8 / depth float32 / segmentation int32 / normal
    float32) are declared per-instance via `_get_return_format` / `_get_cache_dtype`, and the manager lays each out in
    its own dtype's per-class cache. They therefore flow through the same delay / jitter / history / return machinery as
    every other sensor.

    `uses_ring_pipeline = False` means cameras don't use the `_apply_transform` recurrence timeline rings (they render
    directly, with no stateful transform), so no dead timeline rings are allocated - but they still get delay / jitter /
    history through the per-class return-space ring. Rendering is hybrid: lazy (on `read()`) when no delay/history is
    requested (the manager's no-ring alias fast path), eager (every step, inside `_update_shared_cache`) when a
    delay/history ring exists, since those inherently require capturing every past frame.

    This class centralizes attachment handling (via KinematicSensorMixin), the `_stale` render-dedup flag, and the
    render->cache scatter; subclasses implement the backend-specific `_render_current_state`.
    """

    uses_ring_pipeline: ClassVar[bool] = False

    def __init__(
        self,
        options: "SensorOptions",
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._enabled: tuple[str, ...] = _enabled_modalities(options)
        self._stale: bool = True
        # The owned `vis.Camera`; created lazily in each backend's `build`/first render. All pose / intrinsics /
        # rendering is delegated to it, so the sensor holds no duplicate pose or camera-matrix state.
        self._camera: Optional["Camera"] = None

    # ========================== Cache Integration (shared) ==========================

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        # One shape per enabled modality, batch dim dropped (the manager prepends B). Order follows `_CAMERA_MODALITIES`
        # (the backends' render-tuple order), matching `_get_cache_dtype` field-for-field.
        w, h = self._options.res
        return tuple(_modality_shape(name, 1, w, h)[1:] for name in _enabled_modalities(self._options))

    def _get_cache_dtype(self):
        # Per-modality dtype tuple aligned 1:1 with `_get_return_format`: uint8 rgb / float32 depth+normal / int32 seg.
        return tuple(_modality_dtype(name) for name in _enabled_modalities(self._options))

    @classmethod
    def _update_shared_cache(
        cls,
        shared_context: None,
        shared_metadata: SharedSensorMetadata,
        ground_truth_slices: dict,
        ground_truth_data_timelines: dict,
        measured_data_timelines: dict,
        intermediates: dict,
    ):
        # Hybrid render timing. Render eagerly (every step) only when this class has a delay/history return ring, which
        # inherently needs every past frame captured; the render scatters into the manager cache slices, which the ring
        # then snapshots below (in `SensorManager.step`). With no ring, this is a no-op and rendering happens lazily on
        # `read()` into the aliased cache - preserving the "don't render if nobody reads" optimization.
        #
        # Force staleness before rendering rather than reusing the `scene.t`-based dedup: this hook runs during
        # `sim.step()` while `scene.t` is still pre-increment, and an interleaved `read()` at the post-increment `t`
        # would otherwise make the dedup skip this step's render and snapshot a stale frame. `_stale` is still used to
        # dedup the batch renderer's single all-camera pass within this loop.
        sensors = shared_metadata.sensors
        if not sensors:
            return
        manager = sensors[0]._manager
        if not manager.class_has_return_ring(cls):
            return
        shared_metadata.last_render_timestep = -1
        for sensor in sensors:
            sensor._stale = True
        for sensor in sensors:
            sensor._ensure_rendered_for_current_state()

    def _scatter_render_into_manager_cache(self, render_tuple):
        """
        Scatter a backend render tuple `(rgb, depth, segmentation, normal)` into this sensor's columns of the manager's
        per-dtype intermediate + ground-truth caches. Cameras have no measured/GT distinction, so both receive the same
        rendered data. Only enabled, non-None modalities are written.
        """
        manager = self._manager
        batch = max(manager._sim.n_envs, 1)
        w, h = self._options.res
        by_name = dict(zip(_CAMERA_MODALITIES, render_tuple))
        for i, name in enumerate(self._enabled):
            arr = by_name.get(name)
            if arr is None:
                continue
            dtype = self._field_dtypes[i]
            field_slice = self._field_intermediate_slice[i]
            start = self._cache_idx_by_dtype[dtype]
            cols = slice(start + field_slice.start, start + field_slice.stop)
            tensor = _to_cache_tensor(arr, dtype)
            if tensor.ndim == len(_modality_shape(name, batch, w, h)) - 1:
                # Single-environment render missing the batch dim; add it to match the (B, ...) cache layout.
                tensor = tensor.unsqueeze(0)
            flat = tensor.reshape(batch, -1)
            manager._intermediate_cache[dtype][:, cols] = flat
            # GT cache is stored transposed (cols, B); mirror the same data since cameras have no GT/measured split.
            manager._ground_truth_intermediate_cache[dtype][cols, :] = flat.T

    def _draw_debug(self, context: "RasterizerContext"):
        """No debug drawing for cameras."""
        pass

    # ========================== Attachment / pose ==========================

    def _init_camera_pose(self):
        """
        Bind the owned `vis.Camera`'s pose to this sensor's options: attach to the configured link (offset from
        `offset_T`, or from `pos`/`lookat`/`up`), or leave the static world pose already set at `Camera.build`.
        Per-step tracking then happens through `vis.Camera.move_to_attach`.
        """
        if self._link is None:
            return
        if self._options.offset_T is not None:
            offset_T = torch.as_tensor(self._options.offset_T, dtype=gs.tc_float, device=gs.device)
        else:
            offset_T = pos_lookat_up_to_T(
                torch.as_tensor(self._options.pos, dtype=gs.tc_float, device=gs.device),
                torch.as_tensor(self._options.lookat, dtype=gs.tc_float, device=gs.device),
                torch.as_tensor(self._options.up, dtype=gs.tc_float, device=gs.device),
            )
        self._camera.attach(self._link, offset_T)

    # ========================== Hooks for subclasses ==========================

    def _render_current_state(self):
        """Perform the actual render for the current state; subclasses must implement."""
        raise NotImplementedError

    # ========================== Capability delegation to the owned vis.Camera ==========================

    def _require_camera(self) -> "Camera":
        # The owned vis.Camera is created eagerly at build for standalone/headless rendering, but lazily on the first
        # render when sharing a live viewer's rasterizer. Give a clear message instead of a NoneType error if pose /
        # intrinsics are accessed before that first render on the viewer-shared path.
        if self._camera is None:
            gs.raise_exception(
                "Camera not built yet: its underlying vis.Camera is created on the first render when sharing a live "
                "viewer. Step the scene or call `read()` once before accessing camera pose / intrinsics."
            )
        return self._camera

    @property
    def camera(self) -> "Camera":
        """The underlying `vis.Camera` this sensor renders through (pose, intrinsics, extrinsics, ...)."""
        return self._require_camera()

    @gs.assert_built
    def set_pose(self, transform=None, pos=None, lookat=None, up=None, envs_idx=None):
        """Set the camera pose. Delegates to `vis.Camera.set_pose` (per-env when batched)."""
        self._require_camera().set_pose(transform=transform, pos=pos, lookat=lookat, up=up, envs_idx=envs_idx)
        # Invalidate the lazy-render cache so the next read re-renders from the new pose within the same step.
        self._stale = True

    @gs.assert_built
    def get_pos(self, envs_idx=None):
        return self._require_camera().get_pos(envs_idx)

    @gs.assert_built
    def get_quat(self, envs_idx=None):
        return self._require_camera().get_quat(envs_idx)

    @gs.assert_built
    def get_transform(self, envs_idx=None):
        return self._require_camera().get_transform(envs_idx)

    @gs.assert_built
    def get_lookat(self, envs_idx=None):
        return self._require_camera().get_lookat(envs_idx)

    @gs.assert_built
    def get_up(self, envs_idx=None):
        return self._require_camera().get_up(envs_idx)

    @property
    def intrinsics(self):
        """The camera intrinsics matrix `K`."""
        return self._require_camera().intrinsics

    @property
    def extrinsics(self):
        """The camera extrinsics (world-to-camera) matrix."""
        return self._require_camera().extrinsics

    @property
    def projection_matrix(self):
        """The OpenGL projection matrix."""
        return self._require_camera().projection_matrix

    @property
    def f(self):
        """The focal length in pixels."""
        return self._require_camera().f

    @property
    def cx(self):
        return self._require_camera().cx

    @property
    def cy(self):
        return self._require_camera().cy

    def distance_center_to_plane(self, center_dis):
        """Convert Euclidean center distance (range along the ray) to planar Z depth."""
        return self._require_camera().distance_center_to_plane(center_dis)

    @gs.assert_built
    def render_pointcloud(self, world_frame=True):
        """Render a partial point cloud from the camera view (depth-deprojected)."""
        return self._require_camera().render_pointcloud(world_frame=world_frame)

    # ========================== Shared read() ==========================

    def _ensure_rendered_for_current_state(self):
        """Ensure this camera has an up-to-date render before reading.
        Base handles staleness and timestamps; subclasses implement _render_current_state().
        """
        scene = self._manager._sim.scene

        # If the scene time advanced, mark all cameras as stale
        if self._shared_metadata.last_render_timestep != scene.t:
            if self._shared_metadata.sensors is not None:
                for sensor in self._shared_metadata.sensors:
                    sensor._stale = True
            self._shared_metadata.last_render_timestep = scene.t

        # If this camera is not stale, cache is considered fresh
        if not self._stale:
            return

        # Build the owned vis.Camera (deferred on the viewer-shared path), track the attached link, then render.
        self._render_current_state()

        # Mark as fresh
        self._stale = False

    def _is_eager(self) -> bool:
        # Eager (delay/history) cameras are rendered every step into the return-space ring by `_update_shared_cache`;
        # `read()` then just samples the ring and must NOT re-render. Lazy (no-ring) cameras render on read into the
        # aliased cache.
        return self._manager.class_has_return_ring(type(self))

    @gs.assert_built
    def read(self, envs_idx=None) -> CameraReturnType:
        """Read this camera's modalities from the shared sensor cache, rendering first on the lazy (no-ring) path."""
        if not self._is_eager():
            self._ensure_rendered_for_current_state()
        return self._camera_format(self._manager.get_cloned_from_cache(self), envs_idx)

    @gs.assert_built
    def read_ground_truth(self, envs_idx=None) -> CameraReturnType:
        """Ground-truth read. Cameras have no measured/GT split, but the GT path is delay-free (undelayed frame)."""
        if not self._is_eager():
            self._ensure_rendered_for_current_state()
        return self._camera_format(self._manager.get_cloned_from_cache(self, is_ground_truth=True), envs_idx)

    @classmethod
    def reset(cls, shared_metadata: SharedSensorMetadata, shared_ground_truth_cache, envs_idx):
        # The shared cache backing cameras is zeroed on reset; invalidate render staleness so the next read re-renders
        # instead of returning the zeroed cache (cameras own no separate storage anymore).
        shared_metadata.last_render_timestep = -1
        if shared_metadata.sensors is not None:
            for sensor in shared_metadata.sensors:
                sensor._stale = True

    def _camera_format(self, fields, envs_idx=None) -> CameraReturnType:
        """
        Pack the per-field return blocks into a ``CameraReturnType`` keyed by enabled modality name (disabled
        modalities stay ``None``), reshaping each to its image shape and applying env selection. Mirrors the historic
        env-indexing semantics: ``envs_idx=None`` returns the batch (or the sole frame when ``n_envs==0``); an int index
        selects and squeezes a single env; a sequence gathers those envs.
        """
        n_envs = self._manager._sim.n_envs
        named: Dict[str, "torch.Tensor"] = {}
        for name, shape, field in zip(self._enabled, self._return_shapes, fields):
            buf = field.reshape((field.shape[0], *shape))  # (B, [H,] h, w[, 3])
            if envs_idx is None:
                buf = buf[0] if n_envs == 0 else buf
            else:
                buf = buf[envs_idx]
            named[name] = buf
        return CameraReturnType(**named)


# ========================== Rasterizer Camera Sensor ==========================


class RasterizerCameraSensor(
    BaseCameraSensor, Sensor[RasterizerCameraOptions, None, RasterizerCameraSharedMetadata, CameraReturnType]
):
    """
    Rasterizer camera sensor using OpenGL-based rendering.

    This sensor renders RGB images using the existing Rasterizer backend, but operates independently from the scene
    visualizer.
    """

    def __init__(
        self,
        options: RasterizerCameraOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._options: RasterizerCameraOptions

    # ========================== Sensor Lifecycle ==========================

    def build(self):
        """Initialize the rasterizer and register this camera."""
        super().build()

        scene = self._manager._sim.scene

        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = gs.List()

            # If a viewer is active, reuse its windowed OpenGL context (and the visualizer as the camera backend) for
            # both offscreen and onscreen rendering, rather than a separate headless context which is fragile.
            if scene.viewer is not None:
                self._shared_metadata.context = scene.visualizer.context
                self._shared_metadata.renderer = scene.visualizer.rasterizer
                self._shared_metadata.backend = scene.visualizer
            else:
                # No viewer - create standalone rasterizer with offscreen context + a standalone camera backend.
                context = self._create_standalone_context(scene)
                renderer = Rasterizer(viewer=None, context=context)
                renderer.build()
                self._shared_metadata.context = context
                self._shared_metadata.renderer = renderer
                self._shared_metadata.backend = StandaloneCameraBackend(scene, context, rasterizer=renderer)

        self._shared_metadata.sensors.append(self)

        if self._manager._sim.n_envs > 1 and not self._shared_metadata.context.env_separate_rigid:
            gs.raise_exception(
                "RasterizerCameraSensor with n_envs > 1 requires 'env_separate_rigid=True' in VisOptions "
                "for correct per-environment rendering."
            )

        # Build the owned vis.Camera now if the rasterizer is standalone (already built here), or defer to first render
        # when sharing the visualizer's rasterizer (the visualizer isn't built yet at sensor.build() time).
        if self._shared_metadata.renderer.offscreen:
            self._build_camera()

    def _build_camera(self):
        """Construct + build this sensor's `vis.Camera` against the rasterizer backend (idempotent)."""
        if self._camera is not None:
            return

        # Add this camera's lights to the shared context.
        for light_config in self._options.lights:
            self._shared_metadata.context.add_light(self._convert_light_config_to_rasterizer(light_config))

        self._camera = Camera(
            self._shared_metadata.backend,
            idx=self._idx,
            model="pinhole",
            res=self._options.res,
            pos=self._options.pos,
            lookat=self._options.lookat,
            up=self._options.up,
            fov=self._options.fov,
            near=self._options.near,
            far=self._options.far,
            GUI=False,
            debug=False,
        )
        self._camera.build()
        self._init_camera_pose()

    def _create_standalone_context(self, scene):
        """Create a simplified RasterizerContext for camera sensors."""
        if not scene.sim._rigid_only and scene.n_envs > 1:
            gs.raise_exception("Rasterizer with n_envs > 1, does not work when using non rigid simulation")
        if scene.n_envs > 1:
            gs.logger.warning(
                "Rasterizer with n_envs > 1 is slow as it doesn't do batched rendering consider using BatchRenderer instead."
            )
        env_separate_rigid = True
        vis_options = VisOptions(
            show_world_frame=False,
            show_link_frame=False,
            show_cameras=False,
            rendered_envs_idx=range(max(self._manager._sim._B, 1)),
            env_separate_rigid=env_separate_rigid,
        )

        context = RasterizerContext(vis_options)
        context.build(scene)
        context.reset()
        return context

    @staticmethod
    def _convert_light_config_to_rasterizer(light_config):
        """Convert a light config dict to a typed light options object for the rasterizer."""
        from genesis.options.vis import DirectionalLight, PointLight

        light_type = light_config.get("type", "directional")
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)

        if light_type == "point":
            pos = light_config.get("pos", (0.0, 0.0, 5.0))
            return PointLight(pos=pos, color=color, intensity=intensity)
        else:
            dir = light_config.get("dir", (0.0, 0.0, -1.0))
            return DirectionalLight(dir=dir, color=color, intensity=intensity)

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        self._build_camera()  # deferred build on the viewer-shared path
        # Track the attached link; static cameras keep their build-time world pose. `set_pose` inside `move_to_attach`
        # refreshes the rasterizer node pose.
        if self._link is not None:
            self._camera.move_to_attach()

        context = self._shared_metadata.context

        context.update(force_render=True)

        # When env_separate_rigid is enabled, geometry render transforms include env_spacing offsets (baked in by
        # kernel_update_geoms_render_T). For per-env sensor rendering, these offsets must be temporarily removed so each
        # env's geometry renders at local origin relative to the camera. The offsets are restored after rendering to
        # preserve the correct layout for the interactive viewer which shares the same context.
        envs_offset = context.scene.envs_offset
        saved_poses = {}
        if context.env_separate_rigid and (envs_offset != 0).any():
            for node_uid, node in context.rigid_nodes.items():
                poses = node.mesh.primitives[0].poses
                if poses is not None and len(poses) > 1:
                    saved_poses[node_uid] = poses.copy()
                    poses[:, :3, 3] -= envs_offset[context.rendered_envs_idx]
                    context.jit.update_buffer(node, "model", poses.transpose((0, 2, 1)))

        render_out = self._shared_metadata.renderer.render_camera(
            self._camera,
            rgb=self._options.render_rgb,
            depth=self._options.render_depth,
            segmentation=self._options.render_segmentation,
            normal=self._options.render_normal,
        )

        # Restore original geometry transforms with offsets for the interactive viewer
        for node_uid, poses in saved_poses.items():
            node = context.rigid_nodes[node_uid]
            node.mesh.primitives[0].poses = poses
            context.jit.update_buffer(node, "model", poses.transpose((0, 2, 1)))

        # Scatter each requested modality into this sensor's columns of the shared manager cache.
        self._scatter_render_into_manager_cache(render_out)


# ========================== Raytracer Camera Sensor ==========================
class RaytracerCameraSensor(
    BaseCameraSensor, Sensor[RaytracerCameraOptions, None, RaytracerCameraSharedMetadata, CameraReturnType]
):
    """
    Raytracer camera sensor using LuisaRender path tracing.
    """

    def __init__(
        self,
        options: RaytracerCameraOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._options: RaytracerCameraOptions

    def build(self):
        """Register a raytracer camera that reuses the visualizer pipeline."""
        super().build()

        scene = self._manager._sim.scene
        visualizer = scene.visualizer

        renderer = getattr(visualizer, "raytracer", None)
        if renderer is None:
            gs.raise_exception(
                "RaytracerCameraSensor requires the scene to be created with `renderer=gs.renderers.RayTracer(...)`."
            )

        if self._options.render_depth or self._options.render_segmentation or self._options.render_normal:
            gs.logger.warning(
                "Raytracer camera sensor: only RGB is path-traced. `depth`, `segmentation` and `normal` are produced "
                "by the rasterizer fallback (geometry-correct, but not path-traced) and require an OpenGL context."
            )

        # Multi-environment rendering is not yet supported for Raytracer cameras
        n_envs = self._manager._sim.n_envs
        if n_envs > 1:
            gs.raise_exception(
                f"Raytracer camera sensors do not support multi-environment rendering (n_envs={n_envs}). "
                "Use BatchRenderer camera sensors for batched rendering."
            )

        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = []
            self._shared_metadata.renderer = renderer

        self._shared_metadata.sensors.append(self)

        # Add lights from options as mesh lights to the scene
        scene = self._manager._sim.scene
        for light_config in self._options.lights:
            if not scene.is_built:
                self._add_light_as_mesh_light(scene, light_config)

        # Own a real vis.Camera built against the visualizer (the raytracer requires one). The initial pose uses the
        # configured pos/lookat/up in world frame; when attached, `_init_camera_pose` binds the link and the per-step
        # `move_to_attach` corrects the pose.
        self._camera = visualizer.add_camera(
            res=self._options.res,
            pos=self._options.pos,
            lookat=self._options.lookat,
            up=self._options.up,
            model=self._options.model,
            fov=self._options.fov,
            aperture=self._options.aperture,
            focus_dist=self._options.focus_dist,
            GUI=False,
            spp=self._options.spp,
            denoise=self._options.denoise,
            near=0.05,
            far=100.0,
            env_idx=None if n_envs == 0 else 0,
            debug=False,
        )
        self._init_camera_pose()

    def _add_light_as_mesh_light(self, scene, light_config):
        """Add a light as a mesh light to the scene."""
        # Default values for raytracer mesh lights
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)
        radius = light_config.get("radius", 0.5)
        pos = light_config.get("pos", (0.0, 0.0, 5.0))
        revert_dir = light_config.get("revert_dir", False)
        double_sided = light_config.get("double_sided", False)
        cutoff = light_config.get("cutoff", 180.0)

        morph = gs.morphs.Sphere(pos=pos, radius=radius)
        scene.add_mesh_light(
            morph=morph,
            color=(*color, 1.0),
            intensity=intensity,
            revert_dir=revert_dir,
            double_sided=double_sided,
            cutoff=cutoff,
        )

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        if self._link is not None:
            self._camera.move_to_attach()

        # Only RGB is path-traced; depth/segmentation/normal come from the rasterizer fallback inside `Camera.render`
        # (see `RaytracerCameraSensor.build` for the warning). `colorize_seg=False` yields the int32 index map, which is
        # the desired numeric sensor output.
        render_out = self._camera.render(
            rgb=self._options.render_rgb,
            depth=self._options.render_depth,
            segmentation=self._options.render_segmentation,
            colorize_seg=False,
            normal=self._options.render_normal,
            antialiasing=False,
            force_render=True,
        )
        # Raytracer camera sensors reject n_envs > 1, so the render is a single (non-batched) frame; the scatter helper
        # adds the missing batch dim.
        self._scatter_render_into_manager_cache(render_out)


# ========================== Batch Renderer Camera Sensor ==========================


class BatchRendererCameraSensor(
    BaseCameraSensor, Sensor[BatchRendererCameraOptions, None, BatchRendererCameraSharedMetadata, CameraReturnType]
):
    """
    Batch renderer camera sensor using Madrona GPU batch rendering.

    Note: All batch renderer cameras must have the same resolution.
    """

    def __init__(
        self,
        options: BatchRendererCameraOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._options: BatchRendererCameraOptions

    def build(self):
        """Initialize the batch renderer and register this camera."""
        super().build()

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRendererCameraSensor requires CUDA backend.")

        scene = self._manager._sim.scene

        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = gs.List()
            self._shared_metadata.last_render_timestep = -1

            all_sensors = self._manager._sensors_by_type[type(self)]
            resolutions = [s._options.res for s in all_sensors]
            if len(set(resolutions)) > 1:
                gs.raise_exception(
                    f"All BatchRendererCameraSensor instances must have the same resolution. Found: {set(resolutions)}"
                )

            br_options = BatchRendererOptions(use_rasterizer=self._options.use_rasterizer)

            vis_options = VisOptions(
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
                rendered_envs_idx=range(max(self._manager._sim._B, 1)),
            )

            # Standalone backend the owned vis.Cameras build against and which the batch renderer enumerates. The
            # renderer must exist before the cameras build so `Camera.build` takes the batch branch.
            context = RasterizerContext(vis_options)
            context.build(scene)
            context.reset()
            backend = StandaloneCameraBackend(scene, context)
            backend.batch_renderer = BatchRenderer(backend, br_options, vis_options)
            self._shared_metadata.backend = backend
            self._shared_metadata.renderer = backend.batch_renderer

        self._shared_metadata.sensors.append(self)

        # Add lights from options to the renderer
        for light_config in self._options.lights:
            if self._shared_metadata.renderer is not None:
                self._add_light_to_batch_renderer(light_config)

        # This sensor's vis.Camera is a pose/intrinsics holder the batch renderer enumerates (it does not drive
        # per-sensor rendering; the single union pass in `_render_current_state` does).
        self._camera = Camera(
            self._shared_metadata.backend,
            idx=self._idx,
            model=self._options.model,
            res=self._options.res,
            pos=self._options.pos,
            lookat=self._options.lookat,
            up=self._options.up,
            fov=self._options.fov,
            near=self._options.near,
            far=self._options.far,
            GUI=False,
            debug=False,
        )
        self._camera.build()
        self._init_camera_pose()

        if len(self._shared_metadata.sensors) == len(self._manager._sensors_by_type[type(self)]):
            self._shared_metadata.backend._cameras = [s._camera for s in self._shared_metadata.sensors]
            self._shared_metadata.renderer.build()

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        sensors = self._shared_metadata.sensors or [self]

        for sensor in sensors:
            if sensor._link is not None:
                sensor._camera.move_to_attach()

        self._shared_metadata.renderer.update_scene(force_render=True)

        # The batch renderer renders ALL cameras in a single pass with one set of flags, so request the union of the
        # modalities across all batch camera sensors, then write only each sensor's own requested buffers below.
        union = {name: any(getattr(s._options, f"render_{name}") for s in sensors) for name in _CAMERA_MODALITIES}
        render_out = self._shared_metadata.renderer.render(
            rgb=union["rgb"],
            depth=union["depth"],
            segmentation=union["segmentation"],
            normal=union["normal"],
            antialiasing=False,
            force_render=True,
        )

        # render_out is `(rgb, depth, segmentation, normal)`, each either None (not requested) or a per-camera sequence
        # of arrays (one entry per camera in `sensors`). Regroup into a per-sensor render tuple and scatter.
        per_camera = []
        for arrs in render_out:
            if arrs is None:
                per_camera.append([None] * len(sensors))
            elif isinstance(arrs, (tuple, list)):
                per_camera.append(list(arrs))
            else:
                per_camera.append([arrs])
        for cam_i, sensor in enumerate(sensors):
            sensor_render = tuple(per_camera[mod_i][cam_i] for mod_i in range(len(_CAMERA_MODALITIES)))
            sensor._scatter_render_into_manager_cache(sensor_render)
            sensor._stale = False

        self._shared_metadata.last_render_timestep = self._manager._sim.scene.t

    def _add_light_to_batch_renderer(self, light_config):
        """Add a light to the batch renderer."""
        # Default values for batch renderer
        pos = light_config.get("pos", (0.0, 0.0, 5.0))
        dir = light_config.get("dir", (0.0, 0.0, -1.0))
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)
        directional = light_config.get("directional", True)
        castshadow = light_config.get("castshadow", True)
        cutoff = light_config.get("cutoff", 45.0)
        attenuation = light_config.get("attenuation", (1.0, 0.0, 0.0))

        self._shared_metadata.renderer.add_light(
            pos=pos,
            dir=dir,
            color=color,
            intensity=intensity,
            directional=directional,
            castshadow=castshadow,
            cutoff=cutoff,
            attenuation=attenuation,
        )
