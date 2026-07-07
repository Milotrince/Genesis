import importlib
import pkgutil
import sys
from typing import TYPE_CHECKING, ForwardRef, get_args, get_origin

import torch

import genesis as gs
from genesis.options.sensors import types as _sensor_types_namespace
from genesis.options.sensors.options import SensorOptions
from genesis.utils.ring_buffer import TensorRingBuffer

from .base_sensor import Sensor, SharedSensorContext, SharedSensorMetadata

if TYPE_CHECKING:
    from genesis.vis.rasterizer_context import RasterizerContext


class SensorManager:
    # Maps sensor options class -> sensor class for runtime dispatch.
    SENSOR_TYPES_MAP: dict[type[SensorOptions], type["Sensor"]] = {}

    def __init__(self, sim):
        self._sim = sim
        self._sensors_by_type: dict[type["Sensor"], list["Sensor"]] = {}
        self._sensors_metadata: dict[type["Sensor"], SharedSensorMetadata | None] = {}
        # Cross-type shared contexts, keyed by context class so every sensor type declaring the same context resolves
        # to one instance. Built/updated/reset/destroyed by this manager; see ``SharedSensorContext``.
        self._shared_contexts: dict[type, SharedSensorContext] = {}
        # Per-dtype intermediate caches: pre-`_post_process` storage in intermediate space. The transposed GT cache is
        # `(cols, B)` for C-contiguous per-class row slices required by kernel writes.
        self._ground_truth_intermediate_cache: dict[type[torch.dtype], torch.Tensor] = {}
        self._intermediate_cache: dict[type[torch.dtype], torch.Tensor] = {}
        # Per-class return caches in return space - what `read()` and `read_ground_truth()` slice into, keyed by return
        # dtype (a multi-dtype sensor class, e.g. a camera, has one cache per dtype it spans). Separate buffers when a
        # per-class return-space ring is allocated (the orchestrator delay-samples the ring into the cache); alias-views
        # into the per-dtype intermediate cache otherwise (identity `_post_process`, no delay, no history - the per-step
        # write inside `_update_shared_cache` is then directly visible to `read()`).
        self._return_cache: dict[type["Sensor"], dict[torch.dtype, torch.Tensor]] = {}
        self._ground_truth_return_cache: dict[type["Sensor"], dict[torch.dtype, torch.Tensor]] = {}
        # Paired GT and measured timeline rings (post-transform, PRE-hardware-imperfections data). Allocated together
        # per dtype when any sensor in the dtype declares `uses_ring_pipeline = True`. They share the same rotation idx
        # so a single `rotate()` per step advances both.
        self._ground_truth_timeline_ring: dict[type[torch.dtype], TensorRingBuffer] = {}
        self._measured_timeline_ring: dict[type[torch.dtype], TensorRingBuffer] = {}
        # Per-(class, return-dtype) return-space rings (post-everything: post-hardware-imperfections, post-
        # `_post_process`, pre-delay-sample). Allocated when any sensor in the class has `delay > 0`, OR
        # `history_length > 0`, OR the class overrides `_post_process`. Each step the post-everything snapshot is
        # written to slot 0; the source for delay sampling (into the per-class return cache) and for history reads. GT
        # and measured rings share their rotation idx so a single `rotate()` per step advances both.
        self._ground_truth_return_timeline_ring: dict[type["Sensor"], dict[torch.dtype, TensorRingBuffer]] = {}
        self._measured_return_timeline_ring: dict[type["Sensor"], dict[torch.dtype, TensorRingBuffer]] = {}
        # Per-class precomputed history index tensor [0, 1, ..., max_history-1]. Used to fancy-index the rings on
        # history reads.
        self._hist_idx_by_class: dict[type["Sensor"], torch.Tensor] = {}
        # Per-class contiguous slice into each INTERMEDIATE dtype buffer the class spans.
        self._cache_slices_by_type: dict[type["Sensor"], dict[torch.dtype, slice]] = {}
        # Per (sensor class, entity_idx) -> slice within the class return cache (relative). entity_idx == -1 means
        # static sensors. Only meaningful/used for single-dtype ring-pipeline classes (the ones in `read_sensors()`).
        self._entity_slice_in_class: dict[type["Sensor"], dict[int, slice]] = {}
        self._max_history_by_class: dict[type["Sensor"], int] = {}
        # Per-(class, return-dtype) ordered `(global_sensor_idx, size_in_dtype)` list driving `_apply_delay`.
        self._delay_layout_by_type: dict[type["Sensor"], dict[torch.dtype, list[tuple[int, int]]]] = {}

    def create_sensor(self, sensor_options: "SensorOptions") -> "Sensor":
        sensor_options.validate_scene(self._sim.scene)
        sensor_cls = SensorManager._resolve_sensor_cls(type(sensor_options))
        self._sensors_by_type.setdefault(sensor_cls, [])
        if sensor_cls not in self._sensors_metadata:
            self._sensors_metadata[sensor_cls] = sensor_cls._metadata_cls()
        # Create the shared context before the sensor, so the instance exists to hand to it. ``NoneType`` marks
        # "no context"; the sensor then receives ``None``.
        context_cls = sensor_cls._shared_context_cls
        if context_cls is not type(None) and context_cls not in self._shared_contexts:
            self._shared_contexts[context_cls] = context_cls(self._sim)
        sensor = sensor_cls(
            sensor_options,
            len(self._sensors_by_type[sensor_cls]),
            self._shared_contexts.get(context_cls),
            self._sensors_metadata[sensor_cls],
            self,
        )
        self._sensors_by_type[sensor_cls].append(sensor)
        return sensor

    @staticmethod
    def _resolve_sensor_cls(options_cls: type) -> type["Sensor"]:
        """Resolve the sensor class for the given options class, triggering lazy discovery if needed."""
        sensor_cls = SensorManager.SENSOR_TYPES_MAP.get(options_cls)
        if sensor_cls is not None:
            return sensor_cls

        # Not registered yet — check that the options class specifies its sensor type, then try to discover it. The
        # sensor class name is extracted from the generic metadata on the options class bases.
        is_parameterized = False
        for base in options_cls.__bases__:
            meta = base.__pydantic_generic_metadata__
            if meta["origin"] is not None and issubclass(meta["origin"], SensorOptions):
                is_parameterized = bool(meta["args"]) and isinstance(meta["args"][0], str)
                break
        # Fallback: typing introspection on __orig_bases__ (for pydantic versions that flatten bases)
        if not is_parameterized:
            for base in options_cls.__orig_bases__:
                origin = get_origin(base)
                if origin is not None and issubclass(origin, SensorOptions):
                    args = get_args(base)
                    is_parameterized = bool(args) and isinstance(args[0], (str, ForwardRef))
                    break

        if not is_parameterized:
            gs.raise_exception(
                f"{options_cls.__name__} must parameterize its SensorOptions base with a sensor class, "
                f"e.g. `class {options_cls.__name__}(SensorOptions['MySensor']): ...`"
            )

        # Try to discover the sensor module from sibling modules of the options package.
        options_module = options_cls.__module__
        if "." in options_module:
            pkg_name = options_module.rsplit(".", 1)[0]
            pkg = sys.modules.get(pkg_name)
            if pkg is not None:
                pkg_path = pkg.__dict__.get("__path__")
                if pkg_path is not None:
                    for _, modname, _ in pkgutil.iter_modules(pkg_path, pkg.__name__ + "."):
                        if modname not in sys.modules:
                            try:
                                importlib.import_module(modname)
                            except Exception:
                                continue
                        if options_cls in SensorManager.SENSOR_TYPES_MAP:
                            return SensorManager.SENSOR_TYPES_MAP[options_cls]

        gs.raise_exception(
            f"No sensor class registered for {options_cls.__name__}. Ensure the sensor module is in the same "
            "package as the options module, or import the sensor class manually before calling add_sensor()."
        )

    def build(self):
        # Sort each class by entity_idx so sensors attached to the same entity occupy a contiguous slice of the class
        # cache. Static sensors have entity_idx=-1 and group together. Python's sort is stable, so registration order is
        # preserved within each entity bucket.
        for sensors in self._sensors_by_type.values():
            sensors.sort(key=lambda s: s._options.entity_idx)
            for new_idx, sensor in enumerate(sensors):
                sensor._idx = new_idx

        # Field dtypes come from `_get_intermediate_dtype` / `_get_cache_dtype` (single dtype, or one per field). A
        # sensor's fields are grouped by dtype (`sensor._intermediate_size_by_dtype` / `_return_size_by_dtype`), so a
        # multi-dtype sensor (a camera: uint8 rgb / float32 depth+normal / int32 seg) contributes a contiguous column
        # run to EACH dtype's buffer. Each class occupies one contiguous slice per dtype it spans; because a class's
        # sensors are processed consecutively, that per-dtype slice stays contiguous.
        #
        # Ring-pipeline classes are laid out FIRST within each dtype so their columns form a contiguous prefix
        # [0, timeline_size_per_dtype); the `_apply_transform` timeline rings then cover only that prefix and never the
        # (potentially huge) image columns of non-ring camera classes. `timeline_size_per_dtype` is snapshotted at the
        # ring/non-ring boundary.
        cache_size_per_dtype: dict[torch.dtype, int] = {}
        max_history_per_dtype: dict[torch.dtype, int] = {}
        timeline_size_per_dtype: dict[torch.dtype, int] | None = None
        # Per-class delay-depth (max sensor `_delay_ts + 1`) drives the return-space ring sizing for delay sampling.
        delay_depth_by_class: dict[type["Sensor"], int] = {}
        ordered_classes = sorted(self._sensors_by_type.items(), key=lambda kv: not kv[0].uses_ring_pipeline)
        for sensor_cls, sensors in ordered_classes:
            if timeline_size_per_dtype is None and not sensor_cls.uses_ring_pipeline:
                # First non-ring class: freeze the timeline-covered prefix width per dtype.
                timeline_size_per_dtype = dict(cache_size_per_dtype)
            cls_int_start: dict[torch.dtype, int] = {}  # per intermediate dtype: this class's start column (absolute)
            entity_offsets: dict[int, list[int]] = {}
            cls_offset = 0
            cls_max_history = 0
            cls_delay_depth = 1
            for sensor in sensors:
                sensor._cache_idx_by_dtype = {}
                for dt, sz in sensor._intermediate_size_by_dtype.items():
                    start = cache_size_per_dtype.get(dt, 0)
                    cls_int_start.setdefault(dt, start)
                    sensor._cache_idx_by_dtype[dt] = start
                    cache_size_per_dtype[dt] = start + sz
                    max_history_per_dtype.setdefault(dt, 0)
                cls_delay_depth = max(cls_delay_depth, sensor._delay_ts + 1)
                hist = sensor._options.history_length
                if hist > 0:
                    cls_max_history = max(cls_max_history, hist)
                    if sensor_cls.uses_ring_pipeline:
                        # Only ring-pipeline history grows the transform timeline ring; camera history is served by the
                        # per-class return-space ring instead.
                        for dt in sensor._intermediate_size_by_dtype:
                            max_history_per_dtype[dt] = max(max_history_per_dtype[dt], hist)
                eid = sensor._options.entity_idx
                if eid in entity_offsets:
                    entity_offsets[eid][1] = cls_offset + sensor._cache_size
                else:
                    entity_offsets[eid] = [cls_offset, cls_offset + sensor._cache_size]
                cls_offset += sensor._cache_size

            self._cache_slices_by_type[sensor_cls] = {
                dt: slice(cls_int_start[dt], cache_size_per_dtype[dt]) for dt in cls_int_start
            }
            self._entity_slice_in_class[sensor_cls] = {
                eid: slice(start, stop) for eid, (start, stop) in entity_offsets.items()
            }
            self._max_history_by_class[sensor_cls] = cls_max_history
            delay_depth_by_class[sensor_cls] = cls_delay_depth
        if timeline_size_per_dtype is None:  # all classes are ring-pipeline
            timeline_size_per_dtype = dict(cache_size_per_dtype)

        self._ground_truth_timeline_ring.clear()
        self._measured_timeline_ring.clear()
        self._return_cache.clear()
        self._ground_truth_return_cache.clear()
        self._ground_truth_return_timeline_ring.clear()
        self._measured_return_timeline_ring.clear()
        self._hist_idx_by_class.clear()

        for dtype, total_cols in cache_size_per_dtype.items():
            cache_shape = (self._sim._B, total_cols)
            # Ground truth cache is stored transposed (cols, B) so that per-class row slices are C-contiguous, which is
            # required for kernel writes. The cache and ring buffer stay (B, cols) since they only receive data via
            # .copy_() / torch.lerp which handle non-contiguous targets.
            gt_cache_shape = (total_cols, self._sim._B)
            self._ground_truth_intermediate_cache[dtype] = torch.zeros(gt_cache_shape, dtype=dtype, device=gs.device)
            self._intermediate_cache[dtype] = torch.zeros(cache_shape, dtype=dtype, device=gs.device)
            timeline_cols = timeline_size_per_dtype.get(dtype, 0)
            if timeline_cols > 0:
                # Timeline rings serve `_apply_transform` recurrence and cover only the ring-pipeline column prefix.
                # Two slots cover the canonical one-step recurrence; the ring is grown to `max_history` when any
                # ring-pipeline sensor requests history so a multi-tap stateful filter can read deeper without state.
                timeline_shape = (self._sim._B, timeline_cols)
                ring_n = max(2, max_history_per_dtype.get(dtype, 0))
                self._measured_timeline_ring[dtype] = TensorRingBuffer(ring_n, timeline_shape, dtype=dtype)
                self._ground_truth_timeline_ring[dtype] = TensorRingBuffer(
                    ring_n, timeline_shape, dtype=dtype, idx=self._measured_timeline_ring[dtype]._idx
                )

        # Per-(class, return-dtype) return-space caches + rings. The return-space ring records each step's post-
        # `_post_process` snapshot; it is the source for delay sampling and history reads, and provides the `timeline`
        # argument that stateful `_post_process` overrides see. Allocated whenever any sensor in the class has delay > 0,
        # OR history > 0, OR the class overrides `_post_process`. When no ring is needed (no delay, no history, identity
        # `_post_process`), the return cache is a zero-copy alias-view of the intermediate cache so per-step writes
        # propagate without extra work. A multi-dtype class (a camera) gets one entry per dtype in each dict.
        for sensor_cls, sensors in self._sensors_by_type.items():
            cls_max_history = self._max_history_by_class[sensor_cls]
            cls_delay_depth = delay_depth_by_class[sensor_cls]
            pp_overridden = sensor_cls._post_process.__func__ is not Sensor._post_process.__func__
            needs_ring = cls_delay_depth > 1 or cls_max_history > 0 or pp_overridden

            # Per return dtype: total class columns, and per-sensor relative start + delay layout. Return grouping
            # mirrors the intermediate grouping in structure; they differ only in dtype label (e.g. ContactSensor:
            # float intermediate -> bool return).
            cls_ret_size: dict[torch.dtype, int] = {}
            delay_layout: dict[torch.dtype, list[tuple[int, int]]] = {}
            for sensor in sensors:
                sensor._return_idx_by_dtype = {}
                for dt, sz in sensor._return_size_by_dtype.items():
                    start = cls_ret_size.get(dt, 0)
                    sensor._return_idx_by_dtype[dt] = start
                    cls_ret_size[dt] = start + sz
                    delay_layout.setdefault(dt, []).append((sensor._idx, sz))
            self._delay_layout_by_type[sensor_cls] = delay_layout

            self._return_cache[sensor_cls] = {}
            self._ground_truth_return_cache[sensor_cls] = {}
            if needs_ring:
                ring_n = max(cls_delay_depth, cls_max_history, 2 if pp_overridden else 1)
                self._ground_truth_return_timeline_ring[sensor_cls] = {}
                self._measured_return_timeline_ring[sensor_cls] = {}
                for dt, size in cls_ret_size.items():
                    ring_shape = (self._sim._B, size)
                    gt_ring = TensorRingBuffer(ring_n, ring_shape, dtype=dt)
                    m_ring = TensorRingBuffer(ring_n, ring_shape, dtype=dt, idx=gt_ring._idx)
                    self._ground_truth_return_timeline_ring[sensor_cls][dt] = gt_ring
                    self._measured_return_timeline_ring[sensor_cls][dt] = m_ring
                    self._return_cache[sensor_cls][dt] = torch.zeros((self._sim._B, size), dtype=dt, device=gs.device)
                    self._ground_truth_return_cache[sensor_cls][dt] = torch.zeros(
                        (self._sim._B, size), dtype=dt, device=gs.device
                    )
            else:
                # Identity `_post_process` guarantees return dtype == intermediate dtype per field, so each return-dtype
                # block aliases this class's slice of that dtype's intermediate buffer (same relative layout).
                for dt in cls_ret_size:
                    int_slice = self._cache_slices_by_type[sensor_cls][dt]
                    self._return_cache[sensor_cls][dt] = self._intermediate_cache[dt][:, int_slice]
                    self._ground_truth_return_cache[sensor_cls][dt] = self._ground_truth_intermediate_cache[dt][
                        int_slice, :
                    ].T
            if cls_max_history > 0:
                self._hist_idx_by_class[sensor_cls] = torch.arange(cls_max_history, device=gs.device, dtype=torch.int32)

        for sensor_cls, sensors in self._sensors_by_type.items():
            for sensor in sensors:
                sensor.build()
                sensor._is_built = True

    def destroy(self):
        for context in self._shared_contexts.values():
            context.destroy()
        self._shared_contexts.clear()
        for sensors_metadata in self._sensors_metadata.values():
            if sensors_metadata is not None:
                sensors_metadata.destroy()
        self._sensors_metadata.clear()
        self._sensors_by_type.clear()

    def reset(self, envs_idx=None):
        if not self._sensors_by_type:
            return

        envs_idx = self._sim._scene._sanitize_envs_idx(envs_idx)

        for dtype in self._ground_truth_intermediate_cache.keys():
            self._ground_truth_intermediate_cache[dtype][:, envs_idx] = 0.0
            self._intermediate_cache[dtype][envs_idx] = 0.0
            if dtype in self._ground_truth_timeline_ring:
                self._ground_truth_timeline_ring[dtype].buffer[:, envs_idx] = 0.0
            if dtype in self._measured_timeline_ring:
                self._measured_timeline_ring[dtype].buffer[:, envs_idx] = 0.0

        # Reset per-(class, dtype) return caches. When the return cache is an alias-view of the intermediate cache the
        # clear is redundant (the intermediate clear above already wrote zeros to the same memory) but harmless.
        # Return-space rings are always distinct buffers.
        for per_dtype in self._return_cache.values():
            for cache in per_dtype.values():
                cache[envs_idx] = 0
        for per_dtype in self._ground_truth_return_cache.values():
            for cache in per_dtype.values():
                cache[envs_idx] = 0
        for per_dtype in self._ground_truth_return_timeline_ring.values():
            for ring in per_dtype.values():
                ring.buffer[:, envs_idx] = 0
        for per_dtype in self._measured_return_timeline_ring.values():
            for ring in per_dtype.values():
                ring.buffer[:, envs_idx] = 0

        # Reset shared contexts before the per-type sensor reset (a reset may change otherwise-static geometry, so the
        # context must rebuild before any sensor reads it again).
        for context in self._shared_contexts.values():
            context.reset(envs_idx)

        for sensor_cls, sensors in self._sensors_by_type.items():
            # `reset` receives the class's GT intermediate slice(s). Single-dtype classes (the only ones whose `reset`
            # reads it, e.g. TemperatureGrid) get the sole slice tensor unchanged; multi-dtype classes (cameras, whose
            # `reset` ignores it) get the per-dtype dict.
            gt_by_dtype = {
                dt: self._ground_truth_intermediate_cache[dt][sl]
                for dt, sl in self._cache_slices_by_type[sensor_cls].items()
            }
            gt_arg = next(iter(gt_by_dtype.values())) if len(gt_by_dtype) == 1 else gt_by_dtype
            sensor_cls.reset(self._sensors_metadata[sensor_cls], gt_arg, envs_idx)

    def step(self):
        # Timeline rings must rotate before `_update_shared_cache` because `_apply_transform` mutates `at(0)` of the
        # timeline ring and needs a fresh write slot. Return-space rings, by contrast, are read during `_post_process`
        # (past post-output values) and written afterward; their rotation is deferred to inside the per-class loop so
        # `at(0)` during `_post_process` is the previous step's post-output (a meaningful "previous value") rather than
        # stale data from the slot about to be overwritten.
        for ring in self._measured_timeline_ring.values():
            ring.rotate()

        # Refresh each shared context once per step, before the per-type loop reads it, so multiple consuming sensor
        # types (e.g. Raycaster + DepthCamera) rebuild the shared resource at most once rather than once each.
        for context in self._shared_contexts.values():
            context.update()

        for sensor_cls, sensors in self._sensors_by_type.items():
            # Per-dtype views of this class's slice of each buffer it spans. A single-dtype class has one entry per
            # dict (exactly the old scalars); a multi-dtype class (a camera) has one per dtype.
            cls_int_slices = self._cache_slices_by_type[sensor_cls]
            ground_truth_slices = {
                dt: self._ground_truth_intermediate_cache[dt][sl] for dt, sl in cls_int_slices.items()
            }
            intermediates = {dt: self._intermediate_cache[dt][:, sl] for dt, sl in cls_int_slices.items()}
            # Timeline rings exist only for ring-pipeline classes and cover only their column prefix, so non-ring
            # classes (cameras) always get `None` here even when they share a dtype with a ring-pipeline sensor.
            uses_rings = sensor_cls.uses_ring_pipeline
            ground_truth_data_timelines = {
                dt: (
                    self._ground_truth_timeline_ring[dt][:, sl]
                    if uses_rings and dt in self._ground_truth_timeline_ring
                    else None
                )
                for dt, sl in cls_int_slices.items()
            }
            measured_data_timelines = {
                dt: (
                    self._measured_timeline_ring[dt][:, sl]
                    if uses_rings and dt in self._measured_timeline_ring
                    else None
                )
                for dt, sl in cls_int_slices.items()
            }
            metadata = self._sensors_metadata[sensor_cls]
            sensor_cls._update_shared_cache(
                self._shared_contexts.get(sensor_cls._shared_context_cls),
                metadata,
                ground_truth_slices,
                ground_truth_data_timelines,
                measured_data_timelines,
                intermediates,
            )

            gt_return_rings = self._ground_truth_return_timeline_ring.get(sensor_cls)
            if gt_return_rings is None:
                # No return-space ring: identity `_post_process`, no delay, no history. Return cache aliases
                # intermediate (the per-step write inside `_update_shared_cache` is already visible to `read()`).
                continue
            measured_return_rings = self._measured_return_timeline_ring[sensor_cls]
            delay_layout = self._delay_layout_by_type[sensor_cls]
            pp_overridden = sensor_cls._post_process.__func__ is not Sensor._post_process.__func__

            for return_dtype, measured_return_ring in measured_return_rings.items():
                gt_return_ring = gt_return_rings[return_dtype]
                if pp_overridden:
                    # `_post_process` overrides (e.g. ContactSensor float->bool) exist only on single-dtype classes, so
                    # the sole intermediate slice is this class's whole working buffer; it may differ in dtype from the
                    # return space. The ring has not yet rotated, so `timeline.at(0)` is the previous step's post-output.
                    (intermediate,) = intermediates.values()
                    (ground_truth_slice,) = ground_truth_slices.values()
                    measured_projected = sensor_cls._post_process(
                        metadata, intermediate, measured_return_ring, is_measured=True
                    )
                    gt_projected = sensor_cls._post_process(
                        metadata, ground_truth_slice.T, gt_return_ring, is_measured=False
                    )
                else:
                    # Identity projection: return dtype == intermediate dtype, so this dtype's intermediate slice is the
                    # post-output directly (delay/history for cameras and other non-transforming sensors).
                    measured_projected = intermediates[return_dtype]
                    gt_projected = ground_truth_slices[return_dtype].T

                # Rotate now, after `_post_process` reads finished and before writing this step's projections into slot
                # 0. Only one rotate per pair since GT and measured return rings share idx.
                gt_return_ring.rotate()
                measured_return_ring.set(measured_projected)
                gt_return_ring.set(gt_projected)

                # GT has no readout delay (delay is a measured-only effect), so the GT read is just the current slot.
                self._ground_truth_return_cache[sensor_cls][return_dtype].copy_(gt_return_ring.at(0, copy=False))
                # Measured: per-sensor delay + jitter sampling from the return-space ring into the per-class return
                # cache. Default ZOH `_apply_delay` is dtype-safe for any return space (bool, uint8, quantized float).
                sensor_cls._apply_delay(
                    metadata,
                    measured_return_ring,
                    self._return_cache[sensor_cls][return_dtype],
                    delay_layout[return_dtype],
                )

    def draw_debug(self, context: "RasterizerContext"):
        for sensor in self.sensors:
            if sensor._options.draw_debug:
                sensor._draw_debug(context)

    def get_cloned_from_cache(self, sensor: "Sensor", is_ground_truth: bool = False) -> list[torch.Tensor]:
        """
        Return this sensor's data as a per-field list of ``(B, span)`` tensors (``span`` folds in history when enabled).

        One tensor per return field, in field order; a multi-dtype sensor (a camera) reads each field from its own
        dtype's return cache. `_get_formatted_data` reshapes and packs these into the sensor's return type. Cannot be a
        single concatenated tensor because fields may differ in dtype.
        """
        sensor_cls = type(sensor)
        history_length = sensor._options.history_length
        return_caches = (
            self._ground_truth_return_cache[sensor_cls] if is_ground_truth else self._return_cache[sensor_cls]
        )
        hist_by_dtype: dict[torch.dtype, torch.Tensor] = {}

        blocks: list[torch.Tensor] = []
        for i, return_dtype in enumerate(sensor._field_dtypes):
            rel_start = sensor._return_idx_by_dtype[return_dtype]
            field_slice = sensor._field_return_slice[i]
            col = slice(rel_start + field_slice.start, rel_start + field_slice.stop)
            if history_length > 0:
                if return_dtype not in hist_by_dtype:
                    hist_by_dtype[return_dtype] = self._gather_history(
                        sensor_cls, return_dtype, history_length, is_ground_truth
                    )
                blocks.append(hist_by_dtype[return_dtype][:, :, col].flatten(1, 2))
            else:
                # Pure view into the per-class return cache. Eager `_post_process` already populated it during step().
                blocks.append(return_caches[return_dtype][:, col])
        return blocks

    def _gather_history(
        self, sensor_cls: type["Sensor"], dtype: torch.dtype, history_length: int, is_ground_truth: bool
    ) -> torch.Tensor:
        # Gather the last `history_length` snapshots for the whole class (in dtype `dtype`) into a fresh
        # `(B, H, cls_cols_in_dtype)` tensor. Always reads from the per-class return-space ring: it records the
        # post-everything snapshot at each step, so history reads return the final measured (or GT) values observed in
        # the past. The intermediate ring is in pre-hardware-imperfection space and would yield wrong history.
        hist_idx = self._hist_idx_by_class[sensor_cls][:history_length]
        rings = (
            self._ground_truth_return_timeline_ring[sensor_cls]
            if is_ground_truth
            else self._measured_return_timeline_ring[sensor_cls]
        )
        return rings[dtype].at(hist_idx).transpose(0, 1)

    def read_sensors(
        self, entity_idx: int | None = None, envs_idx=None, is_ground_truth: bool = False
    ) -> dict[int, torch.Tensor]:
        """
        Read the latest data of every sensor class in scope as a single tensor per class.

        Always returns a fresh tensor per class, independent of the internal sensor storage; the caller is free to
        mutate the result.

        Parameters
        ----------
        entity_idx : int | None
            - None (default): include every sensor in the scene.
            - k >= 0: include only sensors whose `entity_idx == k`.
            - -1: include only static sensors (those not attached to any entity).
        envs_idx : array-like | int | slice | None
            Environment selection. Defaults to all environments.
        is_ground_truth : bool
            When True, return ground-truth tensors instead of measured tensors.

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from sensor-type tag (`gs.sensors.types.<Name>`) to a tensor of shape
            (B, [history,] class_or_entity_cache_size). For sensors without history, the history
            dimension is omitted. Only ring-pipeline sensor classes are included; lazily-rendered sensors (cameras),
            whose multi-modality output has no single-tensor representation, are read via `sensor.read()` instead.
        """
        # Sanitize envs_idx to a 1D tensor so fancy-indexing the batch axis always allocates a fresh tensor; this is
        # what gives the function its mutation-safe contract.
        env_index = self._sim._scene._sanitize_envs_idx(envs_idx)

        result: dict[int, torch.Tensor] = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            if not sensor_cls.uses_ring_pipeline:
                # Cameras compute their output lazily on render (not through the eager per-step ring pipeline) and span
                # several dtypes; the batched aggregate can't represent them. Read them via `sensor.read()`.
                continue
            return_caches = self._return_cache[sensor_cls]
            # The batched aggregate is one tensor per class, so it only covers single-return-dtype classes (every
            # current ring-pipeline sensor). A hypothetical multi-dtype ring sensor is skipped rather than crashing;
            # read it via `sensor.read()`.
            if len(return_caches) != 1:
                continue
            (return_dtype,) = return_caches.keys()
            entity_slice_map = self._entity_slice_in_class.get(sensor_cls, {})
            if entity_idx is None:
                within_cls_slice = slice(0, return_caches[return_dtype].shape[1])
            else:
                eid = -1 if entity_idx < 0 else entity_idx
                if eid not in entity_slice_map:
                    continue
                within_cls_slice = entity_slice_map[eid]

            cls_max_history = self._max_history_by_class[sensor_cls]
            if cls_max_history > 0:
                sensor_hist = self._gather_history(sensor_cls, return_dtype, cls_max_history, is_ground_truth)
                tensor = sensor_hist[env_index, :, within_cls_slice]
            else:
                return_cache = (
                    self._ground_truth_return_cache[sensor_cls][return_dtype]
                    if is_ground_truth
                    else return_caches[return_dtype]
                )
                tensor = return_cache[env_index, within_cls_slice]

            if self._sim.n_envs == 0:
                tensor = tensor[0]
            options_cls = type(sensors[0]._options)
            type_id = getattr(_sensor_types_namespace, options_cls.__name__)
            result[type_id] = tensor
        return result

    def get_sensors_by_entity(self, entity_idx: int) -> "gs.List[Sensor]":
        """List of all sensors attached to the given entity (or static sensors for entity_idx == -1)."""
        target_eid = -1 if entity_idx < 0 else entity_idx
        return gs.List(
            sensor
            for sensor_list in self._sensors_by_type.values()
            for sensor in sensor_list
            if sensor._options.entity_idx == target_eid
        )

    @property
    def sensors(self):
        return gs.List([sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list])
