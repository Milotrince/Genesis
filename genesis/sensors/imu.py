import taichi as ti
import torch

import genesis as gs
from genesis.options.sensors import SensorOptions
from genesis.utils.geom import (
    euler_to_quat,
    inv_transform_by_trans_quat,
    transform_quat_by_quat,
)

from .base_sensor import Sensor
from .sensor_manager import register_sensor


@ti.data_oriented
class IMU(Sensor):

    def build(self):
        super().build()
        self._solver = self._manager._sim.rigid_solver
        assert self._options.link_idx >= 0 and self._options.link_idx < self._solver.n_links, "Invalid RigidLink index."
        self._link_idx = self._options.link_idx

        quat_offset = euler_to_quat(self._options.euler_offset)

        if len(self._shared_metadata) == 0:
            self._shared_metadata["links_idx"] = []
            self._shared_metadata["offsets_pos"] = torch.tensor([], dtype=torch.float32)
            self._shared_metadata["offsets_quat"] = torch.tensor([], dtype=torch.float32)

        self._shared_metadata["links_idx"].append(self._link_idx)
        self._shared_metadata["offsets_pos"] = torch.cat(
            [self._shared_metadata["offsets_pos"], torch.tensor([self._options.pos_offset], dtype=torch.float32)]
        )
        self._shared_metadata["offsets_quat"] = torch.cat(
            [self._shared_metadata["offsets_quat"], torch.tensor([quat_offset], dtype=torch.float32)]
        )

    def _get_return_format(self) -> dict[str, tuple[int, int]] | None:
        return {
            "lin_acc": (0, 3),
            "ang_vel": (3, 6),
        }

    def _update_shared_gt_cache(self):
        gravity = self._solver.get_gravity()
        quats = self._solver.get_links_quat(links_idx=self._shared_metadata["links_idx"])
        acc = self._solver.get_links_acc(links_idx=self._shared_metadata["links_idx"])
        ang = self._solver.get_links_ang(links_idx=self._shared_metadata["links_idx"])
        if self._solver.n_envs == 0:
            gravity = gravity.unsqueeze(0)
            quats = quats.unsqueeze(0)
            acc = acc.unsqueeze(0)
            ang = ang.unsqueeze(0)

        offset_quats = transform_quat_by_quat(
            quats, self._shared_metadata["offsets_quat"].unsqueeze(0).repeat(quats.shape[0], 1, 1)
        )
        # acc/ang shape: (B, n_links, 3)
        local_acc = inv_transform_by_trans_quat(acc, self._shared_metadata["offsets_pos"], offset_quats)
        local_ang = inv_transform_by_trans_quat(ang, self._shared_metadata["offsets_pos"], offset_quats)

        local_acc = local_acc - gravity.unsqueeze(1).repeat(1, local_acc.shape[1], 1)

        # cache shape: (B, n_links, 6)
        self._gt_cache.copy_(torch.cat([local_acc, local_ang], dim=2))

    def _update_shared_cache(self):
        self._cache.append(self._gt_cache)

    def _get_cache_length(self) -> int:
        return 1

    @classmethod
    def _get_cache_size(cls) -> int:
        return 2 * 3

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float


@register_sensor(IMU)
class IMUOptions(SensorOptions):
    """
    IMU sensor returns the linear acceleration (accelerometer) and angular velocity (gyroscope)
    of the associated entity link.

    Parameters
    ----------
    link_idx : int
        The global index of the RigidLink to which this IMU sensor is attached.
    pos_offset : tuple[float, float, float]
        The offset of the IMU sensor from the RigidLink.
    euler_offset : tuple[float, float, float]
        The offset of the IMU sensor from the RigidLink in euler angles.
    """

    link_idx: int
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
