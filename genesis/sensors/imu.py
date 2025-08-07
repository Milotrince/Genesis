from typing import List, Tuple

import numpy as np
import taichi as ti
import torch

from genesis.utils.geom import (
    euler_to_quat,
    inv_transform_by_trans_quat,
    transform_quat_by_quat,
)

from .base_sensor import Sensor


@ti.data_oriented
class IMU(Sensor):
    CACHE_DTYPE = torch.float32
    CACHE_SHAPE = (2, 3)  # linear_acceleration, angular_velocity

    def build(self):
        super().build()
        assert (
            self._options.return_accelerometer or self._options.return_gyroscope
        ), "At least one of return_accelerometer or return_gyroscope should be True."
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

    def read(self, envs_idx: List[int] | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns tuple(linear_acceleration, angular_velocity)."""
        if not self._is_cache_updated():
            # updated shared cache (all IMUs in all envs)
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

            # cache shape: (B, n_links, 2, 3)
            self._cache.copy_(torch.cat([local_acc.unsqueeze(2), local_ang.unsqueeze(2)], dim=2))
            self._set_cache_updated()

        if envs_idx is None:
            envs_idx = 0 if self._solver.n_envs == 0 else np.arange(self._cache.shape[0])

        ret = []
        if self._options.return_accelerometer:
            ret.append(self._cache[envs_idx, self._cache_idx, 0, :].squeeze())
        if self._options.return_gyroscope:
            ret.append(self._cache[envs_idx, self._cache_idx, 1, :].squeeze())
        return tuple(ret)
