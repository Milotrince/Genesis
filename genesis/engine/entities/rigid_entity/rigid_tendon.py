import numpy as np

import genesis as gs
from genesis.repr_base import RBC


class RigidTendon(RBC):
    """
    Fixed tendon for rigid body entities.

    A fixed tendon defines a scalar length ``L = sum_i coef_i * qpos_i`` over its member joints, coupling them
    through passive spring/damping, length limits, frictionloss, and/or an actuator transmission. Because the length
    is a linear combination of joint positions, the moment arm w.r.t. each member DOF is the constant coefficient
    ``coef_i``. Spatial tendons (site/geom routing) are not supported.
    """

    def __init__(
        self,
        entity,
        name,
        idx,
        kind,
        members,
        wraps,
        stiffness,
        damping,
        springlength,
        frictionloss,
        limited,
        limit,
        sol_params,
        sol_params_limit,
        act_gain,
        act_bias,
        force_range,
        length0=0.0,
    ):
        self._name = name
        self._entity = entity
        self._solver = entity.solver

        self._uid = gs.UID()
        self._idx = idx
        self._kind = kind

        # members: list of (joint_name, coef)  [FIXED tendons]
        self._members = tuple(members)
        # wraps: ordered list of dicts describing the spatial path  [SPATIAL tendons]
        self._wraps = tuple(wraps)
        self._stiffness = float(stiffness)
        self._damping = float(damping)
        self._springlength = np.asarray(springlength, dtype=gs.np_float)
        self._frictionloss = float(frictionloss)
        self._limited = bool(limited)
        self._limit = np.asarray(limit, dtype=gs.np_float)
        self._sol_params = np.asarray(sol_params, dtype=gs.np_float)
        self._sol_params_limit = np.asarray(sol_params_limit, dtype=gs.np_float)
        self._act_gain = float(act_gain)
        self._act_bias = np.asarray(act_bias, dtype=gs.np_float)
        self._force_range = np.asarray(force_range, dtype=gs.np_float)
        self._length0 = float(length0)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def control_position(self, pos, envs_idx=None):
        """Set the position (target length) control target for this tendon's actuator."""
        self._solver.control_tendons_position(pos, tendons_idx=self._idx, envs_idx=envs_idx)

    def control_velocity(self, vel, envs_idx=None):
        """Set the velocity control target for this tendon's actuator."""
        self._solver.control_tendons_velocity(vel, tendons_idx=self._idx, envs_idx=envs_idx)

    def control_force(self, force, envs_idx=None):
        """Set the direct force control target for this tendon's actuator."""
        self._solver.control_tendons_force(force, tendons_idx=self._idx, envs_idx=envs_idx)

    def get_length(self, envs_idx=None):
        """Return the current tendon length ``L = sum_i coef_i * qpos_i``."""
        return self._solver.get_tendons_length(tendons_idx=self._idx, envs_idx=envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """Returns the unique id of the tendon."""
        return self._uid

    @property
    def name(self):
        """Returns the name of the tendon."""
        return self._name

    @property
    def entity(self):
        """Returns the entity that the tendon belongs to."""
        return self._entity

    @property
    def solver(self):
        """The RigidSolver object that the tendon belongs to."""
        return self._solver

    @property
    def idx(self):
        """Returns the global index of the tendon in the rigid solver."""
        return self._idx

    @property
    def idx_local(self):
        """Returns the local index of the tendon in the entity."""
        return self._idx - self._entity._tendon_start

    @property
    def kind(self):
        """Returns the tendon kind (`gs.TENDON_TYPE.FIXED` or `gs.TENDON_TYPE.SPATIAL`)."""
        return self._kind

    @property
    def members(self):
        """Returns the tuple of ``(joint_name, coef)`` members of the fixed tendon."""
        return self._members

    @property
    def wraps(self):
        """Returns the ordered spatial wrap path (tuple of element dicts) for a spatial tendon."""
        return self._wraps

    @property
    def is_built(self):
        """Whether the rigid entity this tendon belongs to is built."""
        return self.entity.is_built

    def _repr_brief(self):
        return f"{self.__repr_name__()}, idx: {self.idx}, name: '{self.name}'"
