import numpy as np

import genesis as gs
from genesis.repr_base import RBC


class RigidSite(RBC):
    """
    Site for rigid body entities.

    A site is a massless frame rigidly attached to a link, used as a reference point (e.g. a via-point or anchor for
    spatial tendons and equality constraints). Its world pose is recomputed each step from the owning link's transform.
    """

    def __init__(self, entity, name, idx, link_name, pos, quat):
        self._name = name
        self._entity = entity
        self._solver = entity.solver

        self._uid = gs.UID()
        self._idx = idx
        self._link_name = link_name
        self._pos = np.asarray(pos, dtype=gs.np_float)
        self._quat = np.asarray(quat, dtype=gs.np_float)

    @property
    def uid(self):
        """Returns the unique id of the site."""
        return self._uid

    @property
    def name(self):
        """Returns the name of the site."""
        return self._name

    @property
    def entity(self):
        """Returns the entity that the site belongs to."""
        return self._entity

    @property
    def idx(self):
        """Returns the global index of the site in the rigid solver."""
        return self._idx

    @property
    def idx_local(self):
        """Returns the local index of the site in the entity."""
        return self._idx - self._entity._site_start

    @property
    def link_name(self):
        """Returns the name of the link the site is attached to."""
        return self._link_name

    @property
    def pos(self):
        """Returns the site's local position in the owning link frame."""
        return self._pos

    @property
    def quat(self):
        """Returns the site's local orientation in the owning link frame."""
        return self._quat

    def _repr_brief(self):
        return f"{self.__repr_name__()}, idx: {self.idx}, name: '{self.name}'"
