"""Read access to solver-owned arrays."""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_map

import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_to_torch

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


class ReadOnlyTensor(torch.Tensor):
    """A tensor view that rejects writes through Torch operations.

    Slices and other storage-sharing results retain read protection. Arithmetic and clones produce ordinary tensors.
    NumPy conversion produces a snapshot. External kernels must consume a snapshot because they bypass Torch dispatch.
    Access after the owning scene is destroyed raises. Python private attributes are outside this access contract.
    Solver gradient history is available through queried states. These views expose numerical observations.
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, owners: "tuple[Solver, ...]" = ()):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.shape,
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            device=tensor.device,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor, owners: "tuple[Solver, ...]" = ()):
        self._tensor = tensor
        self._owners = owners

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        values, _ = tree_flatten((args, kwargs))
        owners = []
        for value in values:
            if isinstance(value, ReadOnlyTensor):
                for owner in value._owners:
                    if not owner.is_built:
                        gs.raise_exception("The scene owning this data has been destroyed.")
                    if owner not in owners:
                        owners.append(owner)
        for index, argument in enumerate(func._schema.arguments):
            if argument.alias_info is not None and argument.alias_info.is_write:
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                targets, _ = tree_flatten(value)
                if any(isinstance(target, ReadOnlyTensor) for target in targets):
                    gs.raise_exception("Solver data is read-only. Use the owning entity's setters or clone the tensor.")
        result = func(*tree_map(_unwrap_tensor, args), **tree_map(_unwrap_tensor, kwargs))
        if any(result_type.alias_info is not None for result_type in func._schema.returns):
            return tree_map(partial(_wrap_tensor, owners=tuple(owners)), result)
        return result

    def __repr__(self):
        return f"ReadOnlyTensor({self.clone()!r})"

    def numpy(self, *, force=False) -> np.ndarray:
        """Return an independent NumPy snapshot."""
        return self.clone().numpy(force=force)

    def tolist(self):
        return self.clone().tolist()

    def data_ptr(self):
        gs.raise_exception("Export a clone to access tensor storage.")

    def storage(self):
        gs.raise_exception("Export a clone to access tensor storage.")

    def untyped_storage(self):
        gs.raise_exception("Export a clone to access tensor storage.")

    def __dlpack__(self, *args, **kwargs):
        gs.raise_exception("Export a clone to use DLPack.")

    def as_subclass(self, cls):
        gs.raise_exception("Clone read-only data before converting its tensor type.")

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        gs.raise_exception("Solver data is read-only. Clone the tensor before replacing its data.")


def _unwrap_tensor(value):
    return value._tensor if isinstance(value, ReadOnlyTensor) else value


def _wrap_tensor(value, *, owners):
    return ReadOnlyTensor(value, owners) if isinstance(value, torch.Tensor) else value


@dataclass(frozen=True)
class DataReference:
    """A read-only subview of an array owned by one solver.

    ``selection`` indexes the array after its environment axis moves to the front. Basic slices retain storage
    sharing. The owner refreshes derived values before a read. References remain valid for the owning scene's built
    lifetime, including stepping and in-place reset. ``read()`` returns a snapshot and ``read(copy=False)`` a live view.
    """

    owner: "Solver"
    name: str
    _array: qd.Tensor | qd.Field | torch.Tensor
    selection: tuple[slice, ...] = ()
    _prepare: Callable[[], None] | None = None

    def __post_init__(self):
        if any(not isinstance(index, slice) for index in self.selection):
            gs.raise_exception(
                "A DataReference selection must contain slices. Gather from a snapshot for other indices."
            )

    def read(self, *, copy: bool = True) -> torch.Tensor:
        """Read a snapshot, or a protected live view when copying is disabled.

        Live views require zero-copy interoperation on the active backend. Numerical reads require a built scene.
        The owner refreshes derived values on each call. A retained view observes subsequent writes to the same array.
        """
        if not self.owner.is_built:
            gs.raise_exception("Reading solver data requires its owning scene to be built.")
        if self._prepare is not None:
            self._prepare()
        if isinstance(self._array, torch.Tensor):
            tensor = self._array
        else:
            tensor = qd_to_torch(self._array, transpose=True, copy=None if copy else False)
        tensor = tensor[self.selection]
        return tensor.clone() if copy else ReadOnlyTensor(tensor, (self.owner,))
