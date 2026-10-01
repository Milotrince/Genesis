"""Typed descriptions and internal runtime data shared by solvers and couplers."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import genesis as gs

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


@dataclass(frozen=True, kw_only=True, eq=False)
class SolverData:
    """A domain record identified by its type, owner, and owner-local index."""

    owner: "Solver"
    idx: int = 0
    entity_idx: int | None = None


DataT = TypeVar("DataT", bound=SolverData)


class SolverDescription(ABC, Generic[DataT]):
    """Resolved inputs and dimensions sufficient to allocate a solver's shared data."""

    @abstractmethod
    def allocate(self, owner: "Solver") -> DataT:
        raise NotImplementedError


class SolverDataArray:
    """Scene-owned runtime records, resolved by type and owner during build."""

    def __init__(self):
        self._records: list[SolverData] = []
        self._index: dict[tuple[type[SolverData], Solver, int], SolverData] = {}
        self._is_built = False

    def allocate(self, owner: "Solver", description: SolverDescription[DataT]) -> DataT:
        assert not self._is_built
        return self.add(description.allocate(owner))

    def add(self, data: DataT) -> DataT:
        assert not self._is_built
        key = (type(data), data.owner, data.idx)
        if key in self._index:
            gs.raise_exception(f"Duplicate {type(data).__name__} at index {data.idx} for {type(data.owner).__name__}.")
        self._index[key] = data
        self._records.append(data)
        return data

    def select(self, data_type: type[DataT], owner: "Solver | None" = None) -> Iterator[DataT]:
        for record in self._records:
            if isinstance(record, data_type) and (owner is None or record.owner is owner):
                yield record

    def get(self, data_type: type[DataT], owner: "Solver", idx: int = 0) -> DataT:
        records = [record for record in self.select(data_type, owner) if record.idx == idx]
        if len(records) != 1:
            gs.raise_exception(
                f"Expected one {data_type.__name__} at index {idx} for {type(owner).__name__}, found {len(records)}."
            )
        return records[0]

    def contains(self, data: SolverData) -> bool:
        return self._index.get((type(data), data.owner, data.idx)) is data

    def bind(self, data: DataT, *, write: str | None = None) -> "SolverDataBinding[DataT]":
        if not self.contains(data):
            gs.raise_exception("Solver data belongs to a different or destroyed scene.")
        commit = None if write is None else data.owner.bind_data_write(data, write)
        return SolverDataBinding(data=data, collection=self, write=write, writer=commit)

    def bind_coupling(self, data: DataT, *, writes: tuple[str, ...]) -> "CouplingDataAccess[DataT]":
        if not self.contains(data) or data is not data.owner._solver_data:
            gs.raise_exception("Coupling requires the owning solver's aggregate record in this scene.")
        if not writes or not set(writes).issubset(data.owner.coupling_fields):
            gs.raise_exception(f"Unsupported coupling writes for {type(data.owner).__name__}: {writes}.")
        return CouplingDataAccess(data=data, collection=self, writes=writes)

    def build(self):
        self._is_built = True

    def clear(self):
        self._records.clear()
        self._index.clear()
        self._is_built = False


@dataclass(frozen=True, kw_only=True)
class SolverDataBinding(Generic[DataT]):
    """A retained record reference with an optional owner-provided write operation."""

    data: DataT
    collection: SolverDataArray
    write: str | None
    writer: Callable | None

    def commit(self, value, envs_idx=None):
        if not self.collection.contains(self.data):
            gs.raise_exception("Solver data belongs to a destroyed scene.")
        if self.writer is None:
            gs.raise_exception("This solver data binding is read-only.")
        self.writer(value, envs_idx=envs_idx)


@dataclass(frozen=True, kw_only=True)
class CouplingDataAccess(Generic[DataT]):
    """Native substep writes, committed after every solver completes its post-coupling phase."""

    data: DataT
    collection: SolverDataArray
    writes: tuple[str, ...]

    def commit(self):
        if not self.collection.contains(self.data):
            gs.raise_exception("Solver data belongs to a destroyed scene.")
        self.data.owner.commit_coupling()
