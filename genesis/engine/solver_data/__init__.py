"""Typed descriptions and internal runtime data shared by solvers and couplers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
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
        self._is_built = False

    def allocate(self, owner: "Solver", description: SolverDescription[DataT]) -> DataT:
        assert not self._is_built
        return self.add(description.allocate(owner))

    def add(self, data: DataT) -> DataT:
        assert not self._is_built
        for record in self._records:
            if type(record) is type(data) and record.owner is data.owner and record.idx == data.idx:
                gs.raise_exception(
                    f"Duplicate {type(data).__name__} at index {data.idx} for {type(data.owner).__name__}."
                )
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

    def build(self):
        self._is_built = True

    def clear(self):
        self._records.clear()
        self._is_built = False
