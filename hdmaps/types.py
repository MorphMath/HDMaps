from typing import Protocol
import numpy as np
import scipy.sparse as sp


class Indexable[T](Protocol):
    """Sequence-like container indexed by an integer."""

    def __getitem__(self, key: int, /) -> T: ...
    def __len__(self) -> int: ...

class Indexable2D(Protocol):
    """Boolean 2D mask indexed by ``(i, j)``."""

    def __getitem__(self, key: tuple[int, int], /) -> bool: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    def nonzero(self) -> tuple[np.ndarray, ...]: ...

class MapLookup(Protocol):
    """Lookup of correspondence maps indexed by ``(i, j)``."""

    def __getitem__(self, key: tuple[int, int], /) -> sp.csr_matrix: ...
