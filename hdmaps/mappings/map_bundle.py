import hashlib
import os
import pickle
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from hdmaps.types import Indexable, Indexable2D, MapLookup


class Storage(Protocol):
    def setup(self, compute_map: Callable[[int, int], sp.csr_matrix], data: Indexable, mask: Indexable2D) -> None: ...
    def get(self, i: int, j: int) -> sp.csr_matrix: ...


def _hash_item(item) -> str:
    return hashlib.sha256(pickle.dumps(item)).hexdigest()[:16]


class MapBundle[T]:
    def __init__(
        self,
        compute_map_f: Callable[[T, T], sp.csr_matrix],
        data: Indexable[T],
        storage: Storage | None = None,
        mask: Indexable2D | None = None,
    ):
        self._build(lambda i, j: compute_map_f(data[i], data[j]), data, storage, mask)

    @classmethod
    def from_maps(
        cls,
        maps: MapLookup,
        data: Indexable[T],
        storage: Storage | None = None,
        mask: Indexable2D | None = None,
    ) -> "MapBundle[T]":
        bundle = cls.__new__(cls)
        bundle._build(lambda i, j: maps[i, j], data, storage, mask)
        return bundle

    def _build(
        self,
        compute_map: Callable[[int, int], sp.csr_matrix],
        data: Indexable[T],
        storage: Storage | None,
        mask: Indexable2D | None,
    ):
        n = len(data)
        self.data = data
        self.mask = np.broadcast_to(True, (n, n)) if mask is None else mask
        if self.mask.shape != (n, n):
            raise ValueError(f"mask shape must be ({n}, {n}), got {self.mask.shape}")

        self.storage = MemoryStorage() if storage is None else storage
        self.storage.setup(compute_map, data, self.mask)

    def __getitem__(self, key: tuple[int, int]) -> sp.csr_matrix:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(f"expected a (i, j) tuple, got {key!r}")
        i, j = key
        n = len(self.data)
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"index out of range for data of length {n}: got ({i}, {j})")
        if not self.mask[i, j]:
            raise LookupError(f"mapping ({i}, {j}) is excluded by mask")
        return self.storage.get(i, j)


class MemoryStorage:
    def setup(self, compute_map, data, mask):
        self.compute_map = compute_map
        self._cache: dict[tuple[int, int], sp.csr_matrix] = {}

    def get(self, i, j):
        if (i, j) not in self._cache:
            self._cache[(i, j)] = self.compute_map(i, j)
        return self._cache[(i, j)]


class DirStorage:
    def __init__(self, dir_path: str, max_cache_size: int | None = 1024):
        self.dir_path = dir_path
        self.max_cache_size = max_cache_size

    def setup(self, compute_map, data, mask):
        os.makedirs(self.dir_path, exist_ok=True)
        self.compute_map = compute_map
        self._hashes = [_hash_item(d) for d in data]
        self._cache: OrderedDict[str, sp.csr_matrix] = OrderedDict()
        for i, j in zip(*mask.nonzero()):
            self.get(i, j)

    def get(self, i: int, j: int) -> sp.csr_matrix:
        key = f"{self._hashes[i]}_{self._hashes[j]}"

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        path = f"{self.dir_path}/{key}.npz"
        if os.path.exists(path):
            result = sp.load_npz(path)
        else:
            result = self.compute_map(i, j)
            sp.save_npz(path, result)

        self._cache[key] = result
        if self.max_cache_size is not None and len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return result


class PackedStorage:
    def __init__(self, path: str):
        self.path = path if path.endswith(".npz") else f"{path}.npz"

    def setup(self, compute_map, data, mask):
        hashes = [_hash_item(d) for d in data]
        pairs = {(i, j): f"{hashes[i]}_{hashes[j]}" for i, j in zip(*mask.nonzero())}

        stored = _load_packed(self.path) if os.path.exists(self.path) else {}
        maps = {
            key: stored[key] if key in stored else compute_map(i, j)
            for (i, j), key in pairs.items()
        }
        if maps and maps.keys() != stored.keys():
            _save_packed(self.path, maps)

        self._maps = {ij: maps[key] for ij, key in pairs.items()}

    def get(self, i, j):
        if (i, j) not in self._maps:
            raise LookupError(f"mapping ({i}, {j}) was not computed")
        return self._maps[i, j]


def _save_packed(path: str, maps: dict[str, sp.csr_matrix]):
    ms = list(maps.values())
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            keys=np.array(list(maps)),
            shapes=np.array([m.shape for m in ms]),
            dtypes=np.array([m.dtype.str for m in ms]),
            nnz=np.cumsum([0, *(m.nnz for m in ms)]),
            data=np.concatenate([m.data for m in ms]),
            indices=np.concatenate([m.indices for m in ms]),
            indptr=np.concatenate([m.indptr for m in ms]),
        )
    os.replace(tmp, path)


def _load_packed(path: str) -> dict[str, sp.csr_matrix]:
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}
    nnz = z["nnz"]
    ptr = np.cumsum([0, *(z["shapes"][:, 0] + 1)])
    return {
        str(key): sp.csr_matrix(
            (
                z["data"][nnz[k]:nnz[k + 1]].astype(dtype, copy=False),
                z["indices"][nnz[k]:nnz[k + 1]],
                z["indptr"][ptr[k]:ptr[k + 1]],
            ),
            shape=tuple(shape),
        )
        for k, (key, shape, dtype) in enumerate(zip(z["keys"], z["shapes"], z["dtypes"]))
    }
