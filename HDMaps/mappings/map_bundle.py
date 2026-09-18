import os
import hashlib
import pickle
from collections import OrderedDict
from typing import Literal
from dataclasses import dataclass, field
from collections.abc import Callable
import scipy.sparse
from scipy.sparse import csr_matrix


def _hash_item(item) -> str:
    return hashlib.sha256(pickle.dumps(item)).hexdigest()[:16]


class LazyBackend:
    def __init__(self, compute_map_f, data):
        self.compute_map_f = compute_map_f
        self.data = data
        self._cache = {}

    def get(self, i, j) -> csr_matrix:
        if (i, j) not in self._cache:
            self._cache[(i, j)] = self.compute_map_f(self.data[i], self.data[j])
        return self._cache[(i, j)]


class DiskBackend:
    def __init__(self, compute_map_f, data, dir_path, max_cache_size=128):
        self.compute_map_f = compute_map_f
        self.data = data
        self.dir_path = dir_path
        self.max_cache_size = max_cache_size
        self._hashes = [_hash_item(d) for d in data]
        self._cache = OrderedDict()

        n = len(data)
        for i in range(n):
            for j in range(n):
                self.get(i, j)

    def get(self, i, j) -> csr_matrix:
        key = f"{self._hashes[i]}_{self._hashes[j]}"

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        path = f"{self.dir_path}/{key}.npz"
        if os.path.exists(path):
            result = scipy.sparse.load_npz(path)
        else:
            result = self.compute_map_f(self.data[i], self.data[j])
            scipy.sparse.save_npz(path, result)

        self._cache[key] = result
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return result


@dataclass
class MapBundle[T]:
    compute_map_f: Callable[[T, T], csr_matrix]
    data: list[T]
    mode: Literal["lazy", "disk"] = "lazy"
    dir_path: str | None = None
    max_cache_size: int = 128

    def __post_init__(self):
        if self.mode == "lazy":
            self.backend = LazyBackend(self.compute_map_f, self.data)
        elif self.mode == "disk":
            if self.dir_path is None:
                raise ValueError("dir_path is required for disk mode")
            os.makedirs(self.dir_path, exist_ok=True)
            self.backend = DiskBackend(
                self.compute_map_f, self.data, self.dir_path, self.max_cache_size
            )
        else:
            raise ValueError(f"unsupported mode: {self.mode}")

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(f"expected a (i, j) tuple, got {key!r}")
        i, j = key
        n = len(self.data)
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"index out of range for data of length {n}: got ({i}, {j})")
        return self.backend.get(i, j)
