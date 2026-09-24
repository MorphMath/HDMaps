from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import torch

from hdmaps.mappings import MapBundle
from hdmaps.types import Indexable


class HDMConfig(NamedTuple):
    base_epsilon: float | None = None
    fiber_epsilon: float | None = None
    num_eigenvectors: int = 5
    device: torch.device = torch.device("cpu")
    base_metric: str = "frobenius"
    verbose: bool = True
    seed: int = 67
    t: float = 1.0
    dtype: type = np.float64
    eig_tol: float = 1e-8
    sinkhorn_max_iter: int = 1000
    sinkhorn_tol: float = 1e-8
    sinkhorn_jitter: float = 1e-8


class HDMResult(NamedTuple):
    eigvecs: np.ndarray
    eigvals: np.ndarray
    HDM: np.ndarray
    HBDM: np.ndarray
    HBDD: np.ndarray
    offsets: np.ndarray
    config: HDMConfig



def get_backend(config: HDMConfig):
    from . import backend

    return backend


def torch_dtype(dtype) -> torch.dtype:
    return torch.from_numpy(np.empty(0, dtype=dtype)).dtype


def validate_dtypes(config: HDMConfig, base_dist: sp.csr_matrix, maps: MapBundle):
    expected = np.dtype(config.dtype)
    if base_dist.dtype != expected:
        raise ValueError(f"base_dist is {base_dist.dtype}, expected {expected}")
    for i, j in zip(*maps.mask.nonzero()):
        block = maps[i, j]
        if block.dtype != expected:
            raise ValueError(f"maps[{i}, {j}] is {block.dtype}, expected {expected}")

    if config.fiber_epsilon is None:
        raise ValueError(f"Fiber epsilon is {None} expected float")


def get_sizes(fiber_dists: Indexable[sp.csr_matrix]) -> tuple[int, list[int]]:
    num_data_samples = len(fiber_dists)
    sizes = []
    for i in range(num_data_samples):
        shape = fiber_dists[i].shape
        assert shape is not None
        sizes.append(shape[0])
    return (num_data_samples, sizes)


def approx_base_eps(D: sp.csr_matrix) -> float:
    row_max = np.asarray(D.max(axis=1).todense()).ravel()
    return float(np.median(row_max))


def _is_cuda(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"
