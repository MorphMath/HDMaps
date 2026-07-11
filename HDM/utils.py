from typing import NamedTuple

import numpy as np
import torch


class HDMConfig(NamedTuple):
    base_epsilon: float | None = None
    num_eigenvectors: int = 5
    device: torch.device = torch.device("cpu")
    base_metric: str = "frobenius"
    base_knn: int = 4
    verbose: bool = True
    seed: int = 67
    alpha: float = 1.0
    t: float = 1.0
    dtype: type = np.float64
    


class HDMResult(NamedTuple):
    eigvecs: np.ndarray
    eigvals: np.ndarray
    HDM: np.ndarray
    HBDD: np.ndarray


def get_backend(config: HDMConfig):
    from . import backend

    return backend


def torch_dtype(dtype) -> torch.dtype:
    return torch.from_numpy(np.empty(0, dtype=dtype)).dtype


def validate_dtypes(base_dist: np.ndarray, maps: np.ndarray, config: HDMConfig):
    expected = np.dtype(config.dtype)
    if base_dist.dtype != expected:
        raise ValueError(f"base_dist is {base_dist.dtype}, expected {expected}")
    for i in range(len(maps)):
        for j in range(len(maps)):
            block = maps[i][j]
            if block is not None and block.dtype != expected:
                raise ValueError(f"maps[{i}][{j}] is {block.dtype}, expected {expected}")

def approx_base_eps(D: np.ndarray):
    return np.median(np.max(D, axis=1)) ** 2


def get_sizes(maps: np.ndarray) -> tuple[int, int]:
    num_data_samples = len(maps)
    sizes = [maps[i][i].shape[0] for i in range(num_data_samples)]
    return (num_data_samples, sizes)

def _is_cuda(device) -> bool:
    try:
        return torch.device(device).type == "cuda"
    except Exception:
        return False