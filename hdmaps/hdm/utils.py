import os
import warnings
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import torch

from hdmaps.types import Indexable


class HDMConfig(NamedTuple):
    """Parameters for :func:`run_hdm`."""

    epsilon: float | None = None
    gamma: float | None = None
    num_eigenvectors: int = 5
    device: torch.device = torch.device("cpu")
    base_metric: str = "frobenius"
    verbose: bool = True
    seed: int = 19203838
    t: float = 1.0
    dtype: type = np.float64
    eig_tol: float = 1e-8
    sinkhorn_max_iter: int = 1000
    sinkhorn_tol: float = 1e-8
    sinkhorn_jitter: float = 1e-8


class HDMResult(NamedTuple):
    """Output of :func:`run_hdm`."""

    eigvecs: np.ndarray
    eigvals: np.ndarray
    HDM: np.ndarray
    HBDM: np.ndarray
    offsets: np.ndarray
    config: HDMConfig



def torch_dtype(dtype) -> torch.dtype:
    return torch.from_numpy(np.empty(0, dtype=dtype)).dtype


def warn(message: str) -> None:
    # point at the user's line, not at ours
    warnings.warn(message, skip_file_prefixes=(os.path.dirname(__file__) + os.sep,))


def warn_cast(name: str, from_dtypes: set[str], dtype: type) -> None:
    if from_dtypes:
        warn(f"{name}: cast from {', '.join(sorted(from_dtypes))} to {np.dtype(dtype)} (config.dtype)")


def cast(mats: list[sp.csr_matrix], dtype: type, name: str) -> list[sp.csr_matrix]:
    warn_cast(name, {str(m.dtype) for m in mats if m.dtype != dtype}, dtype)
    return [sp.csr_matrix(m.astype(dtype, copy=False)) for m in mats]


def get_sizes(fiber_dists: Indexable[sp.csr_matrix]) -> tuple[int, list[int]]:
    num_data_samples = len(fiber_dists)
    sizes = []
    for i in range(num_data_samples):
        shape = fiber_dists[i].shape
        assert shape is not None
        sizes.append(shape[0])
    return (num_data_samples, sizes)


def typical_distance(mats: Indexable[sp.csr_matrix]) -> float:
    # median over the matrices of each one's median non-zero distance
    medians = [np.median(d) for i in range(len(mats)) if (d := mats[i].data[mats[i].data > 0]).size]
    return float(np.median(medians)) if medians else 0.0


def estimate_epsilon(base_dist: sp.csr_matrix) -> float:
    return typical_distance([base_dist])


def estimate_gamma(epsilon: float, fiber_dists: Indexable[sp.csr_matrix]) -> float:
    fiber = typical_distance(fiber_dists)
    return (epsilon / fiber) ** 2 if fiber > 0 else float("inf")


def resolve(config: HDMConfig, base_dist: sp.csr_matrix, fiber_dists: Indexable[sp.csr_matrix]) -> HDMConfig:
    epsilon = estimate_epsilon(base_dist) if config.epsilon is None else config.epsilon
    gamma = estimate_gamma(epsilon, fiber_dists) if config.gamma is None else config.gamma
    for name, value in (("epsilon", epsilon), ("gamma", gamma)):
        if not 0 < value < float("inf"):
            raise ValueError(f"C1: {name}: estimated as {value}; set it in HDMConfig")
    return config._replace(epsilon=epsilon, gamma=gamma)


def _is_cuda(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"
