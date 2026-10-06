import os
import warnings
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import torch

from hdmaps.types import Indexable


class HDMConfig(NamedTuple):
    """Parameters for :func:`run_hdm`."""

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
    """Output of :func:`run_hdm`."""

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


def approx_eps(mats: Indexable[sp.csr_matrix]) -> float:
    # median over the matrices of each one's median non-zero distance
    medians = [np.median(d) for i in range(len(mats)) if (d := mats[i].data[mats[i].data > 0]).size]
    return float(np.median(medians)) if medians else 0.0


def resolve_epsilon(eps: float | None, mats: Indexable[sp.csr_matrix], name: str) -> float:
    if eps is not None:
        return eps
    eps = approx_eps(mats)
    if not eps > 0:
        raise ValueError(f"C1: {name}: estimated as {eps}; set it in HDMConfig")
    return eps


def _is_cuda(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"
