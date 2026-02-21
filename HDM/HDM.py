from typing import Optional

import numpy as np

from .utils import HDMConfig, HDMResult, get_backend


def run_hdm(
    config: HDMConfig = HDMConfig(),
    data_samples: Optional[list[np.ndarray]] = None,
    maps=None,
) -> HDMResult:
    """
    Computes the Horizontal Diffusion Maps (HDM) and Horizontal Base Diffusion Maps (HBDM)
    embedding from input data.

    This function constructs and processes base and fiber kernels from the input data or
    precomputed distances/kernels, normalizes the resulting joint kernel, and computes
    a HDM embedding.

    Parameters:
        config (HDMConfig): Configuration object specifying HDM parameters.
        data_samples (list[np.ndarray], optional): List of data arrays (e.g., sampled fibers).
        base_kernel (coo_matrix, optional): Precomputed base kernel (spatial proximity).
        fiber_kernel (coo_matrix, optional): Precomputed fiber kernel (fiber similarity).
        base_distances (coo_matrix, optional): Precomputed base distances.
        fiber_distances (coo_matrix, optional): Precomputed fiber distances.

    Returns:
        np.ndarray: Diffusion coordinates from the joint HDM embedding.
    """

    backend = get_backend(config)

    if config.verbose:
        print("Compute HDM Embedding")

    base_kern, base_idx = backend.compute_base_kernel(config, data_samples)

    if config.verbose:
        print("Compute base kernel: Done.")

    fiber_kerns, fiber_idxs = backend.compute_fiber_kernels(config, data_samples)

    if config.verbose:
        print("Compute fiber kernel: Done.")

    joint_kernel = backend.compute_joint_kernel(
        config, base_kern, base_idx, fiber_kerns, fiber_idxs, maps
    )
    if config.verbose:
        print("Construct Joint Kernel Matrix: Done.")

    block_sizes = [len(s) for s in data_samples]
    result = backend.spectral_embedding(config, joint_kernel, block_sizes)
    if config.verbose:
        print("Spectral embedding: Done.")

    return result
