import numpy as np

from .utils import HDMConfig, HDMResult, get_backend, get_sizes, validate_dtypes


def run_hdm(
    base_dist: np.ndarray,
    maps: np.ndarray,
    config: HDMConfig = HDMConfig(),
) -> HDMResult:
    """
    Computes the Horizontal Diffusion Maps (HDM) and Horizontal Base Diffusion Distance (HBDD) from precomputed base distances and fiber maps.

    Builds the base kernel from the base distances, assembles the joint kernel over all
    fibers using the maps, normalizes it, and computes the spectral embedding.

    Parameters:
        base_dist (np.ndarray): Dense (num_samples, num_samples) matrix of base distances.
        maps (np.ndarray): (num_samples, num_samples) object array of fiber correspondence
            blocks
        config (HDMConfig): Configuration object specifying HDM parameters.

    Returns:
        HDMResult: Eigenvectors, eigenvalues, HDM coordinates and HBDD coordinates.
    """
    validate_dtypes(base_dist, maps, config)

    num_data_samples, sizes = get_sizes(maps)
    backend = get_backend(config)

    if config.verbose:
        print("Compute HDM Embedding")

    base_kern = backend.base_kernel(base_dist, config)

    if config.verbose:
        print("Compute base kernel: Done.")

    joint_kernel = backend.joint_kernel(
        maps, base_kern, num_data_samples, config
    )
    if config.verbose:
        print("Construct Joint Kernel Matrix: Done.")

    result = backend.spectral_embedding(config, joint_kernel, sizes, num_data_samples)
    if config.verbose:
        print("Spectral embedding: Done.")

    return result
