import numpy as np

from .utils import HDMConfig, HDMResult, get_backend, get_sizes, validate_dtypes


def run_hdm(
    config: HDMConfig,
    base_dist: np.ndarray,
    maps: np.ndarray,
) -> HDMResult:
    """
    Computes the Horizontal Diffusion Maps (HDM) and Horizontal Base Diffusion Distance (HBDD) from precomputed base distances and fiber maps.

    Builds the base kernel from the base distances, assembles the joint kernel over all
    fibers using the maps, normalizes it, and computes the spectral embedding.

    Parameters:
        config (HDMConfig): Configuration object specifying HDM parameters.
        base_dist (np.ndarray): Dense (num_samples, num_samples) matrix of base distances.
        maps (np.ndarray): (num_samples, num_samples) object array of fiber correspondence
            blocks

    Returns:
        HDMResult: Eigenvectors, eigenvalues, HDM coordinates and HBDD coordinates.
    """

    validate_dtypes(config, base_dist, maps)

    num_data_samples, sizes = get_sizes(maps)
    backend = get_backend(config)

    if config.verbose:
        print("Compute HDM Embedding")

    base_kern = backend.build_base_kernel(config, base_dist)

    if config.verbose:
        print("Compute base kernel: Done.")

    horizontal_diffusion_matrix = backend.build_horizontal_diffusion_matrix(
        config, maps, base_kern, num_data_samples
    )
    if config.verbose:
        print("Construct Joint Kernel Matrix: Done.")

    result = backend.compute_spectral_embedding(config, horizontal_diffusion_matrix, sizes, num_data_samples)
    if config.verbose:
        print("Spectral embedding: Done.")

    return result
