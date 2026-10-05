
import numpy as np
import scipy.sparse as sp

from hdmaps.mappings import MapBundle
from hdmaps.types import Indexable

from .utils import (
    HDMConfig,
    HDMResult,
    get_backend,
    cast,
    get_sizes,
)
from .validation import validate_inputs


def run_hdm(
    config: HDMConfig,
    base_dist: sp.csr_matrix,
    maps: MapBundle,
    fiber_dists: Indexable,
) -> HDMResult:
    """
    Computes the Horizontal Diffusion Maps (HDM) and Horizontal Base Diffusion Distance (HBDD) from precomputed base distances and fiber maps.

    Builds the base kernel from the base distances, assembles the joint kernel over all
    fibers using the maps, normalizes it, and computes the spectral embedding.

    Parameters:
        config (HDMConfig): Configuration object specifying HDM parameters.
        base_dist (np.ndarray): Dense (num_samples, num_samples) matrix of base distances.
        maps (np.ndarray): (num_samples, num_samples) object array of fiber correspondence
            blocks.
        fiber_dists (np.ndarray): (num_samples) object array of distance
            matrices on each data sample.

    Returns:
        HDMResult: Eigenvectors, eigenvalues, HDM coordinates and HBDD coordinates.
    """

    validate_inputs(config, base_dist, maps, fiber_dists)
    [base_dist] = cast([base_dist], config.dtype, "base_dist")
    fiber_dists = cast([fiber_dists[i] for i in range(len(fiber_dists))], config.dtype, "fiber_dists")


    num_data_samples, sizes = get_sizes(fiber_dists)
    offsets = np.cumsum([0, *sizes])
    backend = get_backend(config)

    if config.verbose:
        print("Compute HDM Embedding")

    base_kern = backend.build_base_kernel(config, base_dist)

    if config.verbose:
        print("Compute base kernel: Done.")


    horizontal_diffusion_matrix = backend.build_horizontal_diffusion_matrix(
        config, maps, base_kern, fiber_dists, offsets
    )
    if config.verbose:
        print("Construct Joint Kernel Matrix: Done.")

    result = backend.compute_spectral_embedding(config, horizontal_diffusion_matrix, offsets, num_data_samples)
    if config.verbose:
        print("Spectral embedding: Done.")

    return result
