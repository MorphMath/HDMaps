
import numpy as np
import scipy.sparse as sp

from hdmaps.mappings import MapBundle
from hdmaps.types import Indexable

from .utils import (
    HDMConfig,
    HDMResult,
    get_backend,
    get_sizes,
    validate_dtypes,
)


def run_hdm(
    config: HDMConfig,
    base_dist: sp.csr_matrix,
    maps: MapBundle,
    fiber_dists: Indexable,
) -> HDMResult:
    """Compute the horizontal diffusion map embedding.

    Parameters
    ----------
    config : HDMConfig
    base_dist : scipy.sparse.csr_matrix
    maps : MapBundle
    fiber_dists : Indexable[scipy.sparse.csr_matrix]

    Returns
    -------
    HDMResult
    """

    validate_dtypes(config, base_dist, maps)


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
