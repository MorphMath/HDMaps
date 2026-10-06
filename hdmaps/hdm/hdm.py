
import numpy as np
import scipy.sparse as sp

from hdmaps.mapping import MapBundle
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
    """Compute horizontal diffusion maps (HDM) and horizontal base diffusion distances (HBDD).

    Builds a kernel on the base from ``base_dist``, lifts it through the maps onto the fibers into
    a joint kernel over all points, normalizes it with Sinkhorn, and embeds it with its leading
    eigenvectors. Kernels are ``exp(-d² / ε²)``.

    Parameters
    ----------
    config : HDMConfig
        Parameters. ``base_epsilon`` and ``fiber_epsilon`` may be None to estimate them as the
        median over matrices of each matrix's median non-zero distance.
    base_dist : scipy.sparse.csr_matrix
        ``(n, n)`` distances between the ``n`` objects. Stored entries are the neighbors; the zero
        diagonal must be stored.
    maps : MapBundle
        Correspondences between objects. ``maps[i, j]`` is a ``(κᵢ, κⱼ)`` CSR matrix whose rows
        sum to 1, needed in both directions for every pair of neighbors. ``maps[i, i]`` is never
        read; the identity is used.
    fiber_dists : Indexable[scipy.sparse.csr_matrix]
        ``n`` matrices; ``fiber_dists[i]`` is the ``(κᵢ, κᵢ)`` distance matrix between the points
        of object ``i``, with its zero diagonal stored.

    Returns
    -------
    HDMResult
        With ``κ`` the total number of points and ``k = config.num_eigenvectors``:

        - ``eigvecs``: ``(κ, k)`` leading non-trivial eigenvectors of the normalized joint kernel.
        - ``eigvals``: ``(k,)`` their eigenvalues, in (0, 1).
        - ``HDM``: ``(κ, k)`` coordinates of every point, ``eigvecs`` scaled by ``eigvals ** t``.
        - ``HBDM``: ``(n, k²)`` embedding of every object.
        - ``HBDD``: ``(n, n)`` distances between the objects' embeddings.
        - ``offsets``: ``(n + 1,)``; object ``i``'s points are rows ``offsets[i]:offsets[i + 1]``.
        - ``config``: the config used.

    Raises
    ------
    ValueError
        An input breaks an assumption listed in :mod:`hdmaps.hdm.validation`. The message starts
        with the assumption's id, e.g. ``"D3: fiber_dists[4]: diagonal not fully stored"``.
    LookupError
        ``maps`` has no map for a pair of neighbors.

    Warns
    -----
    UserWarning
        When inputs are cast to ``config.dtype``, and when the joint graph is disconnected or
        nearly so (eigenvalue 1 is not simple).

    Notes
    -----
    Differences from the paper:

    - Diffusion within an object is kept: the diagonal blocks of the joint kernel are the fiber
      kernels, where the paper sets them to 0 (Def. 5.1).
    - The joint kernel is normalized with Sinkhorn instead of alpha-normalization (Def. 5.1(2)).

    Soft maps: point ``x`` of object ``i`` is compared with point ``y`` of object ``j`` through the
    map-weighted average of the distances from ``x``'s targets to ``y``, ``exp(-(M @ F)² / ε²)``.
    For a map with one target per row, this is the paper's kernel. A pair is only linked if every
    target has a stored distance to ``y``.

    References
    ----------
    T. Gao, "The diffusion geometry of fibre bundles: Horizontal diffusion maps", Applied and
    Computational Harmonic Analysis 50 (2021) 147–215. https://doi.org/10.1016/j.acha.2019.08.001
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
