
import numpy as np
import scipy.sparse as sp

from hdmaps.mapping import MapBundle
from hdmaps.types import Indexable

from . import backend
from .utils import (
    HDMConfig,
    HDMResult,
    cast,
    get_sizes,
    resolve,
)
from .validation import check_connected, validate_inputs


def run_hdm(
    config: HDMConfig,
    base_dist: sp.csr_matrix,
    maps: MapBundle,
    fiber_dists: Indexable,
) -> HDMResult:
    """Compute horizontal diffusion maps (HDM) and horizontal base diffusion maps (HBDM).

    Combines the base distances and, through the maps, the fiber distances into one squared
    distance between the points of neighboring objects, turns it into a joint kernel over all
    points, normalizes it with Sinkhorn, and embeds it with its leading eigenvectors.

    Parameters
    ----------
    config : HDMConfig
        Parameters. ``epsilon`` is the kernel bandwidth and ``gamma`` the weight of the fiber
        distances against the base distance. Either may be None to estimate it: ``epsilon`` as the
        median non-zero base distance, ``gamma`` as ``(epsilon / f)²`` with ``f`` the median over
        the fibers of each fiber's median non-zero distance.
    base_dist : scipy.sparse.csr_matrix
        ``(n, n)`` distances between the ``n`` objects. Stored entries are the neighbors; the zero
        diagonal must be stored.
    maps : MapBundle
        Correspondences between objects. ``maps[i, j]`` is a ``(κᵢ, κⱼ)`` CSR matrix whose rows
        sum to 1, needed in both directions for every pair of neighbors, and from every object to
        itself: ``maps[i, i]``, normally the identity.
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
        - ``offsets``: ``(n + 1,)``; object ``i``'s points are rows ``offsets[i]:offsets[i + 1]``.
        - ``config``: the config used.

    Raises
    ------
    ValueError
        An input breaks an assumption listed in :mod:`hdmaps.hdm.validation`. The message starts
        with the assumption's id, e.g. ``"D3: fiber_dists[4]: diagonal not fully stored"``. This
        includes a joint graph that is not connected (S2), found before the eigenvectors are computed.
    LookupError
        ``maps`` has no map for a pair of neighbors.

    Warns
    -----
    UserWarning
        When inputs are cast to ``config.dtype``.

    Notes
    -----

    Point ``x`` of object ``i`` and point ``y`` of a neighboring object ``j`` are at squared
    distance

        d²(x, y) = d_B(i, j)² + γ/2 · (Σₜ Mᵢⱼ[x, t] · Fⱼ[t, y]² + Σₛ Mⱼᵢ[y, s] · Fᵢ[s, x]²)

    with ``d_B`` the base distance, ``M`` the maps and ``F`` the fiber distances: the base distance
    plus, in both directions, the map-weighted average of the squared fiber distances from one
    point's targets to the other point. The kernel is ``exp(-d² / ε²)``.

    ``gamma`` weighs the fibers against the base. The larger it is, the more a mismatch within the
    fibers counts, so the kernel concentrates on the pairs that the maps bring close together.

    """

    validate_inputs(config, base_dist, maps, fiber_dists)
    [base_dist] = cast([base_dist], config.dtype, "base_dist")
    fiber_dists = cast([fiber_dists[i] for i in range(len(fiber_dists))], config.dtype, "fiber_dists")
    config = resolve(config, base_dist, fiber_dists)

    num_data_samples, sizes = get_sizes(fiber_dists)
    offsets = np.cumsum([0, *sizes])

    if config.verbose:
        print("Computing HDM Embedding...")

    total_distance_matrix_sq = backend.build_total_distance_matrix(
        config, base_dist, fiber_dists, maps, offsets
    )

    if config.verbose:
        print("Construct Squared Total Distance Matrix: Done.")

    coupled_kernel_matrix = backend.compute_kernel_matrix(
        config, total_distance_matrix_sq
    )
    check_connected(coupled_kernel_matrix)

    if config.verbose:
        print("Construct Joint Kernel Matrix: Done.")

    result = backend.compute_spectral_embedding(config, coupled_kernel_matrix, offsets, num_data_samples)
    if config.verbose:
        print("Spectral embedding: Done.")

    return result
