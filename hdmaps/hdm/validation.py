"""
Assumptions run_hdm makes about its inputs, from the HDM paper or required by our implementation.

Quotes are from T. Gao, "The diffusion geometry of fibre bundles: Horizontal diffusion maps",
Applied and Computational Harmonic Analysis 50 (2021) 147–215, https://doi.org/10.1016/j.acha.2019.08.001.
Lines starting with → are our reading.

Terms:
    base edge    (i, j) where base_dist stores an entry.
    base graph   one vertex per object, one edge per base edge (the paper's G_B).
    joint graph  one vertex per fiber point, edges where the joint kernel is non-zero (the paper's G).

Distances (base_dist and every fiber_dists[i]):
    D1  A square scipy sparse CSR matrix (csr_matrix or csr_array).
        Why: missing distances are non-stored entries, which a dense matrix cannot express.
    D2  Entries are finite and >= 0. A missing distance is a non-stored entry, never inf.
        Sec. 5.2: "Hᵢⱼ (the (i, j)-th block of H) is non-zero only if the sample points ξᵢ, ξⱼ are each
        among the K_B-nearest neighbors of the other"
        → a pair without a distance has no edge, so it is simply not stored.
    D3  The diagonal is stored explicitly and is 0.
        Why: we allow diffusion within an object, so every point needs an edge to itself (Note: this is a deviation from the paper).
    D4  No duplicate entries.
        Why: whether a mapped pair has every distance it needs is decided by counting stored entries,
        so a duplicate counts twice. scipy also sums duplicate values, silently doubling a distance.

Maps:
    M1  maps[i, j] and maps[j, i] exist for every base edge (i, j) (for i = j it does not need to exist since we assume identity).
        Why: both are fetched to build the joint kernel; maps outside base edges are never used.
    M2  A scipy sparse CSR matrix of shape (len(fiber i), len(fiber j)).
    M3  Entries are >= 0, and every row sums to 1.
        Sec. 6: "a transport plan matrix wᵢⱼ, the s-th row of which records the transition probability
        from vertex xᵢ,ₛ of Sᵢ to each vertex on Sⱼ"
        → every row is a probability distribution.
    M4  No explicitly stored zeros.
        Why: a stored entry counts as a map target, so a pair is dropped when that target lacks a distance.
    M5  No duplicate entries.
        Why: as D4.

Structure:
    S1  len(fiber_dists) == len(maps.data) == base_dist.shape[0], and every fiber has >= 1 point.
        Sec. 3.1(1): "The total data set X can be partitioned into a collection of data objects X₁, ⋯, Xₙ"
    S2  The joint graph is connected.
        Sec. 3.1(3): "Without loss of generality, assume G is connected."
        → so the base graph is connected too.
        Checked only after the eigensolve: a second eigenvalue of 1 gives a warning, not an error.

Config:
    C1  base_epsilon and fiber_epsilon are each None or > 0; when None, the estimated value must be > 0.
        Def. 5.1(1): "For ε > 0, δ > 0"
        → ε and δ are the base and fiber bandwidths.
    C2  t > 0.
        Sec. 3.2: "for any fixed diffusion time t ∈ R>0"
    C3  num_eigenvectors >= 1 and num_eigenvectors + 1 < the total number of fiber points.
        Sec. 3.2: "0 = λ₀ < λ₁ ≤ λ₂ ≤ ⋯ ≤ λκ−1"
        → at most κ − 1 non-trivial eigenvalues.
        Why the + 1: we ask eigsh for num_eigenvectors + 1 eigenpairs, and it requires k < n.

Deliberately not assumed:
    - Symmetric distances: base and fiber kernels are symmetrized.
    - Consistent maps (maps[j, i] == maps[i, j].T): both directions are averaged.
    - Equal fiber sizes.

Differs from the paper:
    - Diagonal blocks: the paper sets them to 0 (Def. 5.1); we keep diffusion within an object.
    - Normalization: the paper uses alpha-normalization (Def. 5.1(2)); we use Sinkhorn.
"""

import numpy as np
import scipy.sparse as sp

from hdmaps.mappings import MapBundle
from hdmaps.types import Indexable

from .utils import HDMConfig, get_sizes


def check_distances(D, name: str) -> None:
    if not (sp.issparse(D) and D.format == "csr"):
        raise ValueError(f"D1: {name}: expected a scipy sparse CSR matrix, got {type(D).__name__}")
    n, m = D.shape
    if n != m:
        raise ValueError(f"D1: {name}: expected a square matrix, got shape {D.shape}")
    if not (np.isfinite(D.data) & (D.data >= 0)).all():
        raise ValueError(f"D2: {name}: entries must be finite and >= 0")
    canonical = D.copy()
    canonical.sum_duplicates()
    if canonical.nnz != D.nnz:
        raise ValueError(f"D4: {name}: {D.nnz - canonical.nnz} duplicate entries")
    on_diagonal = np.repeat(np.arange(n), np.diff(D.indptr)) == D.indices
    if on_diagonal.sum() != n:
        raise ValueError(f"D3: {name}: diagonal not fully stored, {on_diagonal.sum()} of {n} entries")
    if (D.data[on_diagonal] != 0).any():
        raise ValueError(f"D3: {name}: diagonal must be 0")


def check_structure(base_dist: sp.csr_matrix, maps: MapBundle, fiber_dists: Indexable[sp.csr_matrix]) -> None:
    assert base_dist.shape is not None
    n = base_dist.shape[0]
    if len(fiber_dists) != n:
        raise ValueError(f"S1: fiber_dists: expected {n} fibers, one per row of base_dist, got {len(fiber_dists)}")
    if len(maps.data) != n:
        raise ValueError(f"S1: maps: expected data for {n} objects, one per row of base_dist, got {len(maps.data)}")
    for i in range(n):
        shape = fiber_dists[i].shape
        assert shape is not None
        if shape[0] == 0:
            raise ValueError(f"S1: fiber_dists[{i}]: fiber has no points")


def validate_inputs(
    config: HDMConfig, base_dist: sp.csr_matrix, maps: MapBundle, fiber_dists: Indexable[sp.csr_matrix]
) -> None:
    check_distances(base_dist, "base_dist")
    for i in range(len(fiber_dists)):
        check_distances(fiber_dists[i], f"fiber_dists[{i}]")
    check_structure(base_dist, maps, fiber_dists)
    check_config(config, sum(get_sizes(fiber_dists)[1]))


def check_map(M, name: str, shape: tuple[int, int]) -> None:
    if not (sp.issparse(M) and M.format == "csr"):
        raise ValueError(f"M2: {name}: expected a scipy sparse CSR matrix, got {type(M).__name__}; convert it with .tocsr()")
    if M.shape != shape:
        raise ValueError(f"M2: {name}: expected shape {shape}, got {M.shape}")
    canonical = M.copy()
    canonical.sum_duplicates()
    if canonical.nnz != M.nnz:
        raise ValueError(f"M5: {name}: {M.nnz - canonical.nnz} duplicate entries")
    if (M.data == 0).any():
        raise ValueError(f"M4: {name}: stored zeros; remove them with .eliminate_zeros()")
    if not (M.data >= 0).all():
        raise ValueError(f"M3: {name}: entries must be >= 0")
    row_sums = np.asarray(M.sum(axis=1)).ravel()
    if (np.abs(row_sums - 1) > 0.01).any():
        raise ValueError(f"M3: {name}: every row must sum to 1, got row sums between {row_sums.min():.3g} and {row_sums.max():.3g}")


def check_config(config: HDMConfig, num_points: int) -> None:
    for name in ("base_epsilon", "fiber_epsilon"):
        eps = getattr(config, name)
        if eps is not None and not eps > 0:
            raise ValueError(f"C1: {name}: must be None or > 0, got {eps}")
    if not config.t > 0:
        raise ValueError(f"C2: t: must be > 0, got {config.t}")
    if not 1 <= config.num_eigenvectors < num_points - 1:
        raise ValueError(
            f"C3: num_eigenvectors: must be between 1 and {num_points - 2} for {num_points} points, "
            f"got {config.num_eigenvectors}"
        )
