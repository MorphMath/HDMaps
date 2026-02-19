import numpy as np
import scipy.sparse as sparse
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
)
from scipy.spatial import KDTree

from .utils import HDMConfig, HDMResult


def gaussian_kernel(dist, eps):
    return np.exp(-(dist**2) / eps**2)


def compute_base_kernel(config: HDMConfig, samples):
    base_knn = config.base_knn

    base_coords = np.concat([sample.reshape(1, -1) for sample in samples], axis=0)
    base_tree = KDTree(base_coords)
    base_dists, base_idx = base_tree.query(base_coords, k=base_knn, workers=-1)

    if config.base_epsilon is None:
        base_eps = np.median(base_dists)
    else:
        base_eps = config.base_epsilon
    base_kern = gaussian_kernel(base_dists, base_eps)

    return base_kern, base_idx


def compute_fiber_kernels(config: HDMConfig, samples):
    num_samples = len(samples)
    fiber_knn = config.fiber_knn

    fiber_trees = [KDTree(fiber_coord) for fiber_coord in samples]
    fiber_query = [
        fiber_trees[i].query(samples[i], k=fiber_knn) for i in range(num_samples)
    ]
    fiber_dists, fiber_idxs = zip(*fiber_query)
    fiber_eps = np.mean([np.median(fd) for fd in fiber_dists])
    fiber_kerns = [gaussian_kernel(fiber_dist, fiber_eps) for fiber_dist in fiber_dists]

    return fiber_kerns, fiber_idxs


def compute_joint_kernel(
    config: HDMConfig,
    base_kern: np.ndarray,
    base_idx: np.ndarray,
    fiber_kerns: list[np.ndarray],
    fiber_idxs: list[np.ndarray],
    maps,
) -> coo_matrix:

    block_sizes = [0] + [len(fiber_kern) for fiber_kern in fiber_kerns]
    delimit = np.cumsum(block_sizes)

    num_samples = len(fiber_kerns)
    num_points = delimit[-1]
    fiber_knn = config.fiber_knn

    rows = []
    cols = []
    data = []

    for i in range(num_samples):
        for j in range(config.base_knn):
            fiber_size = block_sizes[i]
            neighbor_idx = base_idx[i, j]

            base_val = base_kern[i, j]

            soft_map = maps[i, j]
            mapped_vals = soft_map @ fiber_kerns[neighbor_idx]

            mapped_vals = mapped_vals.reshape(-1)
            mapped_idxs = fiber_idxs[neighbor_idx]

            mapped_row = np.tile(np.arange(fiber_size)[:, None], fiber_knn).reshape(-1)
            mapped_col = mapped_idxs.reshape(-1)

            i_offset = delimit[i]
            neighbor_offset = delimit[neighbor_idx]

            rows.append(mapped_row + i_offset)
            cols.append(mapped_col + neighbor_offset)
            data.append(mapped_vals * base_val)

            rows.append(mapped_col + neighbor_offset)
            cols.append(mapped_row + i_offset)
            data.append(mapped_vals * base_val)

        if config.verbose:
            print(f"Sample {i + 1}/{num_samples} done")

    kern = sparse.coo_matrix((data, (rows, cols)), shape=(num_points, num_points))

    return kern


def eigendecomposition(
    config,
    matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform eigendecomposition on a sparse matrix."""
    tol = 1e-10
    maxiter = 10000
    k = config.num_eigenvectors
    which = "LM"

    rng = np.random.default_rng(42)
    v0 = rng.random(size=matrix.shape[0]).astype(np.float64)
    eigvals, eigvecs = sparse.linalg.eigsh(
        matrix, k=k, which=which, maxiter=maxiter, tol=tol, v0=v0
    )

    # Sort in descending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    return eigvals, eigvecs


def spectral_embedding(
    config: HDMConfig,
    kernel: csr_matrix,
    inv_sqrt_diag: np.ndarray,
) -> HDMResult:

    eigvals, eigvecs = eigendecomposition(config, kernel)
    sqrt_diag = sparse.diags(inv_sqrt_diag, 0)

    renormalized_eigvecs = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
    hdm_coords = renormalized_eigvecs @ sqrt_lambda

    results = HDMResult(
        eigvals=eigvals,
        eigvecs=renormalized_eigvecs,
        hdm_coords=hdm_coords,
        hbdm_coords=None,
    )

    return results
