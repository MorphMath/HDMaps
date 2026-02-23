import numpy as np
import scipy.sparse as sparse
from numpy import inf
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
)
from scipy.spatial import KDTree

from .utils import HDMConfig, HDMResult


def gaussian_kernel(dist, eps):
    return np.exp(-(dist**2) / eps)


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
        fiber_trees[i].query(samples[i], k=fiber_knn, workers=-1)
        for i in range(num_samples)
    ]
    fiber_dists, fiber_idxs = zip(*fiber_query)
    if config.fiber_epsilon is None:
        fiber_eps = np.mean([np.median(fd) for fd in fiber_dists])
    else:
        fiber_eps = config.fiber_epsilon
    fiber_kerns = [gaussian_kernel(fiber_dist, fiber_eps) for fiber_dist in fiber_dists]

    return fiber_kerns, fiber_idxs


def compute_joint_kernel(
    config,
    base_kern: np.ndarray,
    base_idx: np.ndarray,
    fiber_kerns: list[np.ndarray],
    fiber_idxs: list[np.ndarray],
    maps,
) -> coo_matrix:

    block_sizes = [len(fk) for fk in fiber_kerns]
    delimit = np.cumsum([0] + block_sizes)

    num_samples = len(fiber_kerns)
    num_points = delimit[-1]
    fiber_knn = config.fiber_knn

    rows = []
    cols = []
    data = []

    for i in range(num_samples):
        for j in range(config.base_knn):
            neighbor_idx = int(base_idx[i, j])
            base_val = base_kern[i, j]

            i_offset = delimit[i]

            neighbor_size = block_sizes[neighbor_idx]
            neighbor_offset = delimit[neighbor_idx]

            if i == neighbor_idx:
                soft_map = sparse.eye(neighbor_size, format="csr")
            else:
                soft_map = maps[i, neighbor_idx]

            mapped_vals = soft_map @ fiber_kerns[neighbor_idx]
            mapped_vals = mapped_vals.reshape(-1)

            mapped_idxs = fiber_idxs[neighbor_idx]

            mapped_row = np.tile(np.arange(neighbor_size)[:, None], fiber_knn).reshape(
                -1
            )
            mapped_col = mapped_idxs.reshape(-1)

            rows.append(mapped_row + i_offset)
            cols.append(mapped_col + neighbor_offset)
            data.append(mapped_vals * base_val)

            rows.append(mapped_col + neighbor_offset)
            cols.append(mapped_row + i_offset)
            data.append(mapped_vals * base_val)

        if config.verbose:
            print(f"Sample {i + 1}/{num_samples} done")

    kern = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(num_points, num_points),
    )

    return kern


def eigendecomposition(
    config,
    matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform eigendecomposition on a sparse matrix."""
    tol = 1e-6
    k = config.num_eigenvectors
    which = "LM"

    eigvals, eigvecs = sparse.linalg.eigsh(matrix, k=k + 1, which=which, tol=tol)

    # Sort in descending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    return eigvals, eigvecs


def normalize(joint_kernel):
    sqrt_inv_D = 1 / np.sqrt(joint_kernel.sum(axis=1).A1)
    sqrt_inv_D[sqrt_inv_D == inf] = 0
    sqrt_inv_D = sparse.diags(sqrt_inv_D)

    kern = sqrt_inv_D @ joint_kernel @ sqrt_inv_D
    kern = (kern + kern.T) / 2

    return kern, sqrt_inv_D


def spectral_embedding(
    config: HDMConfig,
    joint_kernel: csr_matrix,
    block_sizes: list[int],
) -> HDMResult:

    num_samples = len(block_sizes)
    delimit = np.cumsum([0] + list(block_sizes))
    num_eig = config.num_eigenvectors

    normalized_kernel, sqrt_inv_D = normalize(joint_kernel)

    # 1. Eigendecomposition with explicit sorting
    eigvals, eigvecs = sparse.linalg.eigsh(
        normalized_kernel, k=num_eig + 1, which="LM", tol=1e-6
    )

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals = eigvals[1 : num_eig + 1]
    eigvecs = eigvecs[:, 1 : num_eig + 1]

    coords = sqrt_inv_D @ eigvecs
    coords = coords * eigvals

    triu_idx = np.triu_indices(num_eig)
    hbdm = np.zeros((num_samples, len(triu_idx[0])))

    for j in range(num_samples):
        start, end = delimit[j], delimit[j + 1]
        block = coords[start:end, :]

        norms = np.linalg.norm(block, axis=0)
        norms[norms == 0] = 1  # Avoid division by zero
        block = block / norms[None, :]

        block = block * np.sqrt(eigvals)

        X_j = (block.T @ block)[triu_idx]
        hbdm[j] = X_j

    return HDMResult(
        eigvals=eigvals,
        eigvecs=eigvecs,
        hdm_coords=coords,
        hbdm_coords=hbdm,
    )
