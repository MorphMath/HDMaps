import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import KDTree

from .utils import HDMConfig, HDMResult


def _is_cuda(device) -> bool:
    try:
        return torch.device(device).type == "cuda"
    except Exception:
        return False


def gaussian_kernel(dist: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-(dist**2) / eps)


def compute_base_kernel(config: HDMConfig, samples):
    base_knn = config.base_knn

    base_coords = np.concatenate([s.reshape(1, -1) for s in samples], axis=0)
    base_tree = KDTree(base_coords)
    base_dists, base_idx = base_tree.query(base_coords, k=base_knn, workers=-1)

    base_eps = np.median(base_dists) if config.base_epsilon is None else config.base_epsilon
    base_kern = gaussian_kernel(base_dists, base_eps)

    return base_kern, base_idx

def compute_base_kernel_with_precomputed_distances(config: HDMConfig, D):
    base_idx = np.argsort(D, axis=1)[:, :config.base_knn]
    base_dists = np.take_along_axis(D, base_idx, axis=1)

    base_eps = np.median(base_dists) if config.base_epsilon is None else config.base_epsilon
    base_kern = gaussian_kernel(base_dists, base_eps)

    return base_kern, base_idx


def compute_fiber_kernels(config: HDMConfig, samples):
    fiber_knn = config.fiber_knn
    num_samples = len(samples)

    fiber_trees = [KDTree(s) for s in samples]
    fiber_query = [fiber_trees[i].query(samples[i], k=fiber_knn, workers=-1) for i in range(num_samples)]
    fiber_dists, fiber_idxs = zip(*fiber_query)

    fiber_eps = (
        np.mean([np.median(fd) for fd in fiber_dists])
        if config.fiber_epsilon is None
        else config.fiber_epsilon
    )
    fiber_kerns = [gaussian_kernel(fd, fiber_eps) for fd in fiber_dists]

    return fiber_kerns, fiber_idxs


def compute_joint_kernel(
    config,
    base_kern: np.ndarray,
    base_idx: np.ndarray,
    fiber_kerns: list[np.ndarray],
    fiber_idxs: list[np.ndarray],
    maps,
) -> torch.Tensor:
    block_sizes = [len(fk) for fk in fiber_kerns]
    delimit = np.cumsum([0] + block_sizes)

    num_samples = len(fiber_kerns)
    num_points = delimit[-1]
    fiber_knn = config.fiber_knn

    rows_list = []
    cols_list = []
    data_list = []

    for i in range(num_samples):
        for j in range(config.base_knn):
            neighbor_idx = int(base_idx[i, j])
            base_val = float(base_kern[i, j])

            i_offset = delimit[i]
            neighbor_size = block_sizes[neighbor_idx]
            neighbor_offset = delimit[neighbor_idx]

            if i == neighbor_idx:
                soft_map = sp.eye(neighbor_size, format="csr")
            else:
                soft_map = maps[i, neighbor_idx]

            # Build sparse fiber kernel for neighbor
            fk_rows = np.repeat(np.arange(neighbor_size), fiber_knn)
            fk_cols = fiber_idxs[neighbor_idx].reshape(-1)
            fk_data = fiber_kerns[neighbor_idx].reshape(-1)
            fiber_kern_sparse = sp.csr_matrix(
                (fk_data, (fk_rows, fk_cols)), shape=(neighbor_size, neighbor_size)
            )

            # Mapped kernel: soft_map @ fiber_kernel, then extract sparse entries
            mapped_kern = (soft_map @ fiber_kern_sparse).tocoo()

            rows_list.append(mapped_kern.row + i_offset)
            cols_list.append(mapped_kern.col + neighbor_offset)
            data_list.append(mapped_kern.data * base_val)

            rows_list.append(mapped_kern.col + neighbor_offset)
            cols_list.append(mapped_kern.row + i_offset)
            data_list.append(mapped_kern.data * base_val)

        if config.verbose:
            print(f"Sample {i + 1}/{num_samples} done")

    rows = torch.from_numpy(np.concatenate(rows_list)).long()
    cols = torch.from_numpy(np.concatenate(cols_list)).long()
    data = torch.from_numpy(np.concatenate(data_list).astype(np.float64))
    

    kern = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), data, (num_points, num_points), dtype=torch.float64
    )
    return kern.coalesce()


def _normalize(joint_kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """D^{-1/2} K D^{-1/2}, symmetrized. Returns (normalized_kernel, sqrt_inv_D)."""
    kern = joint_kernel.coalesce()
    indices = kern.indices()  # (2, nnz)
    values = kern.values()   # (nnz,)

    # Row sums via scatter
    row_sums = torch.zeros(kern.shape[0], dtype=torch.float64, device=kern.device)
    row_sums.scatter_add_(0, indices[0], values)

    sqrt_inv_D = torch.rsqrt(row_sums)
    sqrt_inv_D[~torch.isfinite(sqrt_inv_D)] = 0.0

    # Element-wise D^{-1/2} scaling on COO values
    scaled = values * sqrt_inv_D[indices[0]] * sqrt_inv_D[indices[1]]

    # Symmetrize: (K + K^T) / 2 by concatenating transposed entries
    all_indices = torch.cat([indices, indices.flip(0)], dim=1)
    all_values = torch.cat([scaled, scaled]) * 0.5

    normalized = torch.sparse_coo_tensor(all_indices, all_values, kern.shape, dtype=torch.float64)
    normalized = normalized.coalesce()

    return normalized, sqrt_inv_D


def _eigsh_scipy(kernel: torch.Tensor, k: int, seed=None) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU eigendecomposition: bridge torch sparse COO → scipy CSR → eigsh."""
    csr = kernel.to_sparse_csr()
    crow = csr.crow_indices().numpy()
    col = csr.col_indices().numpy()
    vals = csr.values().numpy()
    n = csr.shape[0]

    scipy_mat = sp.csr_matrix((vals, col, crow), shape=(n, n))

    rng = np.random.default_rng(seed)
    v0 = rng.random(n)
    eigvals, eigvecs = sp.linalg.eigsh(scipy_mat, k=k + 1, which="LM", tol=1e-6, v0=v0)

    idx = np.argsort(eigvals)[::-1]
    return (
        torch.from_numpy(eigvals[idx].copy()),
        torch.from_numpy(eigvecs[:, idx].copy()),
    )


def _eigsh_cupy(
    kernel: torch.Tensor, k: int, device: str, seed=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU eigendecomposition: bridge torch sparse → CuPy CSR (DLPack) → eigsh → CPU."""
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import cupyx.scipy.sparse.linalg as cpx_linalg

    # Convert to CSR and move to the target CUDA device
    csr = kernel.to_sparse_csr().to(device)

    crow = csr.crow_indices().to(torch.int32)
    col = csr.col_indices().to(torch.int32)
    vals = csr.values()  # float64 on CUDA

    # Zero-copy bridge via DLPack
    cp_crow = cp.from_dlpack(crow)
    cp_col = cp.from_dlpack(col)
    cp_vals = cp.from_dlpack(vals)

    n = csr.shape[0]
    cp_mat = cpx_sparse.csr_matrix((cp_vals, cp_col, cp_crow), shape=(n, n))

    v0 = cp.array(np.random.default_rng(seed).random(n)) if seed is not None else None
    eigvals_cp, eigvecs_cp = cpx_linalg.eigsh(cp_mat, k=k + 1, which="LM", tol=1e-6, v0=v0)

    # Bridge back to torch and move to CPU
    eigvals = torch.from_dlpack(eigvals_cp).cpu()
    eigvecs = torch.from_dlpack(eigvecs_cp).cpu()

    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx]


def spectral_embedding(
    config: HDMConfig,
    joint_kernel: torch.Tensor,
    block_sizes: list[int],
) -> HDMResult:
    num_samples = len(block_sizes)
    delimit = np.cumsum([0] + list(block_sizes))
    num_eig = config.num_eigenvectors

    # Normalize on CPU
    normalized_kernel, sqrt_inv_D = _normalize(joint_kernel)

    # Eigendecomposition: CuPy on CUDA, scipy on CPU
    if _is_cuda(config.device):
        eigvals, eigvecs = _eigsh_cupy(normalized_kernel, num_eig, config.device, config.seed)
    else:
        eigvals, eigvecs = _eigsh_scipy(normalized_kernel, num_eig, config.seed)

    # Drop trivial leading eigenvalue
    eigvals = eigvals[1 : num_eig + 1]
    eigvecs = eigvecs[:, 1 : num_eig + 1]

    # Recover unnormalized coordinates: D^{-1/2} * eigvecs * eigvals
    coords = sqrt_inv_D[:, None] * eigvecs * eigvals[None, :]

    # HBDM: outer-product summaries per sample block
    triu_i, triu_j = torch.triu_indices(num_eig, num_eig)
    hbdm = torch.zeros(num_samples, len(triu_i), dtype=torch.float64)

    for j in range(num_samples):
        start, end = int(delimit[j]), int(delimit[j + 1])
        block = coords[start:end, :]

        norms = torch.linalg.norm(block, dim=0)
        norms[norms == 0] = 1.0
        block = block / norms[None, :]
        block = block * torch.sqrt(eigvals)[None, :]

        hbdm[j] = (block.T @ block)[triu_i, triu_j]

    return HDMResult(
        eigvals=eigvals.numpy(),
        eigvecs=eigvecs.numpy(),
        hdm_coords=coords.numpy(),
        hbdm_coords=hbdm.numpy(),
    )
