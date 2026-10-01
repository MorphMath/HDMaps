import warnings

import numpy as np
import scipy.sparse as sp
import torch

from hdmaps.mappings import MapBundle
from hdmaps.types import Indexable

from .utils import (
    HDMConfig,
    HDMResult,
    _is_cuda,
    approx_base_eps,
    torch_dtype,
)

# torch's notices about its beta sparse CSR support; _normalize checks its tensor with check_invariants=True
warnings.filterwarnings("ignore", "Sparse (invariant checks are implicitly disabled|CSR tensor support is in beta)")


def apply_kernel(dist: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-(dist**2) / eps ** 2)


def symmetrize(A):
    return (A + A.T) * 0.5


def build_base_kernel(config: HDMConfig, base_dist: sp.csr_matrix) -> sp.csr_matrix:
    _assert_diagonal_stored(base_dist, "base")

    base_epsilon = config.base_epsilon
    if base_epsilon is None:
        base_epsilon = approx_base_eps(base_dist)

    base_kernel = base_dist.astype(config.dtype)
    base_kernel.data = apply_kernel(base_kernel.data, base_epsilon)

    return sp.csr_matrix(symmetrize(base_kernel))


def _ones(A: sp.csr_matrix) -> sp.csr_matrix:
    ones = A.copy()
    ones.data[:] = 1
    return ones


def _mapped_fiber_kernel(M, F: sp.csr_matrix, weight: float, eps: float) -> sp.csr_matrix:
    kernel = M @ F
    kernel.data = apply_kernel(kernel.data, eps)
    routes = sp.csr_matrix(_ones(M) @ _ones(F))
    zero_dists = _ones(routes) - _ones(kernel)
    covered = routes.copy()
    covered.data = (routes.data == np.repeat(np.diff(M.indptr), np.diff(routes.indptr))).astype(routes.dtype)
    return (kernel + zero_dists).multiply(covered) * weight


def _assert_diagonal_stored(sparse_mat: sp.csr_matrix, label: str = "") -> None:
    assert sparse_mat.shape is not None
    m = sparse_mat.copy()
    m.sum_duplicates()
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    if (rows == m.indices).sum() != m.shape[0]:
        raise ValueError(f"{label}: diagonal not fully stored")


def _assert_diagonals_stored_list(mats: Indexable[sp.csr_matrix], label: str = "") -> None:
    for j in range(len(mats)):
        _assert_diagonal_stored(mats[j], f"{label} {j}")


def _block_row(blocks: np.ndarray, js: np.ndarray, offsets: np.ndarray, height: int) -> sp.csr_matrix:
    row = sp.csr_matrix(sp.hstack(list(blocks), format="csr"))  # compact: only the neighbor columns
    cols = np.concatenate([np.arange(int(offsets[j]), int(offsets[j + 1])) for j in js])
    return sp.csr_matrix((row.data, cols[row.indices], row.indptr), shape=(height, offsets[-1]))


def _combine_blocks(blocks: np.ndarray, base_kernel: sp.csr_matrix, offsets: np.ndarray) -> sp.csr_matrix:
    rows = []
    for i in range(len(offsets) - 1):
        js = base_kernel.indices[base_kernel.indptr[i] : base_kernel.indptr[i + 1]]  # neighbors of i
        rows.append(_block_row(blocks[i, js], js, offsets, offsets[i + 1] - offsets[i]))
    return sp.csr_matrix(sp.vstack(rows, format="csr"))


def build_horizontal_diffusion_matrix(
    config: HDMConfig,
    maps: MapBundle,
    base_kernel: sp.csr_matrix,
    fiber_dists: Indexable[sp.csr_matrix],
    offsets: np.ndarray,
) -> sp.csr_matrix:
    fiber_epsilon = config.fiber_epsilon
    if fiber_epsilon is None:
        raise ValueError("fiber_epsilon must be set")

    _assert_diagonals_stored_list(fiber_dists, "fiber")

    num_data_samples = len(offsets) - 1
    blocks = np.full((num_data_samples, num_data_samples), None, dtype=object)

    upper = sp.triu(base_kernel, k=1).tocoo()
    for i, j, v in zip(upper.row, upper.col, upper.data):
        forth = _mapped_fiber_kernel(maps[i, j], fiber_dists[j], v, fiber_epsilon)
        back = _mapped_fiber_kernel(maps[j, i], fiber_dists[i], v, fiber_epsilon)
        block = (forth + back.T) * 0.5
        blocks[i, j] = block.tocsr()
        blocks[j, i] = block.T.tocsr()

    for i in range(num_data_samples):
        K = fiber_dists[i].astype(config.dtype)
        K.data = apply_kernel(K.data, fiber_epsilon)
        blocks[i, i] = sp.csr_matrix(symmetrize(K))

    return _combine_blocks(blocks, base_kernel, offsets)


def _normalize(config: HDMConfig, W: sp.csr_matrix) -> sp.csr_matrix:
    assert W.shape is not None
    n = W.shape[0]
    I = sp.eye(n, dtype=W.dtype, format="csr")
    W = W + config.sinkhorn_jitter * I if config.sinkhorn_jitter > 0 else W.copy()
    W.sort_indices()
    W_torch = torch.sparse_csr_tensor(
        *map(torch.from_numpy, (W.indptr, W.indices, W.data)), size=W.shape, check_invariants=True
    ).to(config.device)
    d = torch.ones(n, dtype=W_torch.dtype, device=W_torch.device)
    for _ in range(config.sinkhorn_max_iter):
        Wd = W_torch @ d
        if (d * Wd - 1).abs().max() < config.sinkhorn_tol:
            break
        d = torch.sqrt(d / Wd)
    else:
        if config.verbose:
            print(f"Sinkhorn iteration did not converge after {config.sinkhorn_max_iter} iterations")
    D = sp.diags(d.cpu().numpy(), format="csr")
    Q = D @ W @ D
    return 0.5 * (Q + I)


def _eigsh_scipy(
    config: HDMConfig,
    kernel: sp.csr_matrix,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert kernel.shape is not None
    n = kernel.shape[0]
    rng = np.random.default_rng(config.seed)
    v0 = rng.random(n, dtype=config.dtype)

    eigvals, eigvecs = sp.linalg.eigsh(kernel, k=k + 1, which="LM", tol=config.eig_tol, v0=v0)

    idx = np.argsort(eigvals)[::-1]
    eigvals = 2 * eigvals[idx] - 1
    eigvecs = eigvecs[:, idx]
    return (
        torch.as_tensor(eigvals, dtype=torch_dtype(config.dtype), device=config.device),
        torch.as_tensor(eigvecs, dtype=torch_dtype(config.dtype), device=config.device),
    )


def _eigsh_cupy(
    config: HDMConfig,
    kernel: sp.csr_matrix,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import cupy as cp  # type: ignore[import-not-found]
    import cupyx.scipy.sparse as cpsp  # type: ignore[import-not-found]
    import cupyx.scipy.sparse.linalg as cpx_linalg  # type: ignore[import-not-found]

    kernel = cpsp.csr_matrix(kernel)

    assert kernel.shape is not None
    n = kernel.shape[0]
    v0 = cp.asarray(np.random.default_rng(config.seed).random(n, dtype=kernel.dtype))

    eigvals_cp, eigvecs_cp = cpx_linalg.eigsh(kernel, k=k + 1, which="LM", tol=config.eig_tol, v0=v0)
    eigvals = 2 * torch.from_dlpack(eigvals_cp) - 1
    eigvecs = torch.from_dlpack(eigvecs_cp)

    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx]


def compute_spectral_embedding(
    config: HDMConfig,
    joint_kernel: sp.csr_matrix,
    offsets: np.ndarray,
    num_data_samples: int,
) -> HDMResult:
    num_eig = config.num_eigenvectors

    normalized_kernel = _normalize(config, joint_kernel)

    if _is_cuda(config.device):
        vals, V = _eigsh_cupy(config, normalized_kernel, num_eig)
    else:
        vals, V = _eigsh_scipy(config, normalized_kernel, num_eig)

    if vals[0] >= 1 - config.eig_tol:
        raise ValueError("graph is disconnected: eigenvalue 1 has multiplicity > 1")

    vals = vals[1 : num_eig + 1]
    num_pos = int((vals > 0).sum())
    if num_pos != num_eig:
        raise ValueError(f"only {num_pos} of {num_eig} eigenvalues are positive; lower num_eigenvectors")
    V = V[:, 1 : num_eig + 1]

    HDM = V * (vals ** config.t)

    V_scaled = (vals ** (config.t / 2)) * V

    HBDM = torch.zeros((num_data_samples, num_eig**2), dtype=V.dtype, device=V.device)

    for i in range(num_data_samples):
        block = V_scaled[offsets[i] : offsets[i + 1]]
        HBDM[i] = (block.T @ block).ravel()

    HBDD = torch.cdist(HBDM, HBDM)

    return HDMResult(
        V.cpu().numpy(),
        vals.cpu().numpy(),
        HDM.cpu().numpy(),
        HBDM.cpu().numpy(),
        HBDD.cpu().numpy(),
        offsets,
        config,
    )
