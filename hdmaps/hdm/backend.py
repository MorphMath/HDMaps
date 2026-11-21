import warnings

import numpy as np
import scipy.sparse as sp
import torch

from hdmaps.mapping import MapBundle
from hdmaps.types import Indexable

from .utils import (
    HDMConfig,
    HDMResult,
    _is_cuda,
    torch_dtype,
    warn_cast,
)
from .validation import validate_map_and_distances

# torch's notices about its beta sparse CSR support; _normalize checks its tensor with check_invariants=True
warnings.filterwarnings("ignore", "Sparse (invariant checks are implicitly disabled|CSR tensor support is in beta)")


def rbf_kernel(d_sq: np.ndarray, epsilon: float) -> np.ndarray:
    return np.exp(-d_sq / epsilon**2)


def _ones(A: sp.csr_matrix) -> sp.csr_matrix:
    ones = A.copy()
    ones.data[:] = 1
    return ones


def _mapped_sq_dists(M, F2):
    """Per (x, y): the map-weighted squared distance from x's targets to y, and whether every target has one."""
    covered = sp.csr_matrix(_ones(M) @ _ones(F2))  # how many of x's targets have a distance to y
    covered.data = covered.data == np.repeat(np.diff(M.indptr), np.diff(covered.indptr))
    covered.eliminate_zeros()
    return M @ F2, covered  # the product drops zero distances; covered still has them


def build_total_distance_matrix(
    config: HDMConfig,
    base_dists: sp.csr_matrix,
    fiber_dists: Indexable[sp.csr_matrix],
    maps: MapBundle,
    offsets: np.ndarray,
) -> sp.csr_matrix:

    sizes = np.diff(offsets).tolist()
    N = int(offsets[-1])
    cast_from: set[str] = set()
    rows, cols, vals = [], [], []

    def fetch(i: int, j: int) -> sp.csr_matrix:
        M = maps[i, j]
        validate_map_and_distances(M, fiber_dists[j], f"maps[{i}, {j}]", (sizes[i], sizes[j]))
        if M.dtype != config.dtype:
            cast_from.add(str(M.dtype))
        return sp.csr_matrix(M.astype(config.dtype, copy=False))

    def add(i: int, j: int, block: sp.coo_matrix, values: np.ndarray) -> None:
        r, c = block.row + offsets[i], block.col + offsets[j]
        rows.append(r)
        cols.append(c)
        vals.append(values)
        if i != j:  # the transposed block; a diagonal block is its own transpose
            rows.append(c)
            cols.append(r)
            vals.append(values)

    upper = sp.triu(base_dists).tocoo()
    for i, j, d_b in zip(upper.row, upper.col, upper.data):
        M_ij, M_ji = fetch(i, j), fetch(j, i)
        F_i2, F_j2 = fiber_dists[i].power(2), fiber_dists[j].power(2)

        forth, covered_forth = _mapped_sq_dists(M_ij, F_j2)
        back, covered_back = _mapped_sq_dists(M_ji, F_i2)
        covered = covered_forth.multiply(covered_back.T).tocoo()  # in both directions
        fiber = (config.gamma / 2 * (forth + back.T).multiply(covered)).tocoo()

        # equal entries are summed when the matrix is built, so a pair at fiber distance 0,
        # which the product dropped, still gets its base distance, since sparse matricies drops zeroes implicitly in @ operations
        add(i, j, covered, np.full(covered.nnz, d_b**2, config.dtype))
        add(i, j, fiber, fiber.data)

    warn_cast("maps", cast_from, config.dtype)

    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
        dtype=config.dtype,
    )


def compute_kernel_matrix(config, total_distance_matrix, kernel_fn=rbf_kernel):
    K = total_distance_matrix.copy()
    K.data = kernel_fn(K.data, config.epsilon)
    K.eliminate_zeros()  # a kernel value of exactly 0 is no link
    return K


def sinkhorn_step(d, w_inv, K):
    r = (K @ (w_inv / d)) / d
    return K @ (w_inv / (d * torch.sqrt(r))), r


def _normalize(config: HDMConfig, K: sp.csr_matrix) -> tuple[sp.csr_matrix,sp.csr_matrix]:
    n = K.shape[0]
    K.sort_indices()
    K_torch = torch.sparse_csr_tensor(
        *map(torch.from_numpy, (K.indptr, K.indices, K.data)), size=K.shape, check_invariants=True
    ).to(config.device)
    d = torch.ones(n, dtype=K_torch.dtype, device=K_torch.device)

    w_inv = 1 / (K_torch @ d)
    r = torch.full_like(d, torch.inf)
    i = 0

    while i < config.sinkhorn_max_iter and (r - 1).abs().max() >= config.sinkhorn_tol:
        d, r = sinkhorn_step(d, w_inv, K_torch)
        i += 1

    if i == config.sinkhorn_max_iter:
        print(f"Sinkhorn iteration did not converge after {config.sinkhorn_max_iter} iterations")

    w_inv_sqrt = torch.sqrt(w_inv)
    D_inv = sp.diags((w_inv_sqrt / d).cpu().numpy())
    return D_inv @ K @ D_inv, w_inv_sqrt


def _eigsh_scipy(
    config: HDMConfig,
    kernel: sp.csr_matrix,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert kernel.shape is not None
    n = kernel.shape[0]
    rng = np.random.default_rng(config.seed)
    v0 = rng.random(n, dtype=config.dtype)

    eigvals, eigvecs = sp.linalg.eigsh(kernel, k=k + 1, which="LA", tol=config.eig_tol, v0=v0)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
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

    eigvals_cp, eigvecs_cp = cpx_linalg.eigsh(kernel, k=k + 1, which="LA", tol=config.eig_tol, v0=v0)
    eigvals = torch.from_dlpack(eigvals_cp)
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

    normalized_kernel, w_inv_sqrt = _normalize(config, joint_kernel)

    if _is_cuda(config.device):
        vals, V = _eigsh_cupy(config, normalized_kernel, num_eig)
    else:
        vals, V = _eigsh_scipy(config, normalized_kernel, num_eig)

    vals = vals[1 : num_eig + 1]
    V = V[:, 1 : num_eig + 1]

    num_pos = int((vals > 0).sum())
    if num_pos < num_eig:
        raise ValueError(f"only {num_pos} of {num_eig} eigenvalues are positive; lower num_eigenvectors")

    V = V / w_inv_sqrt[:, None]

    HDM = V * (vals ** config.t)

    V_scaled = (vals ** (config.t / 2)) * V



    HBDM = torch.zeros((num_data_samples, num_eig**2), dtype=V.dtype, device=V.device)

    for i in range(num_data_samples):
        block = V_scaled[offsets[i] : offsets[i + 1]]
        HBDM[i] = (block.T @ block).ravel()

    return HDMResult(
        V.cpu().numpy(),
        vals.cpu().numpy(),
        HDM.cpu().numpy(),
        HBDM.cpu().numpy(),
        offsets,
        config,
    )
