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
    resolve_epsilon,
    torch_dtype,
    warn,
    warn_cast,
)
from .validation import validate_map_and_distances

# torch's notices about its beta sparse CSR support; _normalize checks its tensor with check_invariants=True
warnings.filterwarnings("ignore", "Sparse (invariant checks are implicitly disabled|CSR tensor support is in beta)")


def rbf_kernel(d_sq: np.ndarray, epsilon: float) -> np.ndarray:
    return np.exp(-d_sq / epsilon**2)


def _symmetrize(A):
    return (A + A.T) * 0.5


def _ones(A: sp.csr_matrix) -> sp.csr_matrix:
    ones = A.copy()
    ones.data[:] = 1
    return ones


def _mapped_fiber_kernel(M, F: sp.csr_matrix, weight: float, eps: float) -> sp.csr_matrix:
    kernel = M @ F
    # kernel.data = _apply_kernel(kernel.data, eps)
    routes = sp.csr_matrix(_ones(M) @ _ones(F))
    zero_dists = _ones(routes) - _ones(kernel)
    covered = routes.copy()
    covered.data = (routes.data == np.repeat(np.diff(M.indptr), np.diff(routes.indptr))).astype(routes.dtype)
    return (kernel + zero_dists).multiply(covered) * weight


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

    upper = sp.triu(base_dists).tocoo()
    for i, j, d_b in zip(upper.row, upper.col, upper.data):
        M_ij, M_ji = fetch(i, j), fetch(j, i)
        F_i2, F_j2 = fiber_dists[i].power(2), fiber_dists[j].power(2)

        dist = (config.gamma / 2 * (M_ij @ F_j2 + (M_ji @ F_i2).T)).tocoo()
        dist.data += d_b ** 2

        r, c = dist.row + offsets[i], dist.col + offsets[j]
        rows += [r, c]
        cols += [c, r]
        vals += [dist.data, dist.data]

    warn_cast("maps", cast_from, config.dtype)

    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
        dtype=config.dtype,
    )


def compute_kernel_matrix(config, total_distance_matrix, kernel_fn=rbf_kernel):
    K = total_distance_matrix.copy()
    K.data = kernel_fn(K.data, config.epsilon)
    return K


def build_horizontal_diffusion_matrix(
    config: HDMConfig,
    maps: MapBundle,
    base_kernel: sp.csr_matrix,
    fiber_dists: Indexable[sp.csr_matrix],
    offsets: np.ndarray,
) -> sp.csr_matrix:
    fiber_epsilon = resolve_epsilon(config.fiber_epsilon, fiber_dists, "fiber_epsilon")

    num_data_samples = len(offsets) - 1
    sizes = np.diff(offsets).tolist()
    blocks = np.full((num_data_samples, num_data_samples), None, dtype=object)

    cast_from: set[str] = set()

    def fetch(i: int, j: int) -> sp.csr_matrix:
        M = maps[i, j]
        validate_map_and_distances(M, fiber_dists[j], f"maps[{i}, {j}]", (sizes[i], sizes[j]))
        if M.dtype != config.dtype:
            cast_from.add(str(M.dtype))
        return sp.csr_matrix(M.astype(config.dtype, copy=False))

    upper = sp.triu(base_kernel).tocoo()
    for i, j, v in zip(upper.row, upper.col, upper.data):
        M_ij, M_ji = fetch(i, j), fetch(j, i)
        forth = _mapped_fiber_kernel(M_ij, fiber_dists[j], v, fiber_epsilon)
        back = _mapped_fiber_kernel(M_ji, fiber_dists[i], v, fiber_epsilon)
        block = (forth + back.T) * 0.5
        blocks[i, j] = block.tocsr()
        blocks[j, i] = block.T.tocsr()

    warn_cast("maps", cast_from, config.dtype)

    for i in range(num_data_samples):
        K = fiber_dists[i].astype(config.dtype)
        K.data = _apply_kernel(K.data, fiber_epsilon)
        blocks[i, i] = sp.csr_matrix(_symmetrize(K))

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

def sinkhorn_step(d, w_inv, K):
    r = (K @ (w_inv / d)) / d
    return K @ (w_inv / (d * torch.sqrt(r))), r


def _normalize(config: HDMConfig, K: sp.csr_matrix) -> sp.csr_matrix:
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

    D_inv = sp.diags((torch.sqrt(w_inv) / d).cpu().numpy())
    return D_inv @ K @ D_inv


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

    normalized_kernel = _normalize(config, joint_kernel)

    if _is_cuda(config.device):
        vals, V = _eigsh_cupy(config, normalized_kernel, num_eig)
    else:
        vals, V = _eigsh_scipy(config, normalized_kernel, num_eig)

    vals = vals[1 : num_eig + 1]
    V = V[:, 1 : num_eig + 1]

    if vals[0] >= 1 - config.eig_tol:
        warn(
            f"eigenvalue 1 has multiplicity > 1 (second eigenvalue {float(vals[0]):.8g}): "
            "the graph is disconnected or nearly so"
        )

    num_pos = int((vals > 0).sum())
    if num_pos < num_eig:
        raise ValueError(f"only {num_pos} of {num_eig} eigenvalues are positive; lower num_eigenvectors")

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
