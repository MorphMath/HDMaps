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
    coo = base_dist.tocoo()
    assert not (coo.row == coo.col).any()

    base_kernel = base_dist.astype(config.dtype)

    base_epsilon = config.base_epsilon
    if base_epsilon is None:
        base_epsilon = approx_base_eps(base_dist)

    base_kernel.data = apply_kernel(base_kernel.data, base_epsilon)

    base_kernel = symmetrize(base_kernel)
    assert (np.diff(base_kernel.indptr) > 0).all()
    return base_kernel


def _ones(A: sp.csr_matrix) -> sp.csr_matrix:
    ones = A.copy()
    ones.data[:] = 1
    return ones


def _mapped_fiber_kernel(M, F: sp.csr_matrix, base_kernel, eps: float) -> sp.csr_matrix:
    kernel = M @ F
    kernel.data = apply_kernel(kernel.data, eps)
    routes = sp.csr_matrix(_ones(M) @ _ones(F))  # routes[r, c]: how many of r's map targets k have F[k, c] stored
    # M @ F drops sums that are exactly 0: zero mapped distances, whose kernel value is 1
    zero_dists = _ones(routes) - _ones(kernel)
    # keep only pairs with a stored distance from every map target; M @ F would count the missing ones as 0
    covered = routes.copy()
    covered.data = (routes.data == np.repeat(np.diff(M.indptr), np.diff(routes.indptr))).astype(routes.dtype)
    return (kernel + zero_dists).multiply(covered) * base_kernel


def _assert_diagonals_stored(fiber_dists: Indexable[sp.csr_matrix]) -> None:
    for j in range(len(fiber_dists)):
        f = fiber_dists[j]
        assert f.shape is not None
        rows = np.repeat(np.arange(f.shape[0]), np.diff(f.indptr))
        assert (rows == f.indices).sum() == f.shape[0], f"fiber {j}: diagonal not fully stored"


def _block_row(blocks: np.ndarray, js: np.ndarray, offsets: np.ndarray, height: int) -> sp.csr_matrix:
    row = sp.csr_matrix(sp.hstack(list(blocks), format="csr"))  # compact: only the neighbour columns
    cols = np.concatenate([np.arange(int(offsets[j]), int(offsets[j + 1])) for j in js])
    return sp.csr_matrix((row.data, cols[row.indices], row.indptr), shape=(height, offsets[-1]))


def _combine_blocks(blocks: np.ndarray, base_kernel: sp.csr_matrix, offsets: np.ndarray) -> sp.csr_matrix:
    rows = []
    for i in range(len(offsets) - 1):
        js = base_kernel.indices[base_kernel.indptr[i] : base_kernel.indptr[i + 1]]  # neighbours of i
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

    _assert_diagonals_stored(fiber_dists)

    num_data_samples = len(offsets) - 1
    blocks = np.full((num_data_samples, num_data_samples), None, dtype=object)

    # symmetrize per block pair: block (i, j) of (W + W.T) / 2 is (W_ij + W_ji.T) / 2
    upper = sp.triu(base_kernel, k=1).tocoo()
    for i, j, v in zip(upper.row, upper.col, upper.data):
        forth = _mapped_fiber_kernel(maps[i, j], fiber_dists[j], v, fiber_epsilon)
        back = _mapped_fiber_kernel(maps[j, i], fiber_dists[i], v, fiber_epsilon)
        block = (forth + back.T) * 0.5
        blocks[i, j] = block.tocsr()
        blocks[j, i] = block.T.tocsr()

    return _combine_blocks(blocks, base_kernel, offsets)


def _normalize(config: HDMConfig, W: sp.csr_matrix) -> sp.csr_matrix:
    assert W.shape is not None
    n = W.shape[0]
    I = sp.eye(n, dtype=W.dtype, format="csr")
    if config.sinkhorn_jitter > 0:
        W = W + config.sinkhorn_jitter * I
    W.sort_indices()  # torch CSR requires sorted column indices
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
    v0 = cp.array(np.random.default_rng(config.seed).random(n), dtype=kernel.dtype)

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

    vals = vals[1 : num_eig + 1].clamp_min(0)
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
