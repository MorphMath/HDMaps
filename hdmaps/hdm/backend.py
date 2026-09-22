import numpy as np
import scipy.sparse as sp
import torch

from .utils import (
    HDMConfig,
    HDMResult,
    _is_cuda,
    approx_base_eps,
    torch_dtype,
)

from hdmaps.types import Indexable
from hdmaps.mappings import MapBundle


def apply_kernel(dist: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-(dist**2) / eps)


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


def _mapped_fiber_kernel(M, F: sp.csr_matrix, eps: float) -> sp.csr_matrix:
    M = sp.csr_matrix(M)
    M_pat = M.copy()
    M_pat.data = np.ones_like(M_pat.data)
    F_pat = F.copy()
    F_pat.data = np.ones_like(F_pat.data)

    pat = (M_pat @ F_pat).tocoo()
    dists = np.asarray((M @ F).tocsr()[pat.row, pat.col]).ravel()

    return sp.csr_matrix(
        (apply_kernel(dists, eps), (pat.row, pat.col)),
        shape=(M.shape[0], F.shape[1]),
    )


def build_horizontal_diffusion_matrix(
    config: HDMConfig,
    maps: MapBundle,
    base_kernel: sp.csr_matrix,
    fiber_dists: Indexable[sp.csr_matrix],
) -> sp.csr_matrix:

    fiber_epsilon = config.fiber_epsilon
    if fiber_epsilon is None:
        raise ValueError("fiber_epsilon must be set")

    num_data_samples = len(fiber_dists)

    for j in range(num_data_samples):
        f = fiber_dists[j]
        assert f.shape is not None
        rows = np.repeat(np.arange(f.shape[0]), np.diff(f.indptr))
        assert (rows == f.indices).sum() == f.shape[0], f"fiber {j}: diagonal not fully stored"

    blocks = np.full((num_data_samples, num_data_samples), None, dtype=object)
    base_coo = base_kernel.tocoo()

    for i, j, v in zip(base_coo.row, base_coo.col, base_coo.data):
        blocks[i, j] = _mapped_fiber_kernel(maps[i, j], fiber_dists[j], fiber_epsilon) * v

    W = sp.bmat(blocks.tolist(), format="csr")

    return symmetrize(W)


def _normalize(config: HDMConfig, W: sp.csr_matrix) -> sp.csr_matrix:
    d = np.ones(W.shape[0], dtype=W.dtype)
    for _ in range(config.sinkhorn_max_iter):
        d_new = np.sqrt(d / (W @ d))
        done = np.max(np.abs(d_new - d)) < config.sinkhorn_tol
        d = d_new
        if done:
            break
    else:
        if config.verbose:
            print(f"Sinkhorn iteration did not converge after {config.sinkhorn_max_iter} iterations")
    D = sp.diags(d, format="csr")
    return D @ W @ D


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
    import cupyx.scipy.sparse.linalg as cpx_linalg  # type: ignore[import-not-found]
    import cupyx.scipy.sparse as cpsp  # type: ignore[import-not-found]

    kernel = cpsp.csr_matrix(kernel)

    assert kernel.shape is not None
    n = kernel.shape[0]
    v0 = cp.array(np.random.default_rng(config.seed).random(n), dtype=kernel.dtype)

    eigvals_cp, eigvecs_cp = cpx_linalg.eigsh(kernel, k=k + 1, which="LA", tol=config.eig_tol, v0=v0)
    eigvals = torch.from_dlpack(eigvals_cp)
    eigvecs = torch.from_dlpack(eigvecs_cp)

    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx]


def compute_spectral_embedding(
    config: HDMConfig,
    joint_kernel: sp.csr_matrix,
    sizes: list[int],
    num_data_samples: int,
) -> HDMResult:
    offsets = np.cumsum([0] + list(sizes))
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
