import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import torch

from .utils import HDMConfig, HDMResult, _is_cuda, approx_base_eps, torch_dtype


def compute_gaussian_kernel(dist: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-(dist**2) / eps)


def build_base_kernel(config: HDMConfig, base_dist: np.ndarray) -> sp.csr_matrix:
    nn = NearestNeighbors(n_neighbors=config.base_knn+1, metric="precomputed").fit(base_dist)
    knn = nn.kneighbors_graph(base_dist, mode="distance")

    assert not knn.diagonal().any()
    # knn.setdiag(0)
    knn.eliminate_zeros()
    knn.data = knn.data.astype(config.dtype, copy=False)


    if config.base_epsilon is None:
        config = config._replace(base_epsilon=approx_base_eps(base_dist))

    knn.data = compute_gaussian_kernel(knn.data, config.base_epsilon)

    return (knn + knn.T) * 0.5



def build_horizontal_diffusion_matrix(
    config: HDMConfig,
    maps: np.ndarray,
    base_kernel: sp.csr_matrix,
    data_sample_distances: np.ndarray,
    num_data_samples: int
) -> sp.csr_matrix:
    blocks = np.full((num_data_samples, num_data_samples), None, dtype=object)
    base_coo = base_kernel.tocoo()
    for i, j, v in zip(base_coo.row, base_coo.col, base_coo.data):
        blocks[i, j] = maps[i, j] @ data_sample_distances[j] * v
    W = sp.bmat(blocks.tolist(), format='csr')


    W.data = np.exp(-(W.data ** 2) / config.fiber_epsilon)
    #C = sp.bmat(base_kernel * W, format="csr")
    return (W + W.T) * 0.5


def _normalize(config: HDMConfig, W: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    d = np.asarray(W.sum(axis=1)).ravel()

    if np.any(d <= 0):
        print("d has an entry that is less or equal to 0, this indicates a problem with the kernel")
    d_pow_a = np.zeros_like(d)
    np.power(d, -config.alpha, out=d_pow_a, where=d>0)

    D_neg_pow_a = sp.diags(d_pow_a, format="csr")

    W_a = D_neg_pow_a @ W @ D_neg_pow_a
    D_a = np.asarray(W_a.sum(axis=1)).ravel()

    if np.any(D_a <= 0):
        print("D_a has an entry that is less or equal to 0, this indicates a problem with then")
    d_a_inv_sqrt = np.zeros_like(D_a)
    d_a_inv_sqrt = np.sqrt(D_a)
    np.reciprocal(d_a_inv_sqrt, out=d_a_inv_sqrt, where=d_a_inv_sqrt > 0)

    D_a_inv_sqrt = sp.diags(d_a_inv_sqrt, format="csr")
    A = D_a_inv_sqrt @ W_a @ D_a_inv_sqrt
    return A, d_a_inv_sqrt


def _eigsh_scipy(
    config: HDMConfig,
    kernel: sp.csr_matrix,
    k: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    n = kernel.shape[0]
    rng = np.random.default_rng(config.seed)
    v0 = rng.random(n, dtype=config.dtype)

    eigvals, eigvecs = sp.linalg.eigsh(kernel, k=k + 1, which="LM", tol=config.eig_tol, v0=v0)


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
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cpx_linalg
    import cupyx.scipy.sparse as cpsp

    kernel = cpsp.csr_matrix(kernel)

    n = kernel.shape[0]
    v0 = cp.array(np.random.default_rng(config.seed).random(n), dtype=kernel.dtype)

    eigvals_cp, eigvecs_cp = cpx_linalg.eigsh(kernel, k=k + 1, which="LM", tol=config.eig_tol, v0=v0)
    eigvals = torch.from_dlpack(eigvals_cp)
    eigvecs = torch.from_dlpack(eigvecs_cp)

    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx]



def compute_spectral_embedding(
    config: HDMConfig,
    joint_kernel: torch.Tensor,
    sizes: list[int],
    num_data_samples: int,
) -> HDMResult:
    offsets = np.cumsum([0] + list(sizes))
    num_eig = config.num_eigenvectors

    normalized_kernel, d_a_inv_sqrt = _normalize(config, joint_kernel)

    if _is_cuda(config.device):
        vals, V = _eigsh_cupy(config, normalized_kernel, num_eig)
    else:
        vals, V = _eigsh_scipy(config, normalized_kernel, num_eig)

    d_a_inv_sqrt = torch.as_tensor(d_a_inv_sqrt, dtype=V.dtype, device=V.device)

    vals = vals[1 : num_eig + 1]

    V = V[:, 1 : num_eig + 1]

    V = d_a_inv_sqrt[:, None] * V

    HDM = V * (vals ** config.t)


    V_scaled = (vals ** (config.t/2)) * V

    HBDM = torch.zeros((num_data_samples, num_eig**2), dtype=V.dtype, device=V.device)

    for i in range(num_data_samples):
        HBDM[i] = (V_scaled[offsets[i]:offsets[i+1]].T @ V_scaled[offsets[i]:offsets[i+1]]).ravel()


    HBDD = torch.cdist(HBDM, HBDM)

    return HDMResult(V.cpu().numpy(), vals.cpu().numpy(), HDM.cpu().numpy(), HBDM.cpu().numpy(), HBDD.cpu().numpy())

def gram(V):
    return V.T @ V
