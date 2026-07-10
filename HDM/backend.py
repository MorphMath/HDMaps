import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import torch

from .utils import HDMConfig, HDMResult, _is_cuda, approx_base_eps, torch_dtype





def gaussian_kernel(dist: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-(dist**2) / eps)


def base_kernel(base_dist: np.ndarray, config: HDMConfig) -> sp.csr_matrix:
    nn = NearestNeighbors(n_neighbors=config.base_knn+1, metric="precomputed").fit(base_dist)
    knn = nn.kneighbors_graph(base_dist, mode="distance")
    
    knn.setdiag(0)
    knn.eliminate_zeros()
    knn.data = knn.data.astype(config.dtype, copy=False)


    if config.base_epsilon is None:
        config = config._replace(base_epsilon=approx_base_eps(base_dist))
        
    knn.data = gaussian_kernel(knn.data, config.base_epsilon)

    return (knn + knn.T) * 0.5



def full_kernel(maps: np.ndarray, base_kernel: np.ndarray, num_data_samples: int, config: HDMConfig) -> sp.csr_matrix:
    blocks = np.empty_like(maps)
    for i in range(num_data_samples):
        for j in range(num_data_samples):
            if base_kernel[i, j] == 0:
                blocks[i, j] = None
            else:
                blocks[i, j] = maps[i, j] * base_kernel[i, j]

    W = sp.bmat(blocks.tolist(), format='csr')
    return (W + W.T) * 0.5


def _normalize(W: sp.csr_matrix, config: HDMConfig) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    d = np.asarray(W.sum(axis=1)).ravel()
    d_pow_a = np.where(d > 0, d ** (-config.alpha), 0.0)
    D_neg_pow_a = sp.diags(d_pow_a, format="csr")

    W_a = D_neg_pow_a @ W @ D_neg_pow_a
    d_a = np.asarray(W_a.sum(axis=1)).ravel()
    d_a_inv_sqrt = np.where(d_a > 0, 1.0 / np.sqrt(d_a), 0.0)
    D_a_inv_sqrt = sp.diags(d_a_inv_sqrt, format="csr")
    A = D_a_inv_sqrt @ W_a @ D_a_inv_sqrt
    return A, d_a_inv_sqrt



def _eigsh_scipy(kernel: sp.csr_matrix, k: int, config: HDMConfig) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    n = kernel.shape[0]
    rng = np.random.default_rng(config.seed)
    v0 = rng.random(n, dtype=config.dtype)

    eigvals, eigvecs = sp.linalg.eigsh(kernel, k=k + 1, which="LM", tol=1e-6, v0=v0)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return (
        torch.as_tensor(eigvals, dtype=torch_dtype(config.dtype), device=config.device),
        torch.as_tensor(eigvecs, dtype=torch_dtype(config.dtype), device=config.device),
    )


def _eigsh_cupy(
    kernel: sp.csr_matrix, k: int, config: HDMConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import cupyx.scipy.sparse.linalg as cpx_linalg
    import cupyx.scipy.sparse as cpsp

    kernel = cpsp.csr_matrix(kernel)
    
    n = kernel.shape[0]
    v0 = cp.array(np.random.default_rng(config.seed).random(n), dtype=kernel.dtype) 
    
    eigvals_cp, eigvecs_cp = cpx_linalg.eigsh(kernel, k=k + 1, which="LM", tol=1e-6, v0=v0)

    eigvals = torch.from_dlpack(eigvals_cp)
    eigvecs = torch.from_dlpack(eigvecs_cp)

    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx]


def spectral_embedding(
    config: HDMConfig,
    joint_kernel: torch.Tensor,
    sizes: list[int],
    num_data_samples: int,
) -> HDMResult:
    offsets = np.cumsum([0] + list(sizes))
    num_eig = config.num_eigenvectors

    normalized_kernel, sqrt_inv_D = _normalize(joint_kernel, config)

    if _is_cuda(config.device):
        vals, V = _eigsh_cupy(normalized_kernel, num_eig, config)
    else:
        vals, V = _eigsh_scipy(normalized_kernel, num_eig, config)

    sqrt_inv_D = torch.as_tensor(sqrt_inv_D, dtype=V.dtype, device=V.device)

    vals = vals[1 : num_eig + 1]
    vals = 1.0 - vals
    V = V[:, 1 : num_eig + 1]

    # This is not included in the paper, but tingran does it??
    V = sqrt_inv_D[:, None] * V

    HDM = V * (vals ** config.t)

    HBDD = torch.zeros((num_data_samples, num_data_samples), dtype=V.dtype, device=V.device)

    grams = [gram(V[offsets[i]:offsets[i+1]]) for i in range(num_data_samples)]

    W = torch.outer(vals ** config.t, vals ** config.t)

    def inner(a, b):
        return torch.sum(W * grams[a] * grams[b])

    self_inner = [inner(i, i) for i in range(num_data_samples)]

    for i in range(num_data_samples):
        for j in range(i+1, num_data_samples):
            HBDD[i, j] = HBDD[j, i] = torch.sqrt(self_inner[i] + self_inner[j] - 2 * inner(i, j))

    return HDMResult(V.cpu().numpy(), vals.cpu().numpy(), HDM.cpu().numpy(), HBDD.cpu().numpy())

def gram(V):
    return V.T @ V

