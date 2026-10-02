import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree


def _sinkhorn(kernel: sp.csr_matrix, a: np.ndarray, b: np.ndarray, max_iter: int, tol: float) -> sp.csr_matrix:
    assert kernel.shape is not None
    u, v = np.ones(kernel.shape[0]), np.ones(kernel.shape[1])
    kernel_t = kernel.T.tocsr()
    for _ in range(max_iter):
        u = a / (kernel @ v)
        v = b / (kernel_t @ u)
        if np.abs(u * (kernel @ v) - a).sum() < tol:
            break
    return sp.csr_matrix(sp.diags(u) @ kernel @ sp.diags(v))


def soft_map(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 6,
    epsilon: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> sp.csr_matrix:
    """Sparse soft map from ``source`` to ``target``, two aligned point sets.

    Each point is linked to its ``k`` nearest neighbors in the other set, the links are weighted by
    a Gaussian kernel, balanced to uniform marginals with Sinkhorn, and every row is normalized to
    sum to one. ``epsilon`` defaults to the median squared link length.
    """
    n_a, n_b = len(source), len(target)
    a, b = np.full(n_a, 1 / n_a), np.full(n_b, 1 / n_b)
    k_ab, k_ba = min(k, n_b), min(k, n_a)

    d_ab, c_ab = KDTree(target).query(source, k=k_ab)
    d_ba, c_ba = KDTree(source).query(target, k=k_ba)
    rows = np.concatenate([np.repeat(np.arange(n_a), k_ab), np.ravel(c_ba)])
    cols = np.concatenate([np.ravel(c_ab), np.repeat(np.arange(n_b), k_ba)])
    d2 = np.concatenate([np.ravel(d_ab), np.ravel(d_ba)]) ** 2
    _, idx = np.unique(np.stack([rows, cols], axis=1), axis=0, return_index=True)
    rows, cols, d2 = rows[idx], cols[idx], d2[idx]

    epsilon = float(np.median(d2)) if epsilon is None else epsilon
    row_min = np.full(n_a, np.inf)
    np.minimum.at(row_min, rows, d2)  # subtracting each row's minimum keeps the kernel from underflowing
    kernel = sp.csr_matrix((np.exp(-(d2 - row_min[rows]) / epsilon), (rows, cols)), shape=(n_a, n_b))

    plan = _sinkhorn(kernel, a, b, max_iter, tol)
    return sp.csr_matrix(sp.diags(1 / np.asarray(plan.sum(axis=1)).ravel()) @ plan)
