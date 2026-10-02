import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1 / n)


def ot_plan(
    source: np.ndarray,
    target: np.ndarray,
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
    epsilon: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Entropic optimal transport plan between two weighted point sets, and its transport cost.

    The cost is the squared Euclidean distance, and ``epsilon`` is the entropic regularization
    relative to the mean cost. Weights default to uniform and are normalized to sum to one.
    """
    a = _uniform(len(source)) if source_weights is None else source_weights / source_weights.sum()
    b = _uniform(len(target)) if target_weights is None else target_weights / target_weights.sum()
    cost = cdist(source, target, "sqeuclidean")
    kernel = np.exp(-cost / (epsilon * cost.mean()))  # regularization relative to the mean cost keeps this from underflowing
    u, v = np.ones(len(a)), np.ones(len(b))
    for _ in range(max_iter):
        u = a / (kernel @ v)
        v = b / (kernel.T @ u)
        if np.abs(u * (kernel @ v) - a).sum() < tol:
            break
    plan = u[:, None] * kernel * v[None, :]
    return plan, float((plan * cost).sum())


def ot_map(
    source: np.ndarray,
    target: np.ndarray,
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
    epsilon: float = 0.05,
    threshold: float = 1e-3,
) -> sp.csr_matrix:
    """Sparse soft map from ``source`` to ``target`` given by an entropic optimal transport plan.

    Each row of the plan is normalized to sum to one, and entries below ``threshold`` times the
    row maximum are dropped to keep the map sparse.
    """
    plan, _ = ot_plan(source, target, source_weights, target_weights, epsilon)
    plan[plan < threshold * plan.max(axis=1, keepdims=True)] = 0
    plan /= plan.sum(axis=1, keepdims=True)
    return sp.csr_matrix(plan)


def ot_distance(
    source: np.ndarray,
    target: np.ndarray,
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
    epsilon: float = 0.05,
) -> float:
    """Entropic approximation of the 2-Wasserstein distance between two weighted point sets."""
    return float(np.sqrt(ot_plan(source, target, source_weights, target_weights, epsilon)[1]))
