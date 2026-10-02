"""Rigid alignment of point set collections, in broad strokes modelled after auto3dgm:

    D. M. Boyer, J. Puente, J. T. Gladman, C. Glynn, S. Mukherjee, G. S. Yapuncich and I. Daubechies,
    "A New Fully Automated Approach for Aligning and Comparing Shapes",
    The Anatomical Record 298(1):249-276, 2015. https://doi.org/10.1002/ar.23084
"""

import itertools

import fpsample
import numpy as np
import scipy.sparse as sp
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from scipy.spatial.distance import cdist


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Indices of ``n`` points chosen by farthest point sampling, starting from the point farthest from the centroid."""
    start = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    return fpsample.fps_sampling(points, min(n, len(points)), start_idx=start)


def normalize(points: np.ndarray) -> np.ndarray:
    """Center ``points`` at the origin and scale them to unit root mean square radius."""
    centered = points - points.mean(axis=0)
    return centered / np.sqrt((centered**2).sum(axis=1).mean())


def _kabsch(source: np.ndarray, target: np.ndarray, allow_reflection: bool) -> np.ndarray:
    r, _ = orthogonal_procrustes(source, target)
    if allow_reflection or np.linalg.det(r) > 0:
        return r
    u, _, vt = np.linalg.svd(source.T @ target)  # the best orthogonal map is a reflection: flip the weakest axis
    u[:, -1] *= -1
    return u @ vt


def _principal_axes_candidates(source: np.ndarray, target: np.ndarray, allow_reflection: bool) -> list[np.ndarray]:
    _, _, vs = np.linalg.svd(source, full_matrices=False)
    _, _, vt = np.linalg.svd(target, full_matrices=False)
    candidates = []
    for signs in itertools.product((1.0, -1.0), repeat=source.shape[1]):
        r = vs.T @ np.diag(signs) @ vt
        if allow_reflection or np.linalg.det(r) > 0:
            candidates.append(r)
    return candidates


def align_pair(
    source: np.ndarray, target: np.ndarray, allow_reflection: bool = False, max_iter: int = 100
) -> tuple[np.ndarray, float]:
    """Rotation ``r`` with ``source @ r`` matching ``target``, and the root mean square distance of the match.

    Both point sets must be normalized and have the same number of points. Each principal axes
    candidate is refined by alternating optimal matching and Kabsch, and the best one is kept.
    """
    best_r, best_cost = np.eye(source.shape[1]), np.inf
    for r in _principal_axes_candidates(source, target, allow_reflection):
        perm = None
        for _ in range(max_iter):
            _, new_perm = linear_sum_assignment(cdist(source @ r, target, "sqeuclidean"))
            if perm is not None and np.array_equal(perm, new_perm):
                break
            perm = new_perm
            r = _kabsch(source, target[perm], allow_reflection)
        cost = float(np.sqrt(((source @ r - target[perm]) ** 2).sum(axis=1).mean()))
        if cost < best_cost:
            best_r, best_cost = r, cost
    return best_r, best_cost


def align(
    shapes: list[np.ndarray], n_points: int = 128, allow_reflection: bool = False
) -> list[np.ndarray]:
    """Align a collection of point sets into one common frame.

    Each shape is normalized and subsampled, every pair is aligned with :func:`align_pair`, and
    the pairwise rotations are chained along a minimum spanning tree of the alignment costs.
    Returns the normalized shapes rotated into the frame of the most central shape.
    """
    shapes = [normalize(np.asarray(s, dtype=np.float64)) for s in shapes]
    subs = [normalize(s[farthest_point_sample(s, n_points)]) for s in shapes]
    n = len(shapes)

    rotations = np.empty((n, n), dtype=object)
    costs = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        r, cost = align_pair(subs[i], subs[j], allow_reflection)
        rotations[i, j], rotations[j, i] = r, r.T
        costs[i, j] = costs[j, i] = cost

    tree = minimum_spanning_tree(sp.csr_matrix(costs + np.finfo(float).tiny * (1 - np.eye(n))))
    root = int(np.argmin(costs.sum(axis=1)))
    order, parent = breadth_first_order(tree, root, directed=False)

    to_root = [np.eye(shapes[0].shape[1])] * n
    for node in order[1:]:
        to_root[node] = rotations[node, parent[node]] @ to_root[parent[node]]
    return [s @ r for s, r in zip(shapes, to_root)]
