"""Synthetic test of the full pipeline on a Mobius band and a cylinder, where the answer is known.

Each object is a noisy point cloud sampled across the band at one position along the base circle,
with no point correspondences given. The maps between neighboring objects come from ``soft_map``.
Going once around the Mobius band reverses the fiber, so the computed map across the seam should
flip it, while on the cylinder it should not. HDM then embeds both bundles.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree

from hdmaps.hdm import HDMConfig, run_hdm
from hdmaps.mapping import MapBundle, soft_map
from hdmaps.visual import plot_hbdd_mds, plot_hdm

n_base, n_fiber = 40, 30
rng = np.random.default_rng(0)
u = 2 * np.pi * np.arange(n_base) / n_base


def mobius(u: float, v: np.ndarray) -> np.ndarray:
    r = 1 + v / 2 * np.cos(u / 2)
    return np.column_stack([r * np.cos(u), r * np.sin(u), v / 2 * np.sin(u / 2)])


def cylinder(u: float, v: np.ndarray) -> np.ndarray:
    return np.column_stack([np.full_like(v, np.cos(u)), np.full_like(v, np.sin(u)), v / 2])


def knn_distances(dist: np.ndarray, k: int) -> sp.csr_matrix:
    """Symmetric k-nearest-neighbor graph of a dense distance matrix, with the zero diagonal stored."""
    n = len(dist)
    nearest = np.argsort(dist, axis=1)[:, 1 : k + 1]
    keep = np.eye(n, dtype=bool)
    keep[np.repeat(np.arange(n), k), nearest.ravel()] = True
    keep |= keep.T
    rows, cols = np.nonzero(keep)
    return sp.csr_matrix((dist[rows, cols], (rows, cols)), shape=(n, n))


def run(surface):
    # the position v across the band is only used to check the maps, never given to the pipeline
    vs = [rng.uniform(-1, 1, n_fiber) for _ in u]
    objects = [surface(ui, v) + 0.01 * rng.normal(size=(n_fiber, 3)) for ui, v in zip(u, vs)]

    trees = [KDTree(o) for o in objects]
    base = np.zeros((n_base, n_base))
    for i in range(n_base):
        for j in range(i + 1, n_base):
            base[i, j] = base[j, i] = 0.5 * (trees[j].query(objects[i])[0].mean() + trees[i].query(objects[j])[0].mean())
    base_dist = knn_distances(base, k=4)
    fiber_dists = [knn_distances(np.linalg.norm(o[:, None] - o[None], axis=-1), k=n_fiber - 1) for o in objects]
    maps = MapBundle(soft_map, objects, mask=base_dist.toarray() > 0)

    # does the map across the seam preserve (+1) or reverse (-1) the position across the band?
    seam = np.corrcoef(vs[-1], maps[n_base - 1, 0] @ vs[0])[0, 1]

    local = [np.sort(f.toarray(), axis=1)[:, 1:5] for f in fiber_dists]
    config = HDMConfig(
        base_epsilon=float(np.median(sp.triu(base_dist, k=1).data)),
        fiber_epsilon=float(np.median(np.concatenate([d.ravel() for d in local]))),
        num_eigenvectors=10,
        verbose=False,
    )
    return run_hdm(config, base_dist, maps, fiber_dists), seam, np.concatenate(vs)


fig = plt.figure(figsize=(12, 10))
for row, (name, surface) in enumerate([("Mobius band", mobius), ("Cylinder", cylinder)]):
    result, seam, v = run(surface)
    print(f"{name}: correlation across the seam {seam:+.2f} (expected {'-1' if surface is mobius else '+1'})")

    ax = plot_hdm(result, color=v, ax=fig.add_subplot(2, 2, 2 * row + 1, projection="3d"))
    ax.set_title(f"{name}: HDM, colored by position across the band")
    ax = plot_hbdd_mds(result, color=u, ax=fig.add_subplot(2, 2, 2 * row + 2))
    ax.set_title(f"{name}: MDS on HBDD, colored by base angle")

plt.tight_layout()
plt.show()
