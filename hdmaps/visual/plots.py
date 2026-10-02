from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from hdmaps.hdm import HDMResult


def mds(distances: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical multidimensional scaling of a dense distance matrix."""
    n = len(distances)
    centering = np.eye(n) - 1 / n
    gram = -0.5 * centering @ (distances**2) @ centering
    vals, vecs = np.linalg.eigh(gram)
    top = np.argsort(vals)[::-1][:n_components]
    return vecs[:, top] * np.sqrt(np.maximum(vals[top], 0))


def plot_hdm(
    result: HDMResult,
    color: np.ndarray | None = None,
    ax: Axes3D | None = None,
    max_points: int | None = 20000,
    seed: int = 0,
) -> Axes3D:
    """3D scatter of the first three HDM coordinates, colored by sample unless ``color`` is given."""
    coords = result.HDM[:, :3]
    if color is None:
        color = np.repeat(np.arange(len(result.offsets) - 1), np.diff(result.offsets))
    if max_points is not None and len(coords) > max_points:
        idx = np.random.default_rng(seed).choice(len(coords), max_points, replace=False)
        coords, color = coords[idx], color[idx]
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    assert isinstance(ax, Axes3D)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, s=2)  # pyright: ignore[reportArgumentType]  (stubs type zs as a scalar)
    ax.set_xlabel("HDM 1")
    ax.set_ylabel("HDM 2")
    ax.set_zlabel("HDM 3")
    return ax


def plot_hbdd_mds(result: HDMResult, color: np.ndarray | None = None, ax: Axes | None = None) -> Axes:
    """2D scatter of the samples, placed by classical MDS on the horizontal base diffusion distance."""
    coords = mds(result.HBDD)
    if ax is None:
        ax = plt.figure().add_subplot()
    ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)) if color is None else color)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    return ax
