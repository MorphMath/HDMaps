from typing import NamedTuple

import numpy as np
from scipy.sparse import block_array, coo_matrix, issparse
from sklearn.cluster import KMeans


class HDMConfig(NamedTuple):
    base_epsilon: float | None = None
    fiber_epsilon: float | None = None
    num_eigenvectors: int = 5
    device: str | None = "cpu"  # 'cpu', 'cpu_ragged', 'jax', or 'gpu'
    base_metric: str = "frobenius"
    fiber_metric: str = "euclidean"
    base_knn: int = 8
    fiber_knn: int = 8
    verbose: bool = True


class HDMResult(NamedTuple):
    eigvecs: np.ndarray
    eigvals: np.ndarray
    hdm_coords: np.ndarray
    hbdm_coords: np.ndarray


def compute_block_indices(data_samples: list[np.ndarray]) -> np.ndarray:
    """Compute cumulative start indices for a list of data samples."""
    lengths = np.array([len(s) for s in data_samples], dtype=np.int32)
    return np.concatenate([np.array([0], dtype=np.int32), np.cumsum(lengths)])


def compute_clusters(
    hdm_coords: np.ndarray, num_clusters: int, seed=None
) -> np.ndarray:
    scaled_coords = StandardScaler().fit_transform(hdm_coords)

    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    labels = kmeans.fit_predict(scaled_coords)

    return labels


def visualize_by_eigenvector(
    point_cloud,
    hdm_coords,
    eigenvector_idx,
    start_idx=0,
    title="Eigenvector Visualization",
):
    import pyvista as pv

    plotter = pv.Plotter()

    pv_mesh = pv.PolyData(point_cloud)

    num_vertices = point_cloud.shape[0]

    eigenvector = hdm_coords[start_idx : start_idx + num_vertices, eigenvector_idx]

    pv_mesh.point_data["eigenvector"] = eigenvector

    plotter.add_mesh(pv_mesh, scalars="eigenvector", cmap="jet", show_edges=False)
    plotter.add_title(title)
    plotter.show()


def get_backend(config: HDMConfig):
    """Return the appropriate backend based on the configuration."""
    if config.device == "cpu":
        from . import backend

        return backend

    elif config.device == "gpu":
        print(
            "Warning: gpu backend is currently under development and is currently not supported"
        )
        from . import cupy

        return cupy
    elif config.device == "jax":
        from . import jax as jax_backend

        return jax_backend
    else:
        raise ValueError(f"Unsupported device: {config.device}")


def compute_fiber_kernel_from_maps(maps):
    # maps: 2D array-like of sparse matrices (or arrays)
    blocks = [
        [(m if issparse(m) else coo_matrix(m)).tocoo() for m in row] for row in maps
    ]
    return block_array(blocks, format="coo")
