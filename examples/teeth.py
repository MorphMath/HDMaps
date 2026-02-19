import numpy as np

from scipy.sparse import load_npz
import pyvista as pv
from HDM import (
    hdm_embed,
    HDMConfig,
)

from pathlib import Path
from scipy.io import loadmat

def load_off_vertices(path):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    n_vertices, _, _ = map(int, lines[1].split())

    return np.array(
        [list(map(float, lines[i].split())) for i in range(2, 2 + n_vertices)],
        dtype=float,
    )


def load_all_off_vertices(folder):
    folder = Path(folder)
    arrays = [load_off_vertices(p) for p in sorted(folder.glob("*.off"))]
    return arrays


data_samples = load_all_off_vertices("platyrrhine/ReparametrizedOFF")
np.save("data_samples.npy", np.vstack(data_samples))
maps = loadmat("platyrrhine/softMapMatrix.mat")["softMapMatrix"]
base_distances = load_npz(
    "example-data/teeth/base_distances.npz"
)
# block_indices = np.load("example-data/teeth/block_indices.npy")


config = HDMConfig(
    base_epsilon = 0.04,
    fiber_epsilon = 0.08,
    base_knn=4,
    fiber_knn=4,
    device="cpu")

result = hdm_embed(
    config=config,
    maps=maps,
    base_distances = base_distances,
    data_samples = data_samples
)

points = result.hdm_coords

np.save("eigvals.npy", result.eigvals)
np.save("eigvecs.npy", result.eigvecs)
print(result.eigvecs.shape)

""" 
points = pv.PolyData(points)
plotter = pv.Plotter()
plotter.add_mesh(points, color="red", point_size=5, render_points_as_spheres=True)
plotter.show() """
