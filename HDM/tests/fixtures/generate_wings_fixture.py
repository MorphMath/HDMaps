"""Regenerate the wings end-to-end fixture and its golden outputs.

Unlike the shipped 1862-wing bundles (which store only each wing's 4 nearest
neighbours, so any small subset disconnects), this rebuilds a *small,
fully-connected* wing dataset from the raw skeleton images: N_WINGS wings, each
farthest-point-sampled to ~TARGET_N points, with base_knn = N_WINGS - 1 so every
pair of wings is coupled. That keeps the joint kernel connected and the
embedding non-degenerate at fixture scale.

The pipeline (skeletonize -> FPS -> sliced-Wasserstein base distances ->
entropic-OT fibre maps -> Gaussian kernel) mirrors
workspace/wings/process-raw-wings/trimmed-to-one-to-one/main.py, and is fully
deterministic. Requires the raw PNGs plus POT/scikit-image; only the author runs
it. The committed wings_bundle.npz / wings_expected.npz are what the test uses.

Usage: python generate_wings_fixture.py [path/to/trimmed_png_dir]
"""
import sys
import glob
import os
from pathlib import Path
from itertools import product

import numpy as np
import ot
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent.parent))

from HDM import run_hdm, HDMConfig
from _fixture_common import WINGS_CONFIG, golden_outputs

DEFAULT_PNG_DIR = "/home/sofus/hdm/workspace/wings/data/trimmed"
N_WINGS = 8
TARGET_N = 60
N_PROJ = 50
SEED = 42
REG = 30.0
TOPK = 3
HERE = Path(__file__).parent


def skeleton_pixels(path):
    binary = np.array(Image.open(path).convert("L")) > 128
    return np.argwhere(skeletonize(binary)).astype(np.float64)


def fps_radius(points, radius, seed_idx=0):
    chosen = [seed_idx]
    d = np.linalg.norm(points - points[seed_idx], axis=1)
    while d.max() >= radius:
        i = int(d.argmax())
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return points[chosen]


def sample_wings(files, target_n, r0=19.0):
    skeletons = [skeleton_pixels(f) for f in files]
    radius = r0 * float(np.mean([len(fps_radius(s, r0)) for s in skeletons])) / target_n
    wings = [fps_radius(s, radius) for s in skeletons]
    arr = np.empty(len(wings), dtype=object)
    arr[:] = wings
    return arr


def pairwise_swd(wings, n_proj, seed):
    n = len(wings)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = ot.sliced_wasserstein_distance(wings[i], wings[j], n_projections=n_proj, seed=seed, p=1)
            D[i, j] = D[j, i] = d
    return D


def pairwise_ot(a, b, reg, topk):
    wa, wb = np.ones(len(a)) / len(a), np.ones(len(b)) / len(b)
    M = ot.dist(a, b, metric="euclidean")
    G = ot.sinkhorn(wa, wb, M, reg, method="sinkhorn_log", numItermax=2000)
    cols = np.argsort(G, axis=1)[:, -topk:]
    d = np.maximum(np.take_along_axis(M, cols, axis=1), 1e-6)
    rows = np.repeat(np.arange(len(a)), topk)
    return csr_matrix((d.ravel(), (rows, cols.ravel())), shape=(len(a), len(b)))


def ot_dist_maps(points, D, base_knn, reg, topk):
    n = len(points)
    sizes = [len(w) for w in points]
    order = np.argsort(D, axis=1)
    idx = np.array([[j for j in order[i] if j != i][:base_knn] for i in range(n)])
    dmaps = np.empty((n, n), dtype=object)
    for i, j in product(range(n), range(n)):
        dmaps[i, j] = csr_matrix((sizes[i], sizes[j]))
    for i in range(n):
        for j in idx[i]:
            dmaps[i, j] = pairwise_ot(points[i], points[j], reg, topk)
    return dmaps


def approx_fiber_eps(samples, fiber_knn):
    per_mesh = []
    for P in samples:
        if len(P) < 2:
            continue
        d, _ = cKDTree(P).query(P, k=min(fiber_knn, len(P)))
        per_mesh.append(d.mean())
    return np.median(per_mesh) ** 2


def apply_kernel(dmaps, fiber_eps):
    probs = np.empty_like(dmaps)
    for i in range(dmaps.shape[0]):
        for j in range(dmaps.shape[1]):
            M = dmaps[i, j]
            if M.nnz == 0:
                probs[i, j] = M.astype(np.float64)
                continue
            w = np.exp(-(M.data ** 2) / fiber_eps)
            rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
            rowsum = np.bincount(rows, weights=w, minlength=M.shape[0])
            rowsum[rowsum == 0] = 1
            w = w / rowsum[rows]
            probs[i, j] = csr_matrix((w, M.indices, M.indptr), shape=M.shape).astype(np.float64)
    return probs


def build_bundle(png_dir):
    files = sorted(glob.glob(os.path.join(png_dir, "*.png")))[:N_WINGS]
    if len(files) < N_WINGS:
        raise SystemExit(f"need {N_WINGS} wing PNGs in {png_dir}, found {len(files)}")
    names = [os.path.basename(f) for f in files]

    points = sample_wings(files, TARGET_N)
    base_dist = np.ascontiguousarray(pairwise_swd(points, N_PROJ, SEED), dtype=np.float64)
    dmaps = ot_dist_maps(points, base_dist, N_WINGS - 1, REG, TOPK)
    maps = apply_kernel(dmaps, approx_fiber_eps(points, TOPK))
    return base_dist, maps, names


def main():
    png_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PNG_DIR
    base_dist, maps, names = build_bundle(png_dir)

    np.savez(HERE / "wings_bundle.npz", base_dist=base_dist, maps=maps, names=np.array(names))

    result = run_hdm(base_dist, maps, HDMConfig(**WINGS_CONFIG))
    np.savez(HERE / "wings_expected.npz", **golden_outputs(result))

    print(f"wings fixture: {N_WINGS} wings, fibre sizes {[maps[i, i].shape[0] for i in range(N_WINGS)]}, "
          f"{sum(maps[i, j].nnz for i in range(N_WINGS) for j in range(N_WINGS))} total nnz")
    print("eigvals:", np.round(result.eigvals, 6))


if __name__ == "__main__":
    main()
