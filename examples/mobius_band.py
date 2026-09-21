import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from hdmaps.hdm import HDMConfig, run_hdm

n, m, k = 40, 9, 4

theta = 2 * np.pi * np.arange(n) / n
d = np.abs(theta[:, None] - theta[None, :])
D = np.minimum(d, 2 * np.pi - d)
np.fill_diagonal(D, np.inf)
cols = np.argsort(D, axis=1)[:, :k].ravel()
rows = np.repeat(np.arange(n), k)
base_dist = sp.csr_matrix((D[rows, cols], (rows, cols)), shape=(n, n))

t = np.linspace(-1, 1, m)
F = np.abs(t[:, None] - t[None, :])
r, c = np.indices((m, m)).reshape(2, -1)
fiber = sp.csr_matrix((F.ravel(), (r, c)), shape=(m, m))
fiber_dists = [fiber.copy() for _ in range(n)]

eye = sp.identity(m, format="csr")
flip = sp.csr_matrix(np.eye(m)[::-1])


class MobiusMaps:
    def __init__(self, n):
        self.n = n
        self.shape = (n, n)

    def __getitem__(self, key):
        i, j = key
        return flip if abs(i - j) > self.n // 2 else eye


class CylinderMaps(MobiusMaps):
    def __getitem__(self, key):
        return eye


config = HDMConfig(base_epsilon=4, fiber_epsilon=1, num_eigenvectors=40)

results = {
    "Möbius": run_hdm(config, base_dist, MobiusMaps(n), fiber_dists),
    "Cylinder": run_hdm(config, base_dist, CylinderMaps(n), fiber_dists),
}

U, V = np.meshgrid(theta, t, indexing="ij")
surfaces = {
    "Möbius": (
        (1 + V / 2 * np.cos(U / 2)) * np.cos(U),
        (1 + V / 2 * np.cos(U / 2)) * np.sin(U),
        V / 2 * np.sin(U / 2),
    ),
    "Cylinder": (np.cos(U), np.sin(U), V / 2),
}

point_base = np.repeat(np.arange(n), m)

fig = plt.figure(figsize=(12, 15))
for col, (name, res) in enumerate(results.items()):
    ax0 = fig.add_subplot(3, 2, col + 1, projection="3d")
    X, Y, Z = surfaces[name]
    ax0.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=point_base, cmap="hsv")
    ax0.set_box_aspect((1, 1, 0.5))
    ax0.set_title(f"{name}: true surface")

    ax1 = fig.add_subplot(3, 2, 2 + col + 1)
    ax1.scatter(*res.HDM[:, :2].T, c=point_base, cmap="hsv")
    ax1.set_title(f"{name}: HDM, top 2 columns")
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(3, 2, 4 + col + 1)
    ax2.scatter(*res.HBDM[:, :2].T, c=np.arange(n), cmap="hsv")
    for i, (x, y) in enumerate(res.HBDM[:, :2]):
        ax2.annotate(str(i), (x, y))
    ax2.set_title(f"{name}: HBDM, top 2 columns")

plt.tight_layout()
plt.show()
