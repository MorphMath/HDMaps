import numpy as np
import torch
from pathlib import Path
from scipy.sparse import eye as speye
from HDM import run_hdm, HDMConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_wings_backend_consistency():
    files = sorted(FIXTURES_DIR.glob("*.txt"))
    samples = [np.loadtxt(f, delimiter=",") for f in files]

    n = len(samples)
    size = samples[0].shape[0]

    maps = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            maps[i, j] = speye(size, format="csr")

    config = HDMConfig(
        base_knn=4,
        fiber_knn=3,
        num_eigenvectors=6,
        device=torch.device("cpu"),
        verbose=False,
        seed=42,
    )

    result = run_hdm(config=config, maps=maps, data_samples=samples)
    G = result.hdm_coords @ result.hdm_coords.T

    assert np.allclose(result.eigvals, np.load(FIXTURES_DIR / "expected_wings_eigvals.npy"), atol=1e-5)
    assert np.allclose(G, np.load(FIXTURES_DIR / "expected_wings_gram.npy"), atol=1e-5)
