import numpy as np
from pathlib import Path
from scipy.sparse import eye as speye
from HDM import run_hdm, HDMConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_e2e_wings():
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
        device="cpu",
        verbose=False,
        seed=42,
    )

    result = run_hdm(config=config, maps=maps, data_samples=samples)

    assert np.allclose(result.hdm_coords, np.load(FIXTURES_DIR / "expected_hdm_coords.npy"))
    assert np.allclose(result.hbdm_coords, np.load(FIXTURES_DIR / "expected_hbdm_coords.npy"))
    assert np.allclose(result.eigvals, np.load(FIXTURES_DIR / "expected_eigvals.npy"))


def test_e2e_wings_detects_wrong_result():
    expected = np.load(FIXTURES_DIR / "expected_hdm_coords.npy")
    assert not np.allclose(expected + 1.0, expected)
    assert not np.allclose(np.zeros_like(expected), expected)
