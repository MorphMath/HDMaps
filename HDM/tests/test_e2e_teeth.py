import numpy as np
from pathlib import Path
from HDM import run_hdm, HDMConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_e2e_teeth():
    meshes = np.load(FIXTURES_DIR / "teeth_meshes.npy", allow_pickle=True)
    maps = np.load(FIXTURES_DIR / "teeth_maps.npy", allow_pickle=True)

    config = HDMConfig(
        base_knn=4,
        fiber_knn=4,
        num_eigenvectors=4,
        device="cpu",
        verbose=False,
        seed=42,
    )

    result = run_hdm(config=config, maps=maps, data_samples=list(meshes))

    assert np.allclose(result.hdm_coords, np.load(FIXTURES_DIR / "expected_teeth_hdm_coords.npy"))
    assert np.allclose(result.hbdm_coords, np.load(FIXTURES_DIR / "expected_teeth_hbdm_coords.npy"))
    assert np.allclose(result.eigvals, np.load(FIXTURES_DIR / "expected_teeth_eigvals.npy"))
