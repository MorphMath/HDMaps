import numpy as np
from pathlib import Path

from HDM import run_hdm, HDMConfig
from _fixture_common import TEETH_CONFIG, assert_matches

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_bundle():
    # allow_pickle: maps is an object array of sparse blocks; this is our own committed fixture.
    data = np.load(FIXTURES_DIR / "teeth_bundle.npz", allow_pickle=True)
    return data["base_dist"], data["maps"]


def test_e2e_teeth_matches_golden():
    base_dist, maps = load_bundle()
    result = run_hdm(base_dist, maps, HDMConfig(**TEETH_CONFIG))
    expected = np.load(FIXTURES_DIR / "teeth_expected.npz")
    assert_matches(result, expected)


def test_e2e_teeth_detects_regression():
    base_dist, maps = load_bundle()
    result = run_hdm(base_dist, maps, HDMConfig(**TEETH_CONFIG))
    perturbed = dict(np.load(FIXTURES_DIR / "teeth_expected.npz"))
    perturbed["hbdd"] = perturbed["hbdd"] + 1.0
    try:
        assert_matches(result, perturbed)
    except AssertionError:
        return
    raise AssertionError("assert_matches accepted a perturbed golden output")
