import numpy as np
from pathlib import Path

from HDM import run_hdm, HDMConfig
from _fixture_common import WINGS_CONFIG, assert_matches

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_bundle():
    # allow_pickle: maps is an object array of sparse blocks; this is our own committed fixture.
    data = np.load(FIXTURES_DIR / "wings_bundle.npz", allow_pickle=True)
    return data["base_dist"], data["maps"]


def test_e2e_wings_matches_golden():
    base_dist, maps = load_bundle()
    result = run_hdm(base_dist, maps, HDMConfig(**WINGS_CONFIG))
    expected = np.load(FIXTURES_DIR / "wings_expected.npz")
    assert_matches(result, expected)


def test_e2e_wings_detects_regression():
    base_dist, maps = load_bundle()
    result = run_hdm(base_dist, maps, HDMConfig(**WINGS_CONFIG))
    perturbed = dict(np.load(FIXTURES_DIR / "wings_expected.npz"))
    perturbed["hbdd"] = perturbed["hbdd"] + 1.0
    try:
        assert_matches(result, perturbed)
    except AssertionError:
        return
    raise AssertionError("assert_matches accepted a perturbed golden output")
