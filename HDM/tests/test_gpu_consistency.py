import numpy as np
import pytest
from pathlib import Path

from HDM import run_hdm, HDMConfig
from _fixture_common import TEETH_CONFIG, WINGS_CONFIG, assert_matches

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# cupy's GPU eigensolver differs from scipy's ARPACK, so cross-device agreement is
# looser than the bit-identical CPU-to-CPU reproduction (near-zero HBDD entries inflate
# relative error); still far tighter than any real regression.
GPU_RTOL, GPU_ATOL = 1e-3, 1e-6


def gpu_unavailable():
    try:
        import cupy  # noqa: F401
        import torch
        return not torch.cuda.is_available()
    except Exception:
        return True


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(gpu_unavailable(), reason="requires a CUDA GPU with cupy installed"),
]


@pytest.mark.parametrize("tag, config", [("teeth", TEETH_CONFIG), ("wings", WINGS_CONFIG)])
def test_gpu_matches_cpu_golden(tag, config):
    # allow_pickle: maps is an object array of sparse blocks; this is our own committed fixture.
    bundle = np.load(FIXTURES_DIR / f"{tag}_bundle.npz", allow_pickle=True)
    expected = np.load(FIXTURES_DIR / f"{tag}_expected.npz")
    gpu_config = HDMConfig(**{**config, "device": "cuda"})
    result = run_hdm(bundle["base_dist"], bundle["maps"], gpu_config)
    assert_matches(result, expected, rtol=GPU_RTOL, atol=GPU_ATOL)
