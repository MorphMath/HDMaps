import numpy as np
from hdmaps.hdm.backend import apply_kernel


def test_apply_kernel():
    dist = np.array([0.0, 1.0, 2.0])
    eps = 2.0
    expected = np.exp(-(dist**2) / eps)
    result = apply_kernel(dist, eps)
    np.testing.assert_allclose(result, expected)


def test_apply_kernel_zero_distance():
    result = apply_kernel(np.array([0.0]), eps=1.0)
    np.testing.assert_allclose(result, [1.0])
