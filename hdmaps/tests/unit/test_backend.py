import numpy as np
import scipy.sparse as sp

from hdmaps.hdm import HDMConfig
from hdmaps.hdm.backend import _mapped_fiber_kernel, _normalize, apply_kernel


def test_apply_kernel():
    dist = np.array([0.0, 1.0, 2.0])
    eps = 2.0
    expected = np.exp(-(dist**2) / eps**2)
    result = apply_kernel(dist, eps)
    np.testing.assert_allclose(result, expected)


def test_apply_kernel_zero_distance():
    result = apply_kernel(np.array([0.0]), eps=1.0)
    np.testing.assert_allclose(result, [1.0])


def test_mapped_fiber_kernel_keeps_zero_distances():
    # row 0 maps hard onto point 0, whose only mapped distance is F's stored zero diagonal
    M = sp.csr_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]))
    F = sp.csr_matrix((np.array([0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0]),
                       ([0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 0, 1, 2, 0, 1, 2])), shape=(3, 3))
    eps, v = 1.5, 0.7

    F_pat = F.copy()
    F_pat.data[:] = 1
    stored = (M.toarray() > 0) @ F_pat.toarray() > 0
    expected = np.where(stored, v * apply_kernel(M.toarray() @ F.toarray(), eps), 0)

    result = _mapped_fiber_kernel(M, F, v, eps)
    np.testing.assert_allclose(result.toarray(), expected)
    assert result[0, 0] == v


def test_normalize_is_symmetric_and_doubly_stochastic():
    # like the joint kernel: symmetric, nonnegative, no diagonal
    W = sp.csr_matrix(np.array([[0, .5, .2, .3], [.5, 0, .4, .1], [.2, .4, 0, .6], [.3, .1, .6, 0]]))
    K = _normalize(HDMConfig(), W).toarray()
    np.testing.assert_allclose(K.sum(axis=1), 1)
    np.testing.assert_allclose(K, K.T)


def test_mapped_fiber_kernel_drops_pairs_missing_a_fiber_distance():
    # points at x = 0, 1, 4 storing only the nearest neighbour: d(p0, p2) and d(p2, p0) are missing
    F = sp.csr_matrix((np.array([0.0, 1.0, 1.0, 0.0, 3.0, 0.0]), ([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 1, 2])), shape=(3, 3))
    M = sp.csr_matrix(np.array([[0.5, 0.0, 0.5]]))  # maps half onto p0, half onto p2
    result = _mapped_fiber_kernel(M, F, 1.0, 1.0)
    # only p1 has a stored distance from both p0 and p2: mapped distance 0.5 * 1 + 0.5 * 3 = 2
    np.testing.assert_allclose(result.toarray(), [[0, np.exp(-4), 0]])
    assert result.nnz == 1
