import numpy as np
import scipy.sparse as sp

from hdmaps.hdm.utils import approx_eps


def stored(*dists):
    return sp.csr_matrix((np.array(dists, dtype=float), (np.zeros(len(dists), int), np.arange(len(dists)))))


def test_approx_eps_is_the_median_of_the_medians():
    # medians 1, 2 and 10
    assert approx_eps([stored(1, 1, 5), stored(2), stored(10, 9, 11)]) == 2


def test_approx_eps_ignores_zeros():
    assert approx_eps([stored(0, 0, 0, 3, 5)]) == 4


def test_approx_eps_skips_matrices_without_non_zero_distances():
    assert approx_eps([stored(0), stored(3)]) == 3


def test_approx_eps_is_0_without_any_non_zero_distance():
    assert approx_eps([stored(0), stored(0, 0)]) == 0
