"""Tests for the assumptions in hdmaps/hdm/validation.py, one per id."""

import numpy as np
import pytest
import scipy.sparse as sp

from hdmaps.hdm import HDMConfig
from hdmaps.hdm.validation import check_config, check_distances, check_map, check_structure, validate_inputs
from hdmaps.mapping import MapBundle
from helpers import csr


def dists(points):
    return csr(np.abs(np.subtract.outer(points, points)))


def bundle(sizes):
    return MapBundle(lambda i, j: sp.csr_matrix((sizes[i], sizes[j])), list(range(len(sizes))))


def raises(id_, name):
    return pytest.raises(ValueError, match=rf"^{id_}: {name}:")


D = [[0, 1, 3],
     [1, 0, 2],
     [3, 2, 0]]


def test_valid_distances_pass():
    check_distances(csr(D), "base_dist")
    check_distances(sp.csr_array(csr(D)), "base_dist")


@pytest.mark.parametrize("bad", [np.array(D, dtype=float), csr(D).tocoo()], ids=["dense", "coo"])
def test_D1_not_csr(bad):
    with raises("D1", "base_dist"):
        check_distances(bad, "base_dist")


def test_D1_not_square():
    with raises("D1", "base_dist"):
        check_distances(csr([[0, 1, 2], [1, 0, 1]]), "base_dist")


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_D2_entry_not_finite_and_nonnegative(value):
    A = csr(D)
    A[0, 2] = value
    with raises("D2", r"fiber_dists\[2\]"):
        check_distances(A, "fiber_dists[2]")


def test_D3_diagonal_not_stored():
    stored = np.ones((3, 3), bool)
    stored[1, 1] = False
    with raises("D3", "base_dist"):
        check_distances(csr(D, stored), "base_dist")


def test_D3_diagonal_not_zero():
    A = csr(D)
    A[1, 1] = 0.5
    with raises("D3", "base_dist"):
        check_distances(A, "base_dist")


def test_D4_duplicate_entry():
    A = csr(D)
    # store d(0, 1) twice
    A = sp.csr_matrix((np.r_[A.data, 1.0], np.r_[A.indices, 1], np.r_[A.indptr[:1], A.indptr[1:] + 1]), shape=A.shape)
    with raises("D4", "base_dist"):
        check_distances(A, "base_dist")


def test_S1_valid_structure_passes():
    check_structure(csr(D), bundle([2, 1, 3]), [dists([0, 1]), dists([0]), dists([0, 1, 2])])


def test_S1_wrong_number_of_fibers():
    with raises("S1", "fiber_dists"):
        check_structure(csr(D), bundle([2, 1, 3]), [dists([0, 1]), dists([0])])


def test_S1_wrong_number_of_map_objects():
    with raises("S1", "maps"):
        check_structure(csr(D), bundle([2, 1]), [dists([0, 1]), dists([0]), dists([0, 1, 2])])


def test_S1_empty_fiber():
    with raises("S1", r"fiber_dists\[1\]"):
        check_structure(csr(D), bundle([2, 0, 3]), [dists([0, 1]), csr(np.zeros((0, 0))), dists([0, 1, 2])])


def test_validate_inputs_names_the_bad_fiber():
    fibers = [dists([0, 1]), csr([[0, 1], [1, 0]], [[True, True], [True, False]]), dists([0, 1, 2])]
    with raises("D3", r"fiber_dists\[1\]"):
        validate_inputs(HDMConfig(fiber_epsilon=1), csr(D), bundle([2, 2, 3]), fibers)


MAP = [[1, 0, 0],
       [0, .5, .5]]


def test_valid_map_passes():
    check_map(sp.csr_matrix(MAP), "maps[3, 7]", (2, 3))


@pytest.mark.parametrize("bad", [sp.csc_matrix(MAP), np.array(MAP)], ids=["csc", "dense"])
def test_M2_not_csr(bad):
    with raises("M2", r"maps\[3, 7\]"):
        check_map(bad, "maps[3, 7]", (2, 3))


def test_M2_wrong_shape():
    with raises("M2", r"maps\[3, 7\]"):
        check_map(sp.csr_matrix(MAP), "maps[3, 7]", (2, 4))


def test_M3_negative_entry():
    with raises("M3", r"maps\[3, 7\]"):
        check_map(sp.csr_matrix([[1.5, -.5, 0], [0, .5, .5]]), "maps[3, 7]", (2, 3))


@pytest.mark.parametrize("row", [[.5, .4, 0], [0, 0, 0]], ids=["sums to 0.9", "empty"])
def test_M3_row_does_not_sum_to_1(row):
    with raises("M3", r"maps\[3, 7\]"):
        check_map(sp.csr_matrix([[1, 0, 0], row]), "maps[3, 7]", (2, 3))


def test_M3_row_sum_within_tolerance_passes():
    check_map(sp.csr_matrix([[1, 0, 0], [0, .5, .495]]), "maps[3, 7]", (2, 3))


def test_M4_stored_zero():
    M = sp.csr_matrix((np.array([1.0, 0.0, .5, .5]), np.array([0, 1, 1, 2]), np.array([0, 2, 4])), shape=(2, 3))
    with raises("M4", r"maps\[3, 7\]"):
        check_map(M, "maps[3, 7]", (2, 3))


def test_M5_duplicate_entry():
    # row 1 stores (1, 1) twice: .25 + .25 + .5 still sums to 1
    M = sp.csr_matrix((np.array([1.0, .25, .25, .5]), np.array([0, 1, 1, 2]), np.array([0, 1, 4])), shape=(2, 3))
    with raises("M5", r"maps\[3, 7\]"):
        check_map(M, "maps[3, 7]", (2, 3))


def test_valid_config_passes():
    check_config(HDMConfig(base_epsilon=1, fiber_epsilon=1, num_eigenvectors=3), num_points=5)
    check_config(HDMConfig(base_epsilon=None, fiber_epsilon=None, num_eigenvectors=3), num_points=5)


@pytest.mark.parametrize("field", ["base_epsilon", "fiber_epsilon"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan])
def test_C1_epsilon_not_positive(field, value):
    with raises("C1", field):
        check_config(HDMConfig(**{field: value}), num_points=10)


@pytest.mark.parametrize("t", [0.0, -1.0])
def test_C2_t_not_positive(t):
    with raises("C2", "t"):
        check_config(HDMConfig(t=t), num_points=10)


@pytest.mark.parametrize("num_eigenvectors", [0, 4])
def test_C3_num_eigenvectors_out_of_range(num_eigenvectors):
    # 5 points: at most 3 eigenvectors, since the solver computes num_eigenvectors + 1 < 5
    with raises("C3", "num_eigenvectors"):
        check_config(HDMConfig(num_eigenvectors=num_eigenvectors), num_points=5)
