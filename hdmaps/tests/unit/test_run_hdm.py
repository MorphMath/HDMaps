import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from hdmaps.hdm import HDMConfig, run_hdm
from hdmaps.mapping import MapBundle

CONFIG = HDMConfig(base_epsilon=2, fiber_epsilon=1, num_eigenvectors=4, verbose=False)


def cylinder(n, m, k):
    # base: a ring of n objects linked to their k nearest neighbours, fibers: m points on a line
    ring = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    D = np.minimum(ring, n - ring).astype(float)
    cols = np.argsort(D, axis=1, kind="stable")[:, : k + 1].ravel()
    rows = np.repeat(np.arange(n), k + 1)
    base_dist = sp.csr_matrix((D[rows, cols], (rows, cols)), shape=(n, n))
    t = np.linspace(-1, 1, m)
    r, c = np.indices((m, m)).reshape(2, -1)
    fiber = sp.csr_matrix((np.abs(t[r] - t[c]), (r, c)), shape=(m, m))
    eye = sp.csr_matrix(sp.identity(m, format="csr"))
    return base_dist, MapBundle(lambda i, j: eye, list(range(n))), [fiber] * n


def test_run_hdm_on_a_cylinder():
    n, m = 12, 5
    result = run_hdm(CONFIG, *cylinder(n, m, k=2))

    assert result.HDM.shape == (n * m, 4)
    assert result.HBDD.shape == (n, n)
    assert ((0 < result.eigvals) & (result.eigvals < 1)).all()


def test_disconnected_warning_points_at_the_callers_line():
    # k = 0: every object is linked only to itself
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_hdm(CONFIG, *cylinder(12, 5, k=0))

    [warning] = [w for w in caught if "disconnected" in str(w.message)]
    assert warning.filename == __file__


def test_run_hdm_checks_the_maps_it_fetches():
    base_dist, _, fibers = cylinder(12, 5, k=2)
    half = sp.csr_matrix(0.5 * sp.identity(5, format="csr"))  # rows sum to 0.5
    with pytest.raises(ValueError, match=r"^M3: maps\[0, 1\]:"):
        run_hdm(CONFIG, base_dist, MapBundle(lambda i, j: half, list(range(12))), fibers)


def test_run_hdm_casts_inputs_to_config_dtype_with_one_warning_each():
    config = CONFIG._replace(dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_hdm(config, *cylinder(12, 5, k=2))  # float64 inputs

    casts = [w for w in caught if "cast from float64 to float32" in str(w.message)]
    assert sorted(str(w.message).split(":")[0] for w in casts) == ["base_dist", "fiber_dists", "maps"]
    assert all(w.filename == __file__ for w in casts)
    assert result.HDM.dtype == np.float32


def test_run_hdm_does_not_warn_when_dtypes_match():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_hdm(CONFIG, *cylinder(12, 5, k=2))

    assert not [w for w in caught if "cast from" in str(w.message)]


def test_run_hdm_estimates_fiber_epsilon():
    result = run_hdm(CONFIG._replace(fiber_epsilon=None), *cylinder(12, 5, k=2))
    assert ((0 < result.eigvals) & (result.eigvals < 1)).all()


def test_run_hdm_rejects_an_estimated_fiber_epsilon_of_0():
    # one point per fiber: no non-zero fiber distances to estimate from
    with pytest.raises(ValueError, match="^C1: fiber_epsilon:"):
        run_hdm(CONFIG._replace(fiber_epsilon=None, num_eigenvectors=2), *cylinder(12, 1, k=2))
