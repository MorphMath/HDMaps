"""Assumptions on the inputs are listed in hdmaps/hdm/validation.py."""

import numpy as np
import pytest
import scipy.sparse as sp

from hdmaps.hdm import HDMConfig
from hdmaps.hdm.backend import (
    _mapped_fiber_kernel,
    _normalize,
    _apply_kernel,
    _symmetrize,
    _mapped_fiber_kernel,
    build_base_kernel,
    compute_spectral_embedding,
)


def sparse_allclose(A, B, rtol=1e-5, atol=1e-8):
    if A.shape != B.shape:
        return False
    D = abs(A - B) - rtol * abs(B)
    return D.nnz == 0 or D.max() <= atol


def test_mapped_fiber_kernel():
    map = sp.csr_matrix(np.array([
        [1/3, 1/3, 1/3],
        [0, 1/2, 1/2],
        [0, 0, 1]
    ]))
    dists = sp.csr_matrix(np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ]))

    result = _mapped_fiber_kernel(map, dists, 1, 1)









def test_build_horizontal_diffusion_matrix():
    pass


def test_compute_spectral_embedding_warns_when_disconnected():
    # two objects of 3 points each, with no edges between them
    block = np.full((3, 3), 0.2) + 0.8 * np.eye(3)
    W = sp.csr_matrix(sp.block_diag([block, block]))
    config = HDMConfig(num_eigenvectors=2, verbose=False)
    with pytest.warns(UserWarning, match="disconnected"):
        compute_spectral_embedding(config, W, np.array([0, 3, 6]), 2)


def test_compute_spectral_embedding_rejects_non_positive_eigenvalues():
    # a ring of 6 points without self-loops has eigenvalues 1, .5, .5, -.5, -.5, -1
    W = sp.csr_matrix(np.roll(np.eye(6), 1, axis=1) + np.roll(np.eye(6), -1, axis=1))
    config = HDMConfig(num_eigenvectors=4, sinkhorn_jitter=0, verbose=False)
    with pytest.raises(ValueError, match="only 2 of 4 eigenvalues are positive"):
        compute_spectral_embedding(config, W, np.array([0, 3, 6]), 2)


def test_build_base_kernel_rejects_an_estimated_epsilon_of_0():
    # identical objects: every distance is 0, so the median of the row maxima is too
    base_dist = sp.csr_matrix((np.zeros(4), ([0, 0, 1, 1], [0, 1, 0, 1])), shape=(2, 2))
    with pytest.raises(ValueError, match="^C1: base_epsilon:"):
        build_base_kernel(HDMConfig(), base_dist)
