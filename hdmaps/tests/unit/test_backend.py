import numpy as np
import scipy.sparse as sp

from hdmaps.hdm import HDMConfig
from hdmaps.hdm.backend import (
    _mapped_fiber_kernel,
    _normalize,
    _apply_kernel,
    _symmetrize,
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
    ])
    dists = sp.csr_matrix(np.array([
        [0, 1, 2],
        [0, 0, 1],
        [0, 0, 0]
    ])
    dists = dists + dists.T


    pass



def test_build_horizontal_diffusion_matrix():
    pass
