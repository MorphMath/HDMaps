"""Assumptions on the inputs are listed in hdmaps/hdm/validation.py."""

import numpy as np
import scipy.sparse as sp

from hdmaps.hdm import HDMConfig
from hdmaps.mapping import MapBundle
from helpers import csr
from hdmaps.hdm.backend import (
    _mapped_fiber_kernel,
    build_base_kernel,
    build_horizontal_diffusion_matrix,
    compute_spectral_embedding,
)


def sparse_allclose(A, B, rtol=1e-5, atol=1e-8):
    if A.shape != B.shape:
        return False
    D = abs(A - B) - rtol * abs(B)
    return D.nnz == 0 or D.max() <= atol


def test_build_base_kernel():
    """build_base_kernel turns base_dist into the base kernel: every stored distance d becomes
    exp(-d² / eps²), and the result is made symmetric as (K + K.T) / 2, because base_dist need not
    be symmetric, e.g. when it comes from a k-nearest-neighbor graph.

    Objects 0 and 1 store their distance in both directions, so their kernel value is kept as is.
    Object 1 stores its distance to object 2, but object 2 does not store it back, so that value
    appears in both directions at half weight. Objects 0 and 2 store no distance, so they are not
    neighbors and their entries must be absent, not merely zero. eps = 2 rather than 1, so using
    eps instead of eps² changes the numbers.
    """
    D = np.array([[0, 1, 0],
                  [1, 0, 2],
                  [0, 0, 0]])
    stored = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], bool)
    eps = 2
    K = np.where(stored, np.exp(-D**2 / eps**2), 0)
    expected = (K + K.T) / 2

    result = build_base_kernel(HDMConfig(base_epsilon=eps), csr(D, stored))

    np.testing.assert_allclose(result.toarray(), expected)
    assert result[0, 2] == 0 and result.nnz == 7  # the 3 self-loops and both directions of 0-1 and 1-2


def test_build_base_kernel_estimates_epsilon():
    """With base_epsilon = None, eps is estimated as the median of the stored non-zero distances.
    The zero diagonal is left out, since it says nothing about how far apart neighbors are.

    The non-zero distances are 2, 2 and 4, so eps = 2. It is not 1, so an estimate that silently
    falls back to 1 would change the numbers.
    """
    base_dist = csr([[0, 2, 0],
                     [2, 0, 4],
                     [0, 0, 0]], stored=[[1, 1, 0], [1, 1, 1], [0, 0, 1]])
    expected = np.array([[1, np.exp(-1), 0],
                         [np.exp(-1), 1, np.exp(-4) / 2],
                         [0, np.exp(-4) / 2, 1]])

    result = build_base_kernel(HDMConfig(base_epsilon=None), base_dist)

    np.testing.assert_allclose(result.toarray(), expected)


def test_mapped_fiber_kernel():
    """_mapped_fiber_kernel compares point x of fiber i with point y of fiber j through the map M:
    it averages the distances from x's targets to y, weighted by M (that is M @ F), and turns the
    result into a kernel value w * exp(-d² / eps²), where w is the base weight of the pair.

    This test checks those values for a hard map (row 0, one target) and a soft map (row 1, two
    targets), with inputs chosen so that common mistakes change the result:
    - eps = 2 and weight = 0.5 rather than 1, so using eps instead of eps², or dropping the
      weight, gives different numbers;
    - the map is 2 x 3 rather than square, so a transposed M or F fails on its shape;
    - p0 and p1 coincide, so row 0 has a mapped distance of exactly 0 to two points. M @ F does
      not store sums that are exactly 0, so the function has to add them back with kernel value 1,
      and the nnz check makes sure they end up stored rather than missing.
    """
    F = np.abs(np.subtract.outer([0, 0, 2], [0, 0, 2]))  # points at x = 0, 0, 2: p0 and p1 coincide
    M = np.array([[1, 0, 0],       # hard: onto p0
                  [0, .5, .5]])    # soft: halfway between p1 and p2
    weight, eps = 0.5, 2.0
    expected = weight * np.exp(-(M @ F) ** 2 / eps**2)  # dense, so a mapped distance of 0 simply gives 1

    result = _mapped_fiber_kernel(sp.csr_matrix(M), csr(F), weight, eps)

    np.testing.assert_allclose(result.toarray(), expected)
    assert result.nnz == 6


def test_mapped_fiber_kernel_drops_pairs_missing_a_fiber_distance():
    """Comparing x with y needs the distance from every target of x to y. When one of them is not
    stored, M @ F silently counts it as 0, which would give the pair the largest possible kernel
    value, 1, even if the points are far apart. _mapped_fiber_kernel therefore drops such pairs.

    The fiber has points at x = 0, 1, 4, and the distance between p0 and p2 is not stored. Row 0
    maps onto p0 alone, so it keeps p0 and p1 and loses p2. Row 1 maps half onto p0 and half onto
    p2, so only p1, which has a stored distance from both, is kept. eps and the weight are 1 so
    that this test only fails when the dropping breaks; the test above covers the values.
    """
    dists = np.abs(np.subtract.outer([0, 1, 4], [0, 1, 4]))
    F = csr(dists, stored=dists < 4)             # d(p0, p2) = 4 is not stored
    M = sp.csr_matrix(np.array([[1, 0, 0],       # hard: onto p0
                                [.5, 0, .5]]))   # soft: half p0, half p2
    # row 0 keeps the points p0 has distances to: p0 and p1
    # row 1 keeps only p1, the one point with a stored distance from both p0 and p2: 0.5*1 + 0.5*3 = 2
    expected = np.array([[1, np.exp(-1), 0],
                         [0, np.exp(-4), 0]])

    result = _mapped_fiber_kernel(M, F, weight=1.0, eps=1.0)

    np.testing.assert_allclose(result.toarray(), expected)
    assert result.nnz == 3


def test_build_horizontal_diffusion_matrix():
    """build_horizontal_diffusion_matrix assembles the joint kernel over all points of all objects
    as a block matrix: block (i, j) holds the kernel between the points of object i and those of
    object j.

    This test compares every block with the paper's formula, computed densely with numpy:
    - each diagonal block is that fiber's own kernel exp(-F² / eps²), because we keep diffusion
      within an object (the paper sets these blocks to 0);
    - the block of two neighbors i, j is their base weight times the average of the kernel
      through maps[i, j] and the transposed kernel through maps[j, i], which also makes the whole
      matrix symmetric;
    - objects 0 and 2 are not neighbors, so their blocks must be absent, not merely zero.

    The fibers have 2, 3 and 2 points, so the blocks differ in size and any mistake in placing
    them shows. Every fiber distance is stored, so the dropping rule of _mapped_fiber_kernel keeps
    all pairs and the dense formula applies. The MapBundle mask allows only the four maps of the
    base edges, so reading any other map, such as maps[0, 2] or maps[i, i], raises a LookupError.
    """
    # three objects with fibers of 2, 3 and 2 points on a line; base edges 0-1 and 1-2, not 0-2
    xs = [np.array([0, 1]), np.array([0, 1, 2]), np.array([0, 2])]
    F = [np.abs(np.subtract.outer(x, x)).astype(float) for x in xs]
    base_kernel = csr([[1, .6, 0],
                       [.6, 1, .3],
                       [0, .3, 1]], stored=[[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    M = {(0, 1): [[1, 0, 0], [0, .5, .5]],
         (1, 0): [[1, 0], [1, 0], [0, 1]],
         (1, 2): [[1, 0], [0, 1], [0, 1]],
         (2, 1): [[1, 0, 0], [0, 0, 1]]}
    edges = np.zeros((3, 3), bool)
    edges[tuple(zip(*M))] = True  # any other map, including maps[i, i], raises LookupError if fetched
    maps = MapBundle.from_maps({ij: sp.csr_matrix(np.array(m, float)) for ij, m in M.items()}, [0, 1, 2], mask=edges)
    eps = 1.5

    def K(d):
        return np.exp(-np.asarray(d) ** 2 / eps**2)

    def block(i, j, w):
        # the paper's horizontal kernel, averaged over both map directions
        return w * (K(np.array(M[i, j]) @ F[j]) + K(np.array(M[j, i]) @ F[i]).T) / 2

    B01, B12 = block(0, 1, .6), block(1, 2, .3)
    expected = np.block([[K(F[0]), B01, np.zeros((2, 2))],
                         [B01.T, K(F[1]), B12],
                         [np.zeros((2, 2)), B12.T, K(F[2])]])

    result = build_horizontal_diffusion_matrix(
        HDMConfig(fiber_epsilon=eps), maps, base_kernel, [csr(f) for f in F], np.array([0, 2, 5, 7])
    )

    np.testing.assert_allclose(result.toarray(), expected)
    assert result[:2, 5:].nnz == 0 and result[5:, :2].nnz == 0  # no blocks without a base edge


def test_build_horizontal_diffusion_matrix_symmetrizes_fiber_kernels():
    """Fiber distances need not be symmetric: a k-nearest-neighbor graph can store d(p, q)
    without d(q, p). The diagonal blocks are the fiber kernels made symmetric as (K + K.T) / 2,
    so an entry stored in one direction only appears in both, at half weight, and the joint
    kernel stays symmetric, which the normalization and the eigensolver rely on.

    One object of 2 points stores d(p0, p1) = 1 but not d(p1, p0). With no other objects,
    the joint kernel is just that one diagonal block.
    """
    F = csr([[0, 1], [0, 0]], stored=[[1, 1], [0, 1]])
    maps = MapBundle.from_maps({}, [0], mask=np.zeros((1, 1), bool))
    expected = np.array([[1, np.exp(-1) / 2],
                         [np.exp(-1) / 2, 1]])

    result = build_horizontal_diffusion_matrix(HDMConfig(fiber_epsilon=1), maps, csr([[1]]), [F], np.array([0, 2]))

    np.testing.assert_allclose(result.toarray(), expected)


def test_compute_spectral_embedding():
    """compute_spectral_embedding turns the joint kernel W into the embeddings, which this test
    recomputes densely, step by step as in the paper:
    - Sinkhorn scales W symmetrically to Q = D W D, whose rows and columns all sum to 1;
    - the eigenvectors v_1, ..., v_k of Q after the trivial one (eigenvalue 1, constant vector)
      give the horizontal diffusion map of every point, HDM = v_l * λ_l^t;
    - the horizontal base diffusion map of object i has the entries λ_l^(t/2) λ_m^(t/2) <v_l[i], v_m[i]>,
      where v_l[i] are the rows of v_l that belong to object i (eq. 3.12), so that the inner product of
      two objects' embeddings is the squared norm of their block of Q^t (eq. 3.11);
    - HBDD holds the distances between those embeddings (eq. 3.14).

    Eigenvectors are only defined up to sign, so the dense ones are flipped to match before
    comparing HDM and HBDM; HBDD does not depend on the signs. W is a random symmetric kernel with a
    strong diagonal, so its leading eigenvalues are positive and distinct, and t = 2 rather than 1,
    so mixing up λ^t and λ^(t/2) changes the numbers.
    """
    n, k, t = 6, 3, 2.0
    offsets = np.array([0, 2, 6])  # two objects, of 2 and 4 points
    A = np.random.default_rng(0).random((n, n))
    W = np.eye(n) + 0.3 * (A + A.T) / 2

    d = np.ones(n)
    for _ in range(10_000):
        d = np.sqrt(d / (W @ d))
    Q = d[:, None] * W * d[None, :]
    np.testing.assert_allclose(Q.sum(axis=1), 1)
    lam, V = np.linalg.eigh(Q)
    lam, V = lam[::-1][1 : k + 1], V[:, ::-1][:, 1 : k + 1]
    assert (np.diff(-lam) > 1e-3).all() and (lam > 0).all()  # distinct and positive, as the test needs

    result = compute_spectral_embedding(HDMConfig(num_eigenvectors=k, t=t, verbose=False), sp.csr_matrix(W), offsets, 2)

    signs = np.sign((result.eigvecs * V).sum(axis=0))
    V = V * signs
    HBDM = np.array([((lam ** (t / 2) * V[a:b]).T @ (lam ** (t / 2) * V[a:b])).ravel()
                     for a, b in zip(offsets[:-1], offsets[1:])])
    np.testing.assert_allclose(result.eigvals, lam, rtol=1e-6)
    np.testing.assert_allclose(result.HDM, V * lam**t, atol=1e-6)
    np.testing.assert_allclose(result.HBDM, HBDM, atol=1e-6)
    np.testing.assert_allclose(result.HBDD, np.linalg.norm(HBDM[:, None] - HBDM[None], axis=-1), atol=1e-6)
