"""Shared configuration and golden-output helpers for the end-to-end tests.

The tests are regression tests: run_hdm is run on a small saved bundle and its
outputs are compared against saved goldens. run_hdm's raw eigenvector-based
coordinates are only defined up to per-eigenvector sign (and up to rotation
within degenerate eigenspaces), so they are not compared directly. Instead we
compare three quantities that are invariant to that ambiguity and small enough
to store:

    eigvals            - the retained eigenvalues
    hbdd               - the fibre-to-fibre HBDD distance matrix (n_samples^2)
    hdm_singular_values - singular values of the HDM coordinate matrix

The generators and the tests import the same config and golden_outputs so the
comparison is guaranteed to use the exact settings that produced the goldens.
"""
import numpy as np

TEETH_CONFIG = dict(
    base_knn=4,
    num_eigenvectors=4,
    base_epsilon=0.03,
    device="cpu",
    seed=42,
    verbose=False,
)

WINGS_CONFIG = dict(
    base_knn=7,
    num_eigenvectors=6,
    base_epsilon=None,
    device="cpu",
    seed=42,
    verbose=False,
)


def golden_outputs(result):
    return dict(
        eigvals=np.asarray(result.eigvals, dtype=np.float64),
        hbdd=np.asarray(result.HBDD, dtype=np.float64),
        hdm_singular_values=np.linalg.svd(np.asarray(result.HDM, dtype=np.float64), compute_uv=False),
    )


def assert_matches(result, expected, rtol=1e-5, atol=1e-7):
    got = golden_outputs(result)
    for key in ("eigvals", "hbdd", "hdm_singular_values"):
        np.testing.assert_allclose(
            got[key], expected[key], rtol=rtol, atol=atol,
            err_msg=f"{key} does not match the saved golden output",
        )
