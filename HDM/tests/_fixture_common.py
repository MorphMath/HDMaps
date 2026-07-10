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
