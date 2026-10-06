import numpy as np
import scipy.sparse as sp


def csr(A, stored=None):
    """CSR matrix of A storing the entries where stored is True, all of them by default. Unlike
    sp.csr_matrix(A), it keeps explicit zeros, such as the zero diagonal of a distance matrix."""
    A = np.asarray(A, dtype=float)
    stored = np.ones(A.shape, bool) if stored is None else np.asarray(stored)
    r, c = np.nonzero(stored)
    return sp.csr_matrix((A[r, c], (r, c)), shape=A.shape)
