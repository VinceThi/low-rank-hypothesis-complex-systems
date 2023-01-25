# Authors: Christian Thurau and Vincent Thibeault
# License: BSD 3 Clause
"""
PyMF Semi Non-negative Matrix Factorization.
    SNMF(NMF) : Class for semi non-negative matrix factorization

[1] Ding, C., Li, T. and Jordan, M..
    Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55.
"""
import numpy as np
from matrix_factorization.pymf_base import PyMFBase
from observables.matrix_characteristic import matrix_is_singular

__all__ = ["SNMF"]


class SNMF(PyMFBase):
    """
    SNMF(data, H_init=H_init, W_init=W_init, num_bases=4)

    Semi Non-negative Matrix Factorization. Factorize a data matrix into two
    matrices s.t. F = | data - W*H | is minimal. For Semi-NMF only H is
    constrained to non-negativity.

    Parameters
    ----------
    - data : array_like, shape (_data_dimension, _num_samples)
        the input data
    - num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)

    Attributes
    ----------
    - W : "data_dimension x num_bases" matrix of basis vectors
    - H : "num bases x num_samples" matrix of coefficients
    - ferr : frobenius norm (after calling .factorize())

    Example
    -------
    Applying Semi-NMF to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.factorize(niter=10)

    The basis vectors are now stored in snmf_mdl.W, the coefficients in
    snmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply copy W
    to snmf_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.W = W
    >>> snmf_mdl.factorize(niter=1, compute_w=False)

    The result is a set of coefficients snmf_mdl.H, s.t. data = W * snmf_mdl.H.
    """

    def _update_w(self):
        W1 = np.dot(self.data[:, :], self.H.T)
        W2 = np.dot(self.H, self.H.T)
        # if np.abs(np.linalg.det(W2)) < 1e-8:
        #     raise ValueError("A matrix in the snmf (W2) is singular !")
        # else:
        #     self.W = np.dot(W1, np.linalg.inv(W2))

    def _update_h(self):
        def separate_positive(m):
            return (np.abs(m) + m) / 2.0

        def separate_negative(m):
            return (np.abs(m) - m) / 2.0

        XW = np.dot(self.data[:, :].T, self.W)

        WW = np.dot(self.W.T, self.W)
        WW_pos = separate_positive(WW)
        WW_neg = separate_negative(WW)

        XW_pos = separate_positive(XW)
        H1 = (XW_pos + np.dot(self.H.T, WW_neg)).T

        XW_neg = separate_negative(XW)
        H2 = (XW_neg + np.dot(self.H.T, WW_pos)).T + 10**(-9)

        self.H *= np.sqrt(H1 / H2)


def snmf(data, num_bases, niter=500, W_init=None, H_init=None):
    """
    SNMF: Semi-nonnegative matrix factorization
    :param data: matrix to be factorized
    :param niter: number of iteration in the algorithm, 100 iterations is a
                  safe number of iterations, see Ding 2010
    :param H_init: initialization of the positive matrix
    :param W_init: initialization of the coefficient matrix
    :return: coefficient matrix W, positive matrix H, normalized frobenius err.
    """
    n, N = np.shape(data)
    facto = SNMF(data, H_init=H_init, W_init=W_init, num_bases=num_bases)
    facto.factorize(niter=niter)
    return facto.W, facto.H, facto.ferr[-1]/(n*N)


def snmf_multiple_inits(data, num_bases, number_initializations=1000):

    u, s, vh = np.linalg.svd(data)

    # Initial matrix H with SVD
    H_init = np.absolute(vh[0:num_bases, :])

    # SNMF with SVD initialization
    W_svd, H_svd, frobenius_error_svd =\
        snmf(data, H_init=H_init, num_bases=num_bases)
    W, H = W_svd, H_svd
    snmf_frobenius_error = frobenius_error_svd

    for j in range(number_initializations):
        # SNMF with random initialization
        W_random, H_random, frobenius_error_random = \
            snmf(data, num_bases=num_bases, H_init=None)
        if snmf_frobenius_error > frobenius_error_random:
            W, H, = W_random, H_random
            snmf_frobenius_error = frobenius_error_random

    return W, H, snmf_frobenius_error


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":

    data = np.array([[-0.0375, -0.0375,  0.0725,  0.3418,  0.3304,  0.3304],
                     [0.3304,  0.3304,  0.3418,  0.0725, -0.0375, -0.0375],
                     [0.3,  -0.3304,  0.3418,  -0.0725, 0.0375, -0.01]])
    d1, d2 = np.shape(data)

    # SVD
    u, s, vh = np.linalg.svd(data)

    # Initial matrix H with absolute values of right singular vectors SVD
    num_base = 6    # k in Ding et al. 2010
    h_init = abs(vh[0:num_base, :]).reshape((num_base, d2))

    snmf_mdl = SNMF(data, H_init=h_init, num_bases=num_base)
    snmf_mdl.factorize(niter=1000)

    print(f"H = {snmf_mdl.H} \n")
    print(f"W = {snmf_mdl.W} \n")
    print(f"WH = {snmf_mdl.W@snmf_mdl.H} \n")

    W1, H1, snmf_frob_error = snmf_multiple_inits(data, num_base,
                                                  number_initializations=100)

    print(f"H = {H1} \n")
    print(f"W = {W1} \n")
    print(f"WH = {W1@H1} \n")
    print(f"error = {snmf_frob_error} \n")
