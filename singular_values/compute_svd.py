# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault
import numpy as np
from scipy.stats import ortho_group


def computeTruncatedSVD(W, n):
    """

    :param W: A matrix of size N x M
    :param n: The dimension of the truncation
    :return: The truncated singular value decomposition of W, i.e.,
             Un Sn Vn^dagger where Un is N x n, Sn is n x n and Vn is M x n.

    Note: This is a naive algorithm to do it, since we compute the full SVD.
    """
    U, S, Vh = np.linalg.svd(W)
    return U[:, :n], np.diag(S[:n]), Vh[:n, :]


def computeTruncatedSVD_more_positive(W, n):
    """

    :param W: A matrix of size N x M
    :param n: The dimension of the truncation
    :return: The truncated singular value decomposition of W, i.e.,
             Un Sn Vn^dagger where Un is N x n, Sn is n x n and Vn is M x n.
             However, we multiply Vn by a diagonal matrix in such a way that we
             have more positive values in V than negative values
    """
    Un, Sn, Vhn = computeTruncatedSVD(W, n)
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    return Un@D_sign, Sn, D_sign@Vhn


def generate_random_rank_r_matrix_from_singvals(singvals):
    """

    :param singvals: (N-dim array) The desired singular value
     distribution. The number of non zero entries in singvals is the rank r.
    :return: A random rank r (real) matrix.
    """
    N = len(singvals)
    U = ortho_group.rvs(N)
    Vh = ortho_group.rvs(N)
    return U@np.diag(singvals)@Vh
