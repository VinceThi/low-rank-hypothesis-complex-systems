# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from observables.matrix_characteristic import matrix_is_singular
from tqdm import tqdm


def onmf(M, max_iter=500, W_init=None, H_init=None):
    """
    Orthogonal Non-negative Matrix Factorization of X as X =WH wit HH^T=I.
    Based on Ref. Wang, Y. X., & Zhang, Y. J. (2012).
    Nonnegative matrix factorization: A comprehensive review.
    IEEE Transactions on Knowledge and Data Engineering, 25(6), 1336-1353.

    and

    https://github.com/mstrazar/iONMF

    ----------
    Input
    ----------
    M: array [n x N]
        Data matrix to be factorized.
    max_iter: int
        Maximum number of iterations.
    H_init: array [n x n]
        Fixed initial basis matrix.
    W_init: array [n x N]
        Fixed initial coefficient matrix.
    MoreOrtho: Boolean
        If True, searches for a matrix H with more zeros
    ---------
    Output
    ---------
    W: array [n x n]
    H: array [n x N]
    error: ||X-WH||/(nN)^2
        normalized factorization error
    o_error:  ||I-HH^T||/(n^2)
        normalized orthogonality error

    ex: SVD initialization
    n,N = X.shape
    # SVD
    u,s,vh = np.linalg.svd(X)
    # Initial matrix H
    h_init= abs(vh[0:n,:])
    # NMF
    W,H,e,oe = onmf(X, H_init = h_init)

    """

    n, N = np.shape(M)

    # add a small value, otherwise nmf and related methods get
    # into trouble as they have difficulties recovering from zero.
    W = np.random.random((n, n)) + 10**(-4) if isinstance(W_init, type(None))\
        else W_init
    H = np.random.random((n, N)) + 10**(-4) if isinstance(H_init, type(None))\
        else H_init

    for itr in range(max_iter):
        # update H
        numerator = W.T@M
        denominator = H@M.T@W@H
        H = np.nan_to_num(H*numerator/denominator)

        # new lines added to get orthonormalized rows
        row_norm = np.sqrt(np.diag(H@H.T))
        normalization_matrix = np.linalg.inv(np.diag(row_norm))
        H = normalization_matrix@H

        # update W
        numerator = M@H.T
        denominator = W@H@H.T
        W = np.nan_to_num(W*numerator/denominator)

    # error with normalized Frobenius norm
    error = np.linalg.norm(M - W@H)/(n*N)**2

    # orthogonality error with Frobenius norm
    o_error = np.linalg.norm(np.eye(n, n) - H@H.T)/(n**2)

    # ---------- Normalized frobenius error and normalized orthogonal error
    return W, H, error, o_error


def onmf_multiple_inits(M, number_initializations, print_errors=False):
    """
    Notation: W -> F   and   H -> G
    """
    n, N = np.shape(M)

    """ ---------------------------- SVD ---------------------------------- """
    u, s, vh = np.linalg.svd(M)

    # Initial matrix H with SVD
    G_init = np.absolute(vh[0:n, :])

    # Ortogonal nonnegative matrix factorization
    # with SVD initialization
    F_svd, G_svd, frobenius_error_svd, ortho_error_svd \
        = onmf(M, H_init=G_init)
    F, G = F_svd, G_svd
    onmf_frobenius_error, onmf_ortho_error = \
        frobenius_error_svd, ortho_error_svd
    if print_errors:
        print(f"\nonmf_frobenius_error_svd = {onmf_frobenius_error}",
              f"\nonmf_ortho_error_svd = {onmf_ortho_error}")

    """ --------------------------- Random -------------------------------- """
    print("\n Iterating on random matrix initializations for ONMF... \n")

    for j in tqdm(range(number_initializations)):
        # Ortogonal nonnegative matrix factorization
        # with random initialization
        F_random, G_random, frobenius_error_random, ortho_error_random \
            = onmf(M, H_init=None)   # S'assurer que c'est ok

        # 1. The condition below is the one used for transitions vs. n in the
        #    reply to the referee. The errors are normalized.
        # if frobenius_error_random**2 + ortho_error_random**2 < \
        #         onmf_frobenius_error**2 + onmf_ortho_error**2:

        # 2. We can penalize the orthogonal errors by adding weights if we want
        # if 0.1*frobenius_error_random**2 + 0.9*ortho_error_random**2 < \
        #         0.1*onmf_frobenius_error**2 + 0.9*onmf_ortho_error**2:
        #

        # 3. The condition below is the one used for FIG. 6 and 7 of the paper
        # if frobenius_error_random < onmf_frobenius_error:

        if frobenius_error_random**2 + ortho_error_random**2 < \
                onmf_frobenius_error**2 + onmf_ortho_error**2:
            F, G, = F_random, G_random
            onmf_frobenius_error, onmf_ortho_error = \
                frobenius_error_random, ortho_error_random
            # print("Result improved by a random initialization !")
            if print_errors:
                print(f"onmf_frobenius_error_random = {onmf_frobenius_error}",
                      f"\nonmf_ortho_error_random = {onmf_ortho_error}")

    if matrix_is_singular(F):
        ValueError('F is singular.')

    # print("onmf_frobenius_error_final = ", onmf_frobenius_error,
    #       "\nonmf_ortho_error_final = ", onmf_ortho_error)

    # ---------- Normalized frobenius error  and normalized ortho errors
    return F, G, onmf_frobenius_error, onmf_ortho_error
