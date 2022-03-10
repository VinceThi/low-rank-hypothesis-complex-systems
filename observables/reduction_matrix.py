# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from numpy.linalg import pinv
from observables.matrix_characteristic import *
from matrix_factorization.snmf import snmf_multiple_inits
from matrix_factorization.onmf import onmf_multiple_inits


def normalize_rows_matrix_M1(M):
    return (M.T / np.sum(M, axis=1)).T


def normalize_rows_complex_matrix_M1(M):
    return (M.T / np.sum(M, axis=1)).T


def normalize_rows_matrix_MM_T(M):
    return (M.T / np.sqrt(np.sum(M**2, axis=1))).T


def get_reduced_parameter_matrix(M, X):
    return M@X@pinv(M)


def get_reduction_matrix_snmf_onmf(V_T, number_initializations=500):
    """
    Get the reduction matrix M for the dimension-reduction.

    :param V_T: Target matrix
    :param number_initializations:
            Number of different initializations of the semi and the
            orthogonal nonnegative matrix factorization (SNMF and ONMF).
            If niter=1, the algorithm will initialize SNMF and ONMF with SVD.
            If niter>1, the algorithm will initialize SNMF and ONMF with SVD in
            the first iteration and then, it will try random
            initializations to find the lowest Frobenius norm error
            frobenius_error = ||M - WH||.

    :return: M : n x N positive array/matrix. np.sum(M[mu,:], axis=1) = 1
                 for all mu which means that the matrix is normalized according
                 to its rows (the sum over the columns is one for each row)
    """
    if matrix_is_negative(V_T):
        F_snmf, G_snmf, snmf_frobenius_error = \
            snmf_multiple_inits(V_T, number_initializations)
        print("\nsnmf_ferr = ", snmf_frobenius_error)
        M_possibly_not_ortho = G_snmf
    else:
        M_possibly_not_ortho = V_T
        snmf_frobenius_error = None

    if not matrix_is_orthogonal(M_possibly_not_ortho):
        # If the matrix is not already orthogonal...
        F_onmf, M_not_normalized, onmf_frobenius_error, onmf_ortho_error =\
            onmf_multiple_inits(M_possibly_not_ortho,
                                number_initializations)
        # import matplotlib.pyplot as plt
        print(f"\nonmf_ferr = {onmf_frobenius_error} ",
              f"\nonmf_oerr = {onmf_ortho_error}")

    else:
        M_not_normalized = M_possibly_not_ortho
        onmf_frobenius_error, onmf_ortho_error = None, None

    M = normalize_rows_matrix_M1(M_not_normalized)

    if not matrix_is_positive(M):
        raise ValueError("The reduced matrix M is not positive anymore after"
                         "using orthonormal matrix factorization.")

    return M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error

