# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from dynamics.dynamics import qmf_sis
from dynamics.reduced_dynamics import reduced_qmf_sis
from dynamics.error_vector_fields import *
from graphs.compute_tensors import *
from graphs.get_real_networks import *
from scipy.linalg import pinv
from singular_values.compute_svd import computeTruncatedSVD_more_positive

plot_weight_matrix_bool = False


def test_relative_error_vector_fields():
    """ Graph parameters """
    A = get_epidemiological_weight_matrix("high_school_proximity")
    N = len(A[0])  # Dimension of the complete dynamics

    """ Dynamical parameters """
    t = 0  # It is not involved in the error computation
    D = np.eye(N)

    """ SVD and dimension reduction """
    n = 10  # Dimension of the reduced dynamics
    Un, Sn, Vhn = computeTruncatedSVD_more_positive(A, n)
    L, M = Un@Sn, Vhn
    W = A/Sn[0][0]  # We normalize the network by the largest singular value
    Mp = pinv(M)
    calD = M@D@Mp
    calW, calW_tensor3 = M@W@Mp, compute_tensor_order_3(M, W)

    """ Evaluate the errors at 2 points"""
    # After the threshold there should be a non zero error at n = 10
    x = np.random.random(N)
    coupling = 2
    args_qmf_sis = (t, x, W, coupling, D)
    args_reduced_qmf_sis = (t, M@x, calW, calW_tensor3, coupling, calD)

    err1 = relative_error_vector_fields(M, qmf_sis, reduced_qmf_sis,
                                        args_qmf_sis, args_reduced_qmf_sis)

    # There should be a zero error at x = 0, despite the division by zero in
    # the denominator of the relative error (coupling arbitrary)
    x = np.zeros(N)
    args_qmf_sis = (t, x, W, coupling, D)
    args_reduced_qmf_sis = (t, M@x, calW, calW_tensor3, coupling, calD)
    err2 = relative_error_vector_fields(M, qmf_sis, reduced_qmf_sis,
                                        args_qmf_sis, args_reduced_qmf_sis)
    # print(err1, err2)
    assert err1 >= 0 and np.allclose(err2, 0)


def test_x_prime_Jx_Jy_SIS():
    N = 100
    n = 10
    x = np.random.random(N)
    D = np.eye(N)
    coupling = 2
    A = np.random.uniform(-1, 1, (N, N))
    U, S, Vh = np.linalg.svd(A)
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign@Vhn
    W = A/S[0]  # We normalize the network by the largest singular value
    Mp = pinv(M)
    P = Mp@M

    Jx = jacobian_x_SIS(x_prime_SIS(x, W, M), W, coupling, D)
    Jy = jacobian_y_SIS(x_prime_SIS(x, W, M), coupling)
    chi = (np.eye(N) - P)@x

    LHS_taylor = qmf_sis(0, x, W, coupling, D)
    RHS_taylor = qmf_sis(0, P@x, W, coupling, D) + Jx@chi + Jy@W@chi

    assert np.allclose(LHS_taylor, RHS_taylor,)


if __name__ == "__main__":
    pytest.main()
