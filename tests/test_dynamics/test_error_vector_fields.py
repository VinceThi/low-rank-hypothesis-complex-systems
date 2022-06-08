# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from dynamics.dynamics import qmf_sis, rnn, wilson_cowan, microbial
# from dynamics.reduced_dynamics import reduced_rnn_vector_field
from dynamics.error_vector_fields import *
# from graphs.compute_tensors import *
from graphs.get_real_networks import *
from scipy.linalg import pinv
# from singular_values.optimal_shrinkage import optimal_shrinkage
# from singular_values.compute_effective_ranks import computeRank


# def test_x_prime_Jx_Jy_SIS():
#     N = 100
#     n = 10
#     x = np.random.random(N)
#     D = np.eye(N)
#     coupling = 2
#     A = np.random.uniform(-1, 1, (N, N))
#     U, S, Vh = np.linalg.svd(A)
#     Vhn = Vh[:n, :]
#     D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
#         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
#     M = D_sign @ Vhn
#     W = A / S[0]  # We normalize the network by the largest singular value
#     Mp = pinv(M)
#     P = Mp @ M
#
#     Jx = jacobian_x_SIS(x_prime_SIS(x, W, P), W, coupling, D)
#     Jy = jacobian_y_SIS(x_prime_SIS(x, W, P), coupling)
#     chi = (np.eye(N) - P)@x
#
#     LHS_taylor = qmf_sis(0, x, W, coupling, D)
#     RHS_taylor = qmf_sis(0, P@x, W, coupling, D) + Jx@chi + Jy@W@chi
#
#     assert np.allclose(LHS_taylor, RHS_taylor)
#
#
# def test_dsig_prime_Jx_Jy_rnn():
#     N = 100
#     n = 10
#     x = np.random.random(N)
#     D = np.eye(N)
#     coupling = 2
#     A = np.random.uniform(-1, 1, (N, N))
#     U, S, Vh = np.linalg.svd(A)
#     Vhn = Vh[:n, :]
#     D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
#         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
#     M = D_sign @ Vhn
#     W = A / S[0]
#     Mp = pinv(M)
#     P = Mp @ M
#
#     dsig_prime = derivative_sigmoid_prime_rnn(x, W, P, coupling)
#     Jx = -D
#     Jy = jacobian_y_rnn(dsig_prime, coupling)
#     chi = (np.eye(N) - P)@x
#
#     LHS_taylor = rnn(0, x, W, coupling, D)
#     RHS_taylor = rnn(0, P@x, W, coupling, D) + Jx@chi + Jy@W@chi
#
#     assert np.allclose(LHS_taylor, RHS_taylor)
#
#
# def test_dsig_prime_approx_Jx_Jy_wilson_cowan():
#     N = 100
#     n = 10
#     x = np.random.random(N)
#     D = np.eye(N)
#     coupling = 2
#     A = np.random.uniform(-1, 1, (N, N))
#     U, S, Vh = np.linalg.svd(A)
#     Vhn = Vh[:n, :]
#     D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
#         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
#     M = D_sign @ Vhn
#     W = A
#     Mp = pinv(M)
#     P = Mp@M
#     a, b, c = 0.1, 1, 3
#     # 'a = 0.1' is a "pretty high" value of 'a'. To have a successful test,
#     # one should choose a smaller. Of course, for a smaller 'a', smaller
#     # tolerances can be chosen in np.allclose
#
#     dsig_prime = derivative_sigmoid_prime_wilson_cowan(x, W, P, coupling, b, c)
#     Jx_approx = -D
#     Jy_approx = jacobian_y_approx_wilson_cowan(dsig_prime, coupling, b)
#     chi = (np.eye(N) - P)@x
#
#     # import matplotlib.pyplot as plt
#     # plt.figure()
#     # ax1 = plt.subplot(311)
#     # ax1.matshow(Jy_approx)
#     # ax2 = plt.subplot(312)
#     # ax2.matshow(M)
#     # ax3 = plt.subplot(313)
#     # ax3.matshow(M@Jy_approx)
#     # plt.show()
#
#     LHS_taylor = wilson_cowan(0, x, W, coupling, D, a, b, c)
#     RHS_taylor = wilson_cowan(0, P@x, W, coupling, D, a, b, c)\
#         + Jx_approx@chi + Jy_approx@W@chi
#
#     assert np.allclose(LHS_taylor, RHS_taylor, rtol=1e-1, atol=1e-1)
#
#
# def test_Jx_Jy_approx_wilson_cowan():
#     N = 100
#     n = 50
#     # 'n' is very important in this test. If 'n' is high enough, x and Px
#     # should be close to the best x' to choose for the error bound
#     x = np.random.random(N)
#     D = np.eye(N)
#     coupling = 2
#     A = np.random.uniform(-1, 1, (N, N))
#     U, S, Vh = np.linalg.svd(A)
#     Vhn = Vh[:n, :]
#     D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
#         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
#     M = D_sign @ Vhn
#     W = A
#     Mp = pinv(M)
#     P = Mp@M
#     a, b, c = 0.1, 1, 3
#
#     Jx = jacobian_x_wilson_cowan(x, W, coupling, D, a, b, c)
#     Jy = jacobian_y_wilson_cowan(x, W, coupling, a, b, c)
#     e_x = error_vector_fields_upper_bound(x, Jx, Jy, S, M, P)
#
#     Jx_tilde = jacobian_x_wilson_cowan(P@x, W, coupling, D, a, b, c)
#     Jy_tilde = jacobian_y_wilson_cowan(P@x, W, coupling, a, b, c)
#     e_Px = error_vector_fields_upper_bound(P@x, Jx_tilde, Jy_tilde,
#                                            S, M, P)
#     chi = (np.eye(N) - P)@x
#
#     LHS_taylor = wilson_cowan(0, x, W, coupling, D, a, b, c)
#     if e_Px < e_x:
#         RHS_taylor = wilson_cowan(0, P@x, W, coupling, D, a, b, c)\
#             + Jx@chi + Jy@W@chi
#     else:
#         RHS_taylor = wilson_cowan(0, P@x, W, coupling, D, a, b, c) \
#                      + Jx_tilde@chi + Jy_tilde@W@chi
#
#     assert np.allclose(LHS_taylor, RHS_taylor, rtol=1e-1, atol=1e-1)
#
#
# def test_x_prime_Jx_Jy_wilson_cowan():
#     N = 100
#     n = 10
#     x = np.random.random(N)
#     D = np.eye(N)
#     coupling = 2
#     a, b, c = 1, 1, 3
#     A = np.random.uniform(-1, 1, (N, N))
#     U, S, Vh = np.linalg.svd(A)
#     Vhn = Vh[:n, :]
#     D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
#         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
#     M = D_sign @ Vhn
#     W = A / S[0]  # We normalize the network by the largest singular value
#     Mp = pinv(M)
#     P = Mp@M
#
#     xp = x_prime_wilson_cowan(x, W, P, coupling, a, b, c)
#     Jx = jacobian_x_wilson_cowan(xp, W, coupling, D, a, b, c)
#     Jy = jacobian_y_wilson_cowan(xp, W, coupling, a, b, c)
#     chi = (np.eye(N) - P)@x
#
#     LHS_taylor = wilson_cowan(0, x, W, coupling, D, a, b, c)
#     RHS_taylor = wilson_cowan(0, P@x, W, coupling, D, a, b, c)+Jx@chi+Jy@W@chi
#
#     assert np.all(xp < 1) and np.all(xp > 0) and \
#         np.allclose(LHS_taylor, RHS_taylor, rtol=1e-2, atol=1e-2)
#
#
# def test_error_vector_fields_upper_bound_RNN():
#     """ Graph parameters """
#     A = get_learned_weight_matrix("mouse_control_rnn")
#     N = len(A[0])  # Dimension of the complete dynamics
#
#     """ Dynamical parameters """
#     t = 0  # Time is not involved in the vector-field error computation
#     D = 2*np.eye(N)
#     coupling = 4
#
#     """ SVD """
#     U, S, Vh = np.linalg.svd(A)
#     shrink_s = optimal_shrinkage(S, 1, 'operator')
#     W = U@np.diag(shrink_s)@Vh
#
#     n = computeRank(shrink_s)
#     Vhn = Vh[:n, :]
#     D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
#         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
#     M = D_sign @ Vhn
#     Mp = pinv(M)
#     P = Mp@M
#     x = np.random.uniform(0, 1, N)
#
#     error = rmse(M@rnn(t, x, W, coupling, D),
#                  reduced_rnn_vector_field(t, M@x, W, coupling, M, Mp, D))
#     dsigp = derivative_sigmoid_prime_rnn(x, W, P, coupling)
#     Jx = -D
#     Jy = jacobian_y_rnn(dsigp, coupling)
#     error_upper_bound =\
#         error_vector_fields_upper_bound(x, Jx, Jy, shrink_s, M, P)
#     assert np.allclose(error, error_upper_bound)


def test_x_prime_Jx_Jy_microbial():
    """ We test the very crude approximation of x_prime for microbial in
    the specific context of the gut microbiome network used in the paper. """
    A = get_microbiome_weight_matrix("gut")
    N = len(A[0])  # Dimension of the complete dynamics
    n = 10
    x = np.random.random(N)
    D = 0.01*np.eye(N)
    a, b, c = 0.00005, 0.1, 0.9
    coupling = 3
    U, S, Vh = np.linalg.svd(A)
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign @ Vhn
    W = A / S[0]  # We normalize the network by the largest singular value
    Mp = np.linalg.pinv(M)
    P = Mp@M
    xp = x_prime_microbial(x, W, P, coupling, b, c)
    Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
    Jy = jacobian_y_microbial(xp, coupling)
    chi = (np.eye(N) - P)@x

    LHS_taylor = microbial(0, x, W, coupling, D, a, b, c)
    RHS_taylor = microbial(0, P@x, W, coupling, D, a, b, c) + Jx@chi + Jy@W@chi

    print(np.linalg.norm(LHS_taylor-RHS_taylor))
    print(np.sum((np.abs(LHS_taylor-RHS_taylor))/N))
    print(LHS_taylor-RHS_taylor)

    assert np.allclose(LHS_taylor, RHS_taylor)


# def test_relative_error_vector_fields():
#     """ Graph parameters """
#     A = get_epidemiological_weight_matrix("high_school_proximity")
#     N = len(A[0])  # Dimension of the complete dynamics
#
#     """ Dynamical parameters """
#     t = 0  # It is not involved in the error computation
#     D = np.eye(N)
#
#     """ SVD and dimension reduction """
#     n = 10  # Dimension of the reduced dynamics
#     Un, Sn, Vhn = computeTruncatedSVD_more_positive(A, n)
#     L, M = Un@Sn, Vhn
#     W = A/Sn[0][0]  # We normalize the network by the largest singular value
#     Mp = pinv(M)
#     calD = M@D@Mp
#     calW, calW_tensor3 = M@W@Mp, compute_tensor_order_3(M, W)
#
#     """ Evaluate the errors at 2 points"""
#     # After the threshold there should be a non zero error at n = 10
#     x = np.random.random(N)
#     coupling = 2
#     args_qmf_sis = (t, x, W, coupling, D)
#     args_reduced_qmf_sis = (t, M@x, calW, calW_tensor3, coupling, calD)
#
#     err1 = relative_error_vector_fields(M, qmf_sis, reduced_qmf_sis,
#                                         args_qmf_sis, args_reduced_qmf_sis)
#
#     # There should be a zero error at x = 0, despite the division by zero in
#     # the denominator of the relative error (coupling arbitrary)
#     x = np.zeros(N)
#     args_qmf_sis = (t, x, W, coupling, D)
#     args_reduced_qmf_sis = (t, M@x, calW, calW_tensor3, coupling, calD)
#     err2 = relative_error_vector_fields(M, qmf_sis, reduced_qmf_sis,
#                                         args_qmf_sis, args_reduced_qmf_sis)
#     # print(err1, err2)
#     assert err1 >= 0 and np.allclose(err2, 0)


if __name__ == "__main__":
    # pytest.main()
    test_x_prime_Jx_Jy_microbial()
