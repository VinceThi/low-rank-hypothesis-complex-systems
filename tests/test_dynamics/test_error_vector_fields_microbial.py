# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from dynamics.dynamics import microbial
from dynamics.error_vector_fields import *
from graphs.get_real_networks import *


# def test_crude_x_prime_Jx_Jy_microbial():
#     """ We test the very crude approximation of x_prime for microbial in
#     the specific context of the gut microbiome network used in the paper. """
#     A = get_microbiome_weight_matrix("gut")
#     N = len(A[0])  # Dimension of the complete dynamics
#     n = 10
#     x = np.random.random(N)
#     D = 0.01*np.eye(N)
#     a, b, c = 0.00005, 0.1, 0.9
#     coupling = 3
#     U, S, Vh = np.linalg.svd(A)
#     Vhn = Vh[:n, :]
#     D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
#         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
#     M = D_sign @ Vhn
#     W = A / S[0]  # We normalize the network by the largest singular value
#     Mp = np.linalg.pinv(M)
#     P = Mp@M
#     xp = x_prime_microbial(x, W, P, coupling, b, c)
#     Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
#     Jy = jacobian_y_microbial(xp, coupling)
#     chi = (np.eye(N) - P)@x
#
#     LHS_taylor = microbial(0, x, W, coupling, D, a, b, c)
#     RHS_taylor = microbial(0, P@x, W, coupling, D, a, b, c)+Jx@chi + Jy@W@chi
#
#     print(np.linalg.norm(LHS_taylor-RHS_taylor))
#     print(np.sum((np.abs(LHS_taylor-RHS_taylor))/N))
#     print(LHS_taylor-RHS_taylor)
#
#     assert np.allclose(LHS_taylor, RHS_taylor)


# def test_optimized_x_prime_Jx_Jy_microbial_random():
#     """ We test the x_prime obtained through optimization for the microbial
#      dynamics in the specific context of the gut microbiome network used in
#      the paper. """
#     N = 15
#     A = np.random.random((N, N))
#     # A = get_microbiome_weight_matrix("gut")
#     # N = len(A[0])  # Dimension of the complete dynamics
#     n = 1
#     x = np.random.random(N)
#     # D = 0.01*np.eye(N), this is not used
#     a, b, c = 0.00005, 0.1, 0.9
#     coupling = 3
#     U, S, Vh = np.linalg.svd(A)
#     Vhn = Vh[:n, :]
#     D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
#         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
#     M = D_sign@Vhn
#     W = A/S[0]  # We normalize the network by the largest singular value
#     Mp = np.linalg.pinv(M)
#     P = Mp@M
#     xp = x_prime_microbial_optimize(x, W, P, coupling, b, c)[0]
#     evaluated_objective_function_microbial = \
#         objective_function_microbial(xp, x, W, P, coupling, b, c)
#     tol = 1e-6
#     print(evaluated_objective_function_microbial)
#     assert np.all(evaluated_objective_function_microbial < tol)


def test_optimized_x_prime_Jx_Jy_microbial_gut():
    """ We test the x_prime obtained through optimization for the microbial
     dynamics in the specific context of the gut microbiome network used in
     the paper. """
    A = get_microbiome_weight_matrix("gut")
    N = len(A[0])  # Dimension of the complete dynamics
    n = 1
    x = np.random.random(N)
    # D = 0.01*np.eye(N), this is not used
    a, b, c = 0.00005, 0.1, 0.9
    coupling = 3
    U, S, Vh = np.linalg.svd(A)
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign@Vhn
    W = A/S[0]  # We normalize the network by the largest singular value
    Mp = np.linalg.pinv(M)
    P = Mp@M
    xp = x_prime_microbial_optimize(x, W, P, coupling, b, c)[0]
    evaluated_objective_function_microbial = \
        objective_function_microbial(xp, x, W, P, coupling, b, c)
    tol = 1e-6
    # ^ we have a pretty high tolerance to errors. In many case,
    # the elements of evaluated_objective_function_microbial are at the
    # numerical zero
    print(evaluated_objective_function_microbial)
    # if evaluated_objective_function_microbial < 1e-13:
    #     print("The elements of evaluated_objective_function_microbial"
    #           "are at the numerical zero.")
    # elif evaluated_objective_function_microbial < tol:
    #     print("The zero found with scipy.optimize.root is ok for the sake"
    #           "of the test and the paper. "
    #           "It might be a local minimum near zero")
    assert np.all(evaluated_objective_function_microbial < tol)

# def test_optimized_x_prime_Jx_Jy_microbial_2():
#     """ We test the x_prime obtained through optimization for the microbial
#      dynamics in the specific context of the gut microbiome network used in
#      the paper. """
#     N = 15
#     A = np.random.random((N, N))
#     # A = get_microbiome_weight_matrix("gut")
#     # N = len(A[0])  # Dimension of the complete dynamics
#     n = 10
#     z = np.random.random(N)
#     D = 0.01*np.eye(N)
#     a, b, c = 0.00005, 0.1, 0.9
#     coupling = 3
#     U, S, Vh = np.linalg.svd(A)
#     Vhn = Vh[:n, :]
#     D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
#         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
#     M = D_sign@Vhn
#     W = A/S[0]  # We normalize the network by the largest singular value
#     Mp = np.linalg.pinv(M)
#     P = Mp@M
#     xp = x_prime_microbial_optimize(z, W, P, coupling, b, c)
#     print(f"x' = {xp}")
#     print("f(x') = ", objective_function_microbial(xp, z, W, P, coupling, b, c))
#     Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
#     Jy = jacobian_y_microbial(xp, coupling)
#     chi = (np.eye(N) - P)@z
#
#     LHS_taylor = microbial(0, z, W, coupling, D, a, b, c)
#     RHS_taylor = microbial(0, P@z, W, coupling, D, a, b,
#                            c) + Jx@chi + Jy@W@chi
#
#     # print(np.linalg.norm(LHS_taylor - RHS_taylor))
#     # print(np.sum((np.abs(LHS_taylor - RHS_taylor)) / N))
#     print(f"LHS_taylor = {LHS_taylor}")
#     print(f"RHS_taylor = {RHS_taylor}")
#     diff = LHS_taylor - RHS_taylor
#     print(f"LHS_taylor - RHS_taylor = {diff}")
#     print(f"mean(LHS_taylor - RHS_taylor) = {np.mean(np.abs(diff))}")
#     print(np.mean(np.abs(np.random.random(N) - np.random.random(N))))
#
#     assert np.allclose(LHS_taylor, RHS_taylor)
#     # assert np.all(np.abs(diff) < 1) and np.mean(np.abs(diff)) < 0.4


if __name__ == "__main__":
    pytest.main()
