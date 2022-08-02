# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from dynamics.dynamics import microbial
from dynamics.error_vector_fields import *
from graphs.get_real_networks import *


""" Global parameters """
matrix_str = "gut"  # "gut" or "random_uniform"
setup = 2   # This is what we use for the paper (see errors_microbial.py)
if matrix_str == "random":
    N = 50
    adj = np.random.uniform(-1, 1, (N, N))
elif matrix_str == "gut":
    adj = get_microbiome_weight_matrix("gut")
    N = len(adj[0])  # Dimension of the complete dynamics
else:
    raise ValueError("The matrix_str is 'gut' or 'random'.")

n = 500
U, S, Vh = np.linalg.svd(adj)
Vhn = Vh[:n, :]
D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
         + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))

if setup == 1:
    # 1. The one used in SI and in Sanhedrai et al., Nat. Phys. (2022).
    a, b, c, D = 5, 13, 10/3, 30*np.eye(N)
    coupling = np.random.uniform(0.5, 3)
    W = adj
    singvals_W = S
    max_x = 20

elif setup == 2:
    # 2. To have trajectories between 0 and 1 approximately
    a, b, c = 0.00005, 0.1, 0.9
    D = 0.01 * np.eye(N)
    coupling = np.random.uniform(0.5, 3)
    W = adj / S[0]  # We normalize the network by the largest singular value
    singvals_W = S/S[0]
    max_x = 1

else:
    ValueError("setup must be 1 or 2")

x = np.random.uniform(0, max_x, N)
M = D_sign @ Vhn
Mp = np.linalg.pinv(M)
P = Mp@M
p = P@x

""" ---------------------- Tests for approximated x'----------------------- """


def test_x_prime_approx_x_Px_microbial():
    """
    We test the approximation that x' is x or Px.
    'n' is very important in this test. If 'n' is high enough, x and Px
    should be close enough to the x' used to evaluate the error bound.
    """

    Jx = jacobian_x_microbial(x, W, coupling, D, b, c)
    Jy = jacobian_y_microbial(x, coupling)

    Jx_tilde = jacobian_x_microbial(P @ x, W, coupling, D, b, c)
    Jy_tilde = jacobian_y_microbial(P @ x, coupling)
    e_x = error_vector_fields_upper_bound(x, Jx, Jy, singvals_W, M, P)
    e_Px = error_vector_fields_upper_bound(P @ x, Jx_tilde, Jy_tilde,
                                           singvals_W, M, P)

    chi = (np.eye(N) - P)@x

    LHS_taylor = microbial(0, x, W, coupling, D, a, b, c)
    if e_Px < e_x:
        RHS_taylor = microbial(0, P@x, W, coupling, D, a, b, c)\
            + Jx@chi + Jy@W@chi
    else:
        RHS_taylor = microbial(0, P@x, W, coupling, D, a, b, c) \
                     + Jx_tilde@chi + Jy_tilde@W@chi

    # assert np.allclose(LHS_taylor, RHS_taylor, rtol=1e-1, atol=1e-1)
    print("max_x_Px: |LHS - RHS| = ", np.mean(np.abs(LHS_taylor - RHS_taylor)))
    # assert np.all(np.abs(LHS_taylor - RHS_taylor) < 0.3)
    print("max_x_Px: MSE(LHS - RHS) = ", np.mean((LHS_taylor - RHS_taylor)**2))
    assert np.mean((LHS_taylor - RHS_taylor)**2) < 0.2

# def test_x_prime_microbial_neglect_linear_term():
#     """
#         We test the approximation of x_prime for the microbial dynamics for
#         which we neglect the whole coupling term B in the objective function.
#     """
#     xp = x_prime_microbial_neglect_linear_term(A, C)
#     Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
#     Jy = jacobian_y_microbial(xp, coupling)
#     chi = (np.eye(N) - P)@x
# 
#     LHS_taylor = microbial(0, x, W, coupling, D, a, b, c)
#     RHS_taylor = microbial(0, P@x, W, coupling, D, a, b, c)+Jx@chi + Jy@W@chi
# 
#     MSE_LHS_RHS = np.mean((LHS_taylor-RHS_taylor)**2)
#     # print(f"Neglect B")
#     # print(f"LHS_taylor-RHS_taylor = {LHS_taylor-RHS_taylor}")
#     # print(f"MSE_LHS_RHS = {MSE_LHS_RHS}")
#     # assert np.allclose(LHS_taylor, RHS_taylor)
#     if matrix_str == "gut":
#         assert MSE_LHS_RHS < 1e-1
#     else:
#         print("The test was not applied for this matrix.")
# 
# 
# def test_x_prime_microbial_neglect_coupling():
#     """
#         We test the approximation of x_prime for the microbial dynamics for
#      which we neglect the coupling term only in B in the objective function.
#     """
#     xp = x_prime_microbial_neglect_coupling(x, W, P, coupling, b, c)
#     Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
#     Jy = jacobian_y_microbial(xp, coupling)
#     chi = (np.eye(N) - P)@x
# 
#     LHS_taylor = microbial(0, x, W, coupling, D, a, b, c)
#     RHS_taylor = microbial(0, P@x, W, coupling, D, a, b, c)+Jx@chi + Jy@W@chi
# 
#     MSE_LHS_RHS = np.mean((LHS_taylor-RHS_taylor)**2)
#     print(f"\nNeglect coupling")
#     print(f"LHS_taylor-RHS_taylor = {LHS_taylor-RHS_taylor}")
#     print(f"MSE_LHS_RHS = {MSE_LHS_RHS}")
#     # assert np.allclose(LHS_taylor, RHS_taylor)
#     assert MSE_LHS_RHS < 1e-1

#
# def test_Jx_Jy_approx_microbial():
#     """
#     'n' is very important in this test. If 'n' is high enough, x and Px
#     should be close enough to the x' used to evaluate the error bound.
#     """
#     a, b, c = 0.00005, 0.1, 0.9
#     coupling = 3
#
#     Jx = jacobian_x_microbial(x, W, coupling, D, b, c)
#     Jy = jacobian_y_microbial(x, coupling)
#     e_x = error_vector_fields_upper_bound(x, Jx, Jy, S, M, P)
#
#     Jx_tilde = jacobian_x_microbial(P@x, W, coupling, D, b, c)
#     Jy_tilde = jacobian_y_microbial(P@x, coupling)
#     e_Px = error_vector_fields_upper_bound(P@x, Jx_tilde, Jy_tilde,
#                                            S, M, P)
#     chi = (np.eye(N) - P)@x
#
#     LHS_taylor = microbial(0, x, W, coupling, D, a, b, c)
#     if e_Px < e_x:
#         RHS_taylor = microbial(0, P@x, W, coupling, D, a, b, c)\
#             + Jx@chi + Jy@W@chi
#     else:
#         RHS_taylor = microbial(0, P@x, W, coupling, D, a, b, c) \
#                      + Jx_tilde@chi + Jy_tilde@W@chi
#
#     # assert np.allclose(LHS_taylor, RHS_taylor, rtol=1e-1, atol=1e-1)
#     # print(np.abs(LHS_taylor - RHS_taylor) < 1e-1)
#     # assert np.all(np.abs(LHS_taylor - RHS_taylor) < 0.3)
#     # print(np.mean((LHS_taylor - RHS_taylor)**2), 0.1*np.max(LHS_taylor),
#     #       np.mean((LHS_taylor - RHS_taylor)**2)/np.mean(LHS_taylor**2),
#     #       np.mean((np.abs(LHS_taylor - RHS_taylor))/np.abs(LHS_taylor)))
#     assert np.mean((LHS_taylor - RHS_taylor)**2) < 1e-2
#     relative_error = np.linalg.norm(LHS_taylor - RHS_taylor) / \
#         np.linalg.norm(LHS_taylor)
#     assert relative_error < 0.2


""" ------------------ Tests for x' find with optimizaton ----------------- """

A = A_microbial(x, p, c)
B = B_microbial(x, p, W, b, coupling)
C = C_microbial(x, p, W, b, c, coupling)

xp = x_prime_microbial_optimize(A, B, C, max_x=max_x)


def test_objective_function_microbial():
    # xp = x_prime_microbial_optimize(x, W, P, coupling, b, c)
    obj_fct = objective_function_microbial(xp, A, B, C)

    Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
    Jy = jacobian_y_microbial(xp, coupling)
    chi = (np.eye(N) - P) @ x

    LHS = (Jx + Jy @ W)@chi
    RHS = microbial(0, x, W, coupling, D, a, b, c) \
        - microbial(0, P@x, W, coupling, D, a, b, c)

    assert np.allclose(obj_fct, LHS-RHS)


def test_optimized_x_prime_microbial():
    """ We test the x_prime obtained through optimization for the microbial
     dynamics. """
    # xp = x_prime_microbial_optimize(x, W, P, coupling, b, c)
    obj_fct = objective_function_microbial(xp, A, B, C)
    MSE_evaluated_obj_fct = np.mean(obj_fct**2)
    print("evaluated objective function = ", obj_fct)
    print("MSE evaluated objective function = ", MSE_evaluated_obj_fct)

    if matrix_str == "random":
        assert np.all(xp < 1) and np.all(xp > 0) and \
               np.allclose(obj_fct, np.zeros(len(x)), rtol=1e-1, atol=1e-1)
    elif matrix_str == "gut":
        assert np.all(xp < 1) and np.all(xp > 0) and \
               MSE_evaluated_obj_fct < 1e-2


def test_optimized_x_prime_Jx_Jy_microbial():
    """ We test the x_prime obtained through optimization for the microbial
     dynamics with the Taylor theorem. We also assert that the elements
     of x' are between 0 and 1. """

    Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
    Jy = jacobian_y_microbial(xp, coupling)
    chi = (np.eye(N) - P)@x

    LHS_taylor = microbial(0, x, W, coupling, D, a, b, c)
    RHS_taylor = microbial(0, P@x, W, coupling, D, a, b, c) + Jx@chi + Jy@W@chi

    obj_fct = objective_function_microbial(xp, A, B, C)
    MSE_obj_fct = np.mean(obj_fct**2)
    if matrix_str == "random":
        assert np.all(xp < 1) and np.all(xp > 0) and \
            np.allclose(LHS_taylor, RHS_taylor, rtol=1e-1, atol=1e-1)
    elif matrix_str == "gut":
        assert np.all(xp < 1) and np.all(xp > 0) and MSE_obj_fct < 1e-2


# test_objective_function_microbial()
if __name__ == "__main__":
    pytest.main()
