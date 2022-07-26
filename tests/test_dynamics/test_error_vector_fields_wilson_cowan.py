# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from dynamics.dynamics import wilson_cowan
from dynamics.error_vector_fields import *
from graphs.get_real_networks import *
from scipy.linalg import pinv


def test_dsig_prime_approx_Jx_Jy_wilson_cowan():
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
    M = D_sign @ Vhn
    W = A
    Mp = pinv(M)
    P = Mp@M
    a, b, c = 0.1, 1, 3
    # 'a = 0.1' is a "pretty high" value of 'a'. To have a successful test,
    # one should choose a smaller. Of course, for a smaller 'a', smaller
    # tolerances can be chosen in np.allclose

    dsig_prime = derivative_sigmoid_prime_wilson_cowan(x, W, P, coupling, b, c)
    Jx_approx = -D
    Jy_approx = jacobian_y_approx_wilson_cowan(dsig_prime, coupling, b)
    chi = (np.eye(N) - P)@x

    # import matplotlib.pyplot as plt
    # plt.figure()
    # ax1 = plt.subplot(311)
    # ax1.matshow(Jy_approx)
    # ax2 = plt.subplot(312)
    # ax2.matshow(M)
    # ax3 = plt.subplot(313)
    # ax3.matshow(M@Jy_approx)
    # plt.show()

    LHS_taylor = wilson_cowan(0, x, W, coupling, D, a, b, c)
    RHS_taylor = wilson_cowan(0, P@x, W, coupling, D, a, b, c)\
        + Jx_approx@chi + Jy_approx@W@chi

    assert np.allclose(LHS_taylor, RHS_taylor, rtol=1e-1, atol=1e-1)


def test_Jx_Jy_approx_wilson_cowan():
    N = 100
    n = 50
    # 'n' is very important in this test. If 'n' is high enough, x and Px
    # should be close to the best x' to choose for the error bound
    x = np.random.random(N)
    D = np.eye(N)
    coupling = 2
    A = np.random.uniform(-1, 1, (N, N))
    U, S, Vh = np.linalg.svd(A)
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign @ Vhn
    W = A
    Mp = pinv(M)
    P = Mp@M
    a, b, c = 0.1, 1, 3

    Jx = jacobian_x_wilson_cowan(x, W, coupling, D, a, b, c)
    Jy = jacobian_y_wilson_cowan(x, W, coupling, a, b, c)
    e_x = error_vector_fields_upper_bound(x, Jx, Jy, S, M, P)

    Jx_tilde = jacobian_x_wilson_cowan(P@x, W, coupling, D, a, b, c)
    Jy_tilde = jacobian_y_wilson_cowan(P@x, W, coupling, a, b, c)
    e_Px = error_vector_fields_upper_bound(P@x, Jx_tilde, Jy_tilde,
                                           S, M, P)
    chi = (np.eye(N) - P)@x

    LHS_taylor = wilson_cowan(0, x, W, coupling, D, a, b, c)
    if e_Px < e_x:
        RHS_taylor = wilson_cowan(0, P@x, W, coupling, D, a, b, c)\
            + Jx@chi + Jy@W@chi
    else:
        RHS_taylor = wilson_cowan(0, P@x, W, coupling, D, a, b, c) \
                     + Jx_tilde@chi + Jy_tilde@W@chi

    assert np.allclose(LHS_taylor, RHS_taylor, rtol=1e-1, atol=1e-1)


def test_x_prime_Jx_Jy_wilson_cowan():
    N = 100
    n = 10
    x = np.random.random(N)
    D = np.eye(N)
    coupling = 2
    a, b, c = 1, 1, 3
    A = np.random.uniform(-1, 1, (N, N))
    U, S, Vh = np.linalg.svd(A)
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign @ Vhn
    W = A / S[0]  # We normalize the network by the largest singular value
    Mp = pinv(M)
    P = Mp@M

    xp = x_prime_wilson_cowan(x, W, P, coupling, a, b, c)
    Jx = jacobian_x_wilson_cowan(xp, W, coupling, D, a, b, c)
    Jy = jacobian_y_wilson_cowan(xp, W, coupling, a, b, c)
    chi = (np.eye(N) - P)@x

    LHS_taylor = wilson_cowan(0, x, W, coupling, D, a, b, c)
    RHS_taylor = wilson_cowan(0, P@x, W, coupling, D, a, b, c)+Jx@chi+Jy@W@chi

    assert np.all(xp < 1) and np.all(xp > 0) and \
        np.allclose(LHS_taylor, RHS_taylor, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main()
