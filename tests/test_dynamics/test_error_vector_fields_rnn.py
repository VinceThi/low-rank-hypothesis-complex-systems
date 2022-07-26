# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from dynamics.dynamics import rnn
from dynamics.reduced_dynamics import reduced_rnn_vector_field
from singular_values.compute_effective_ranks import *
from dynamics.error_vector_fields import *
from graphs.get_real_networks import *
from scipy.linalg import pinv


def test_dsig_prime_Jx_Jy_rnn():
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
    W = A / S[0]
    Mp = pinv(M)
    P = Mp@M

    dsig_prime = derivative_sigmoid_prime_rnn(x, W, P, coupling)
    Jx = -D
    Jy = jacobian_y_rnn(dsig_prime, coupling)
    chi = (np.eye(N) - P)@x

    LHS_taylor = rnn(0, x, W, coupling, D)
    RHS_taylor = rnn(0, P@x, W, coupling, D) + Jx@chi + Jy@W@chi

    assert np.allclose(LHS_taylor, RHS_taylor)


def test_error_vector_fields_upper_bound_RNN():
    """ Graph parameters """
    A = get_learned_weight_matrix("mouse_control_rnn")
    N = len(A[0])  # Dimension of the complete dynamics

    """ Dynamical parameters """
    t = 0  # Time is not involved in the vector-field error computation
    D = 2*np.eye(N)
    coupling = 4

    """ SVD """
    U, S, Vh = np.linalg.svd(A)
    shrink_s = optimal_shrinkage(S, 1, 'operator')
    W = U@np.diag(shrink_s)@Vh

    n = computeRank(shrink_s)
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign @ Vhn
    Mp = pinv(M)
    P = Mp@M
    x = np.random.uniform(0, 1, N)

    error = rmse(M@rnn(t, x, W, coupling, D),
                 reduced_rnn_vector_field(t, M@x, W, coupling, M, Mp, D))
    dsigp = derivative_sigmoid_prime_rnn(x, W, P, coupling)
    Jx = -D
    Jy = jacobian_y_rnn(dsigp, coupling)
    error_upper_bound =\
        error_vector_fields_upper_bound(x, Jx, Jy, shrink_s, M, P)
    assert np.allclose(error, error_upper_bound)


if __name__ == "__main__":
    pytest.main()
