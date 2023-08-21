# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from dynamics.dynamics import qmf_sis
from dynamics.error_vector_fields import *
from graphs.get_real_networks import *
from scipy.linalg import pinv


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
    M = D_sign @ Vhn
    W = A / S[0]  # We normalize the network by the largest singular value
    Mp = pinv(M)
    P = Mp @ M

    Jx = jacobian_x_SIS(x_prime_SIS(x, W, P), W, coupling, D)
    Jy = jacobian_y_SIS(x_prime_SIS(x, W, P), coupling)
    chi = (np.eye(N) - P)@x

    LHS_taylor = qmf_sis(0, x, W, coupling, D)
    RHS_taylor = qmf_sis(0, P@x, W, coupling, D) + Jx@chi + Jy@W@chi

    assert np.allclose(LHS_taylor, RHS_taylor)


if __name__ == "__main__":
    pytest.main()
