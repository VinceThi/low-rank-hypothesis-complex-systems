# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import pytest
from dynamics.dynamics import qmf_sis
from dynamics.reduced_dynamics import reduced_qmf_sis
from graphs.compute_tensors import compute_tensor_order_3


def test_complete_vs_reduced_qmf_sis_full():
    """ Compares the complete quenched mean-field SIS dynamics with
    the reduced one for n = N and heterogeneous parameters. """
    t = 10
    N = 5
    x = np.random.random(N)
    W = np.random.random((N, N))
    coupling = 1
    D = np.diag(np.random.random(N))
    U, s, M = np.linalg.svd(W)
    calD = M@D@np.linalg.pinv(M)
    calW = M@W@np.linalg.pinv(M)
    calW_tensor3 = compute_tensor_order_3(M, W)
    Mdotx = M@qmf_sis(t, x, W, coupling, D)
    dotX = reduced_qmf_sis(t, M@x, calW, calW_tensor3, coupling, calD, N)
    assert np.allclose(Mdotx, dotX)


def test_complete_vs_reduced_qmf_sis_rank():
    """ Compares the complete the complete quenched mean-field SIS dynamics
     with the reduced one for n = rank(W), the rank of the weight matrix W
      and the parameters matrix D is correlated to W.
       TODO precise "correlated" """
    t = 10
    N = 3
    x = np.array([0.2, 0.8, 0.1])
    W = np.array([[0, 0.2, 0.1],
                  [0.5, 0.4, 0.2],
                  [0.5, 0.4, 0.2]])
    rankW = 2
    coupling = 1
    D = np.diag([3, 2, 2])
    U, s, Vh = np.linalg.svd(W)
    M = Vh[:rankW, :]
    calW_tensor3 = compute_tensor_order_3(M, W)
    calD = M@D@np.linalg.pinv(M)
    calW = M @ W @ np.linalg.pinv(M)
    Mdotx = M@qmf_sis(t, x, W, coupling, D)
    dotX = reduced_qmf_sis(t, M@x, calW, calW_tensor3, coupling, calD, N)
    assert np.allclose(Mdotx, dotX)


if __name__ == "__main__":
    pytest.main()
