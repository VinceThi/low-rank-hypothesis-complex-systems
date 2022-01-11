# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import pytest
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.reduced_dynamics import reduced_kuramoto_sakaguchi
from graphs.compute_tensors import compute_tensor_order_4


def test_complete_vs_reduced_kuramoto_sakaguchi_full():
    """ Compares the complete Kuramoto-Sakaguchi dynamics with the reduced one
     for n = N and heterogeneous parameters. """
    t = 10
    N = 5
    alpha = np.pi/4
    theta = np.random.uniform(0, 2*np.pi, N)
    z = np.exp(1j*theta)
    W = np.random.random((N, N))
    coupling = 1
    D = np.diag(np.random.random(N))
    U, s, M = np.linalg.svd(W)
    calD = M@D@np.linalg.pinv(M)
    calW = M@W@np.linalg.pinv(M)
    calW_tensor4 = compute_tensor_order_4(M, W)
    Mdotz = M@(1j*z*kuramoto_sakaguchi(t, theta, W, coupling, D, alpha))
    dotZ = reduced_kuramoto_sakaguchi(t, M@z, calW, calW_tensor4,
                                      coupling, calD, alpha, N)
    assert np.allclose(Mdotz, dotZ)


def test_complete_vs_reduced_kuramoto_sakaguchi_rank():
    """ Compares the complete Kuramoto-Sakaguchi dynamics with the reduced one
     for n = rank(W), the rank of the weight matrix W and the parameters matrix
     D is correlated to W. TODO precise "correlated" """
    t = 10
    N = 3
    alpha = np.pi/4
    theta = np.random.uniform(0, 2*np.pi, N)
    z = np.exp(1j*theta)
    W = np.array([[0, 0.2, 0.1],
                  [0.5, 0.4, 0.2],
                  [0.5, 0.4, 0.2]])
    rankW = 2
    coupling = 1
    D = np.diag([3, 2, 2])
    # D = np.diag([2, 2, 2])
    U, s, Vh = np.linalg.svd(W)
    M = Vh[:rankW, :]
    calD = M@D@np.linalg.pinv(M)
    calW = M@W@np.linalg.pinv(M)
    calW_tensor4 = compute_tensor_order_4(M, W)
    Mdotz = M@(1j*z*kuramoto_sakaguchi(t, theta, W, coupling, D, alpha))
    dotZ = reduced_kuramoto_sakaguchi(t, M@z, calW, calW_tensor4,
                                      coupling, calD, alpha, N)
    print(Mdotz, dotZ)
    print(np.linalg.norm(M@W@(np.eye(N)-np.linalg.pinv(M)@M)))
    print(np.linalg.norm(M@D@(np.eye(N)-np.linalg.pinv(M)@M)))
    print(np.linalg.norm(W @ (np.eye(N) - np.linalg.pinv(M) @ M)))
    print(np.linalg.norm(D @ (np.eye(N) - np.linalg.pinv(M) @ M)))
    print(np.linalg.norm(Mdotz - dotZ))

    assert np.allclose(Mdotz, dotZ)


if __name__ == "__main__":
    pytest.main()
