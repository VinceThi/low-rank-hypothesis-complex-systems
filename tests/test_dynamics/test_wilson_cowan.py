# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import pytest
from dynamics.dynamics import wilson_cowan
from dynamics.reduced_dynamics import reduced_wilson_cowan_vector_field


def test_complete_vs_reduced_wilson_cowan_full():
    """ Compares the complete Wilson-Cowan dynamics with the reduced one
     for n = N and heterogeneous parameters. """
    t = 10
    N = 5
    x = np.random.random(N)
    W = np.random.random((N, N))
    coupling = 1
    D = np.diag(np.random.random(N))
    a, b, c = 1, 1, 3
    U, s, M = np.linalg.svd(W)
    Mp = np.linalg.pinv(M)
    Mdotx = M@wilson_cowan(t, x, W, coupling, D, a, b, c)
    dotX = reduced_wilson_cowan_vector_field(t, M@x, W, coupling,
                                             M, Mp, D, a, b, c)
    assert np.allclose(Mdotx, dotX)


def test_complete_vs_reduced_wilson_cowan_rank_identical_params_a_0():
    """ Compares the complete Wilson-Cowan dynamics with the reduced one
     for n = rank(W), the rank of the weight matrix W and identical
     parameters. We set a = 0, which essentially gives the dynamics
     of a recurrent neural network. """
    t = 10
    x = np.array([0.2, 0.8, 0.1])
    W = np.array([[0, 0.2, -0.8],
                  [0.5, -0.4, 0.2],
                  [0.5, -0.4, 0.2]])
    rankW = 2
    coupling = 1
    D = np.diag([2, 2, 2])
    a, b, c = 0, 1, 3
    U, s, Vh = np.linalg.svd(W)
    M = Vh[:rankW, :]
    Mp = np.linalg.pinv(M)
    Mdotx = M@wilson_cowan(t, x, W, coupling, D, a, b, c)
    dotX = reduced_wilson_cowan_vector_field(t, M@x, W, coupling,
                                             M, Mp, D, a, b, c)
    assert np.allclose(Mdotx, dotX)


def test_complete_vs_reduced_wilson_cowan_rank_special():
    """ Compares the complete Wilson-Cowan dynamics with the reduced one
     for n = rank(W), the rank of the weight matrix W and the parameters matrix
     D is correlated to W. TODO precise "correlated" """

    """ WARNING: This is a very particular case that works."""
    t = 10
    x = np.array([0.2, 0.8, 0.1])
    W = np.array([[0, 0.2, 0.1],
                  [0.5, 0.4, 0.2],
                  [0.5, 0.4, 0.2]])
    rankW = 2
    coupling = 1
    D = np.diag([3, 2, 2])
    a, b, c = 1, 1, 3
    U, s, Vh = np.linalg.svd(W)
    M = Vh[:rankW, :]
    Mp = np.linalg.pinv(M)
    Mdotx = M@wilson_cowan(t, x, W, coupling, D, a, b, c)
    dotX = reduced_wilson_cowan_vector_field(t, M@x, W, coupling,
                                             M, Mp, D, a, b, c)
    assert np.allclose(Mdotx, dotX)


if __name__ == "__main__":
    pytest.main()
