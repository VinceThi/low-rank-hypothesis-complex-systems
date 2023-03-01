# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.generate_s1_random_graph import *
import pytest


def test_s1_model():
    N = 3
    beta = 2
    kappa_in = np.array([4, 5, 7])
    kappa_out = np.array([11, 3, 2])
    theta = 2*np.pi*np.array([0.1, 0.6, 0.2])  # 2*np.pi*uniform.rvs(size=N)
    directed = True
    W, valid_expected_weight_matrix = \
        s1_model(beta, kappa_in, kappa_out, theta,
                 directed=directed, expected=True)

    EW = np.zeros((N, N))
    nb_instances = 100000
    for i in range(nb_instances):
        W = s1_model(beta, kappa_in, kappa_out, theta,
                     directed=directed, expected=False)
        EW = EW + W/nb_instances

    print(valid_expected_weight_matrix, "\n", EW)

    assert np.all((EW - valid_expected_weight_matrix) < 1e-2)


def test_generate_nonnegative_arrays_with_same_average():
    a = [0, 1, 1]
    b = [1, 2, 3]
    a, b = generate_nonnegative_arrays_with_same_average(a, b)

    c = [9, 5, 7, 8, 4, 9]
    d = [2, 1, 3, 2, 2, 1]
    c, d = generate_nonnegative_arrays_with_same_average(c, d)

    assert np.mean(a) == np.mean(b) and np.mean(c) == np.mean(d)


def test_thetaij_triangle_inequality():
    N = 1000
    theta = 2*np.pi*uniform.rvs(size=N)
    thetaij = np.absolute(theta.reshape(-1, 1) - theta)
    thetaij = np.pi - np.absolute(np.pi - thetaij)

    for k in range(N):
        row = thetaij[k, :].reshape(1, N)
        col = thetaij[:, k].reshape(N, 1)
        assert np.all(-1e-10 <= (row + col) - thetaij)


if __name__ == "__main__":
    pytest.main()
