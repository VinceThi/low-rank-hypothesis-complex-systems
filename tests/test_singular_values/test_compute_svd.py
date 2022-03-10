# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from singular_values.compute_svd import *
import pytest
from plots.plot_weight_matrix import plot_weight_matrix
from plots.plot_singular_values import plot_singular_values


def test_D_sign_in_computeTruncatedSVD_more_positive():
    Vh = np.array([[1, -2, 0],
                   [3, 0, -2],
                   [2, -2, 0]])
    D_sign_true = np.array([[-1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    DVh = np.array([[-1, 2, 0],
                    [3, 0, -2],
                    [2, -2, 0]])
    D_sign_test = np.diag(-(np.sum(Vh, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vh, axis=1) >= 0).astype(float))
    assert np.allclose(D_sign_true, D_sign_test)\
        and np.all(DVh == D_sign_test@Vh)


def test_computeTruncatedSVD_vs_computeTruncatedSVD_more_positive():
    N = 100
    n = 10
    W = np.random.uniform(-1, 1, (N, N))
    Un, Sn, Vhn = computeTruncatedSVD(W, n)
    UnD, Sn, DVhn = computeTruncatedSVD_more_positive(W, n)
    assert np.allclose(Un@Sn@Vhn, UnD@Sn@DVhn) and np.sum(Vhn) < np.sum(DVhn)


def test_generate_randomly_rank_r_matrix_from_singvals():
    N = 100
    rank = 20
    singvals = np.block([np.sqrt(np.linspace(20, 30, rank))[::-1],
                         np.zeros(N-rank)])
    W = generate_random_rank_r_matrix_from_singvals(singvals)
    plot_weight_matrix(W)
    plot_singular_values(singvals)
    assert np.allclose(np.linalg.matrix_rank(W), rank)


if __name__ == "__main__":
    pytest.main()
