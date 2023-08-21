# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.generate_s1_random_graph import *
from singular_values.compute_effective_ranks import computeRank
from plots.plot_singular_values import plot_singular_values
from scipy.linalg import svdvals
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


def test_thetaij_rank():
    """ Test Theorem 7 of Gower 1985 for the angular distance matrix
    (dimensionality = 1). The rank is thus less or equal to 3.
    The computation of the rank is very sensible to the dispersion for
    "theta". Very sensible to errors for the computation of the singular values
    """
    plot_singvals = False
    N = 1000
    # theta = 2*np.pi*prng.uniform(size=N)
    # ^^^ Very sensible to errors for the computation of the singular values
    theta = 2*np.pi/np.random.randint(1, 30, N)
    thetaij = np.absolute(theta.reshape(-1, 1) - theta)
    thetaij = np.pi - np.absolute(np.pi - thetaij)

    # s = np.abs(np.linalg.eigvalsh(thetaij**2 ))
    # s = np.linalg.svd(thetaij**2 , compute_uv=False, hermitian=True)
    singularValues = svdvals(thetaij**2)

    if plot_singvals:
        plot_singular_values(thetaij**2, effective_ranks=False,
                             cum_explained_var=True)

    assert computeRank(singularValues, tolerance=1e-5) <= 3


if __name__ == "__main__":
    pytest.main()
