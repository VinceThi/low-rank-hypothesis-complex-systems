# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from singular_values.optimal_shrinkage import optimal_shrinkage,\
    optimal_threshold
from numpy.random import RandomState
import numpy as np

# The values below must not be changed for the test of optimal_shrinkage
N = 100
rank_r = 20
beta = 1
prng = RandomState(1234567890)
L = prng.uniform(0, 0.5, (N, rank_r))
M = prng.normal(0, 0.5, (rank_r, N))
sigma = 2.5
noise_level = sigma/np.sqrt(N)
W = L@M + noise_level*prng.normal(0, 0.2, (N, N))
# Note that we violate the unit variance assumption of Gavish,Donoho 2014, 2017
# It does not matter for the test below, we test for unknown noise.
U, S, Vh = np.linalg.svd(W)
# from scipy.io import savemat
# from plots.plot_weight_matrix import plot_weight_matrix
# from plots.plot_singular_values import plot_singular_values
# plot_weight_matrix(W)
# savemat("matrix_test_optimal_shrinkage.mat",{"W": W, "label": "test_matrix"})
#  plot_singular_values(S)


def test_optimal_shrinkage_frobenius_unknown_noise():

    shrinked_singvals_Python = np.round(optimal_shrinkage(S, beta, 'fro'), 1)
    shrinked_singvals_Matlab = np.round(np.array(
        [62.6968, 10.3039, 10.0925, 9.3524, 9.1710, 8.4156, 8.3366, 7.4465,
         7.3157, 7.0097, 6.4597, 6.0529, 5.9396, 5.3009, 4.8881, 4.8049,
         4.4511, 4.2319, 3.9716, 3.4962, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0]), 1)

    assert np.allclose(shrinked_singvals_Python, shrinked_singvals_Matlab)


def test_optimal_shrinkage_operator_unknown_noise():
    shrinked_singvals_Python = np.round(optimal_shrinkage(S, beta, 'op'), 1)
    shrinked_singvals_Matlab = np.round(np.array(
        [62.7021, 10.3358, 10.1250,  9.3875,  9.2068,  8.4546,  8.3760,
         7.4905, 7.3605, 7.0564, 6.5103, 6.1068, 5.9945, 5.3624,
         4.9546, 4.8725, 4.5239, 4.3084, 4.0529, 3.5880, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1)

    assert np.allclose(shrinked_singvals_Python, shrinked_singvals_Matlab)


def test_optimal_shrinkage_nuclear_unknown_noise():
    shrinked_singvals_Python = np.round(optimal_shrinkage(S, beta, 'nuc'), 1)
    shrinked_singvals_Matlab = np.round(
        np.array([62.6916, 10.2720, 10.0600,  9.3173,  9.1352,  8.3767,
                  8.2973,  7.4025,  7.2710,  6.9631,  6.4091,  5.9989,
                  5.8846,  5.2395,  4.8217,  4.7372,  4.3783,  4.1554,
                  3.8903,  3.4044, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]), 1)

    assert np.allclose(shrinked_singvals_Python, shrinked_singvals_Matlab)


def test_optimal_thresold_frobenius_unknown_noise():
    hard_threshold_Python = np.round(optimal_threshold(S, 1), 2)
    hard_threshold_Matlab = np.round(1.3255, 2)
    assert np.allclose(hard_threshold_Python, hard_threshold_Matlab)


if __name__ == "__main__":
    pytest.main()
