# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from optht import optht
import numpy as np
from scipy.linalg import svdvals
# from plots.plot_singular_values import plot_singular_values


def test_optimal_thresold_known_noise():
    """
    We test optht for a square matrix (beta = 1) and a known noise.

    Warning: See thm 1 in Gavish and Donoho 2014. We look for lambda > 2
    """
    N = 100
    noise_level = 1/np.sqrt(N)
    gaussian_random_matrix = np.random.normal(0, 0.2, (N, N))
    cut_value = 60
    X = np.block([[np.ones(N - cut_value), np.zeros(cut_value)],
                  [np.zeros(N - cut_value), np.ones(cut_value)]])
    unknown_weight_matrix = X.T@X
    W = unknown_weight_matrix + noise_level*gaussian_random_matrix

    singularValues = svdvals(W)

    # plot_singular_values(singularValues)

    perryGavishDonohoThreshold = optht(1, sv=singularValues, sigma=noise_level)
    cutoff = 4/np.sqrt(3)*np.sqrt(N)*noise_level
    greater_than_cutoff = np.where(singularValues > cutoff)
    if greater_than_cutoff[0].size > 0:
        perryGavishDonohoThreshold_true = np.max(greater_than_cutoff) + 1
    else:
        perryGavishDonohoThreshold_true = 0

    assert np.allclose(perryGavishDonohoThreshold,
                       perryGavishDonohoThreshold_true)


if __name__ == "__main__":
    pytest.main()
