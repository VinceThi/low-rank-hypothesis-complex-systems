# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from singular_values.compute_effective_ranks import *
from singular_values.optimal_shrinkage import *
import numpy as np
from plots.plot_singular_values import plot_singular_values


def test_compare_optimal_shrinkage_and_threshold_random_matrix():
    N = 100
    rank_r = 20
    prng = np.random.RandomState(1234567890)
    L = prng.uniform(0, 0.5, (N, rank_r))
    M = prng.normal(0, 0.5, (rank_r, N))
    # D = np.diag(np.exp(np.linspace(0, 1, rank_r)) + )
    # utiliser ortho np.random
    sigma = 2.5
    noise_level = sigma / np.sqrt(N)
    W = L@M + noise_level * prng.normal(0, 0.2, (N, N))
    U, S, Vh = np.linalg.svd(W)

    threshold_shrink_fro = computeOptimalShrinkage(S, 'frobenius')
    threshold_shrink_op = computeOptimalShrinkage(S, 'operator')
    threshold_shrink_nuc = computeOptimalShrinkage(S, 'nuclear')

    hard_threshold_fro = np.round(optimal_threshold(S, 1), 2)

    # print(f"\n\nhard_thresold_fro = {hard_threshold_fro} \n\n",
    #       f"threshold_shrink_fro = {threshold_shrink_fro} \n\n",
    #       f"threshold_shrink_op =  {threshold_shrink_op}\n\n",
    #       f"threshold_shrink_nuc = {threshold_shrink_nuc} \n\n")
    assert np.allclose(20, hard_threshold_fro) \
        and np.allclose(20, threshold_shrink_fro) \
        and np.allclose(20, threshold_shrink_op) \
        and np.allclose(20, threshold_shrink_nuc)


def test_compare_optimal_shrinkage_and_threshold_drosophila():
    """ This test trivially pass. """
    path = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "singular_values/properties/"
    networkName = "drosophila"
    singularValuesFilename = path + networkName + '_singular_values.txt'
    S = np.loadtxt(singularValuesFilename)
    threshold_shrink_fro = computeOptimalShrinkage(S, 'frobenius')
    threshold_shrink_op = computeOptimalShrinkage(S, 'operator')
    threshold_shrink_nuc = computeOptimalShrinkage(S, 'nuclear')

    hard_threshold_fro = np.round(optimal_threshold(S, 1), 2)

    print(f"\n\nhard_thresold_fro = {hard_threshold_fro} \n\n",
          f"thresold_shrink_fro = {threshold_shrink_fro} \n\n",
          f"thresold_shrink_op =  {threshold_shrink_op}\n\n",
          f"thresold_shrink_nuc = {threshold_shrink_nuc} \n\n")

    plot_singular_values(S)
    assert 1


if __name__ == "__main__":
    pytest.main()
