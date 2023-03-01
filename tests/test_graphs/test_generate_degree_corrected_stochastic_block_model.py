# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.generate_degree_corrected_stochastic_block_model import *
from plots.plot_weight_matrix import plot_weight_matrix
import pytest


def test_degree_corrected_stochastic_block_model():
    plot_weight_mat = False
    block_sizes = np.array([2, 2])
    N = np.sum(block_sizes)
    expected_nb_edges = N*np.array([[0.8, 0.2],
                                    [0.4, 0.6]])
    nkappa_in = np.array([0.8, 0.2, 0.9, 0.1])
    # ^satisfies the normalization per group
    nkappa_out = np.array([0.4, 0.6, 0.25, 0.75])
    # ^ satisfies the normalization per group
    Lambda = expected_adjacency_matrix(expected_nb_edges, block_sizes,
                                       self_loops=True)
    valid_expected_weight_matrix = Lambda*np.outer(nkappa_in, nkappa_out)

    EW = np.zeros((N, N))
    nb_instances = 10000
    for i in range(nb_instances):
        W = degree_corrected_stochastic_block_model(block_sizes,
                                                    expected_nb_edges,
                                                    nkappa_in, nkappa_out,
                                                    selfloops=True,
                                                    expected=False)
        if plot_weight_mat:
            plot_weight_matrix(W)
        EW = EW + W/nb_instances

    # print("\n\n", valid_expected_weight_matrix, "\n\n", EW, "\n\n",
    #       (EW - valid_expected_weight_matrix)**2)
    assert np.all((EW - valid_expected_weight_matrix)**2 < 1e-2)


def test_degree_corrected_stochastic_block_model_propensities():
    block_sizes = np.array([2, 2])
    expected_nb_edges = np.array([[5, 1],
                                  [2, 3]])
    nkappa_in = np.array([0.8, 0.2, 0.9, 0.1])
    # ^ satisfies the normalization per group
    nkappa_out = np.array([0.4, 0.6, 0.3, 0.7])
    # ^ satisfies the normalization per group
    Lambda = expected_adjacency_matrix(expected_nb_edges, block_sizes,
                                       self_loops=True)
    # print(Lambda)
    expected_weight_matrix = Lambda*np.outer(nkappa_in, nkappa_out)

    valid_expected_in_degrees = np.array([0.8*6, 0.2*6, 0.9*5, 0.1*5])
    valid_expected_out_degrees = np.array([0.4*7, 0.6*7, 0.3*4, 0.7*4])
    test_expected_in_degrees = np.sum(expected_weight_matrix, axis=1)
    test_expected_out_degrees = np.sum(expected_weight_matrix, axis=0)

    assert np.allclose(np.sum(expected_weight_matrix[:2, :2]), 5)\
        and np.allclose(np.sum(expected_weight_matrix[:2, 2:]), 1)\
        and np.allclose(np.sum(expected_weight_matrix[2:, :2]), 2)\
        and np.allclose(np.sum(expected_weight_matrix[2:, 2:]), 3)\
        and np.allclose(valid_expected_in_degrees, test_expected_in_degrees)\
        and np.allclose(valid_expected_out_degrees, test_expected_out_degrees)


if __name__ == "__main__":
    pytest.main()
