# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.sbm_properties import *
from graphs.generate_truncated_pareto import truncated_pareto
import pytest


def test_expected_adjacency_matrix_self_loops():
    pq = np.array([[0.42, 0.09, 0.30],
                   [0.05, 0.60, 0.20],
                   [0.20, 0.02, 0.70]])
    sizes = [3, 2, 2]
    valid_expected_adjacency_matrix = \
        np.array([[0.42, 0.42, 0.42, 0.09, 0.09, 0.30, 0.30],
                  [0.42, 0.42, 0.42, 0.09, 0.09, 0.30, 0.30],
                  [0.42, 0.42, 0.42, 0.09, 0.09, 0.30, 0.30],
                  [0.05, 0.05, 0.05, 0.60, 0.60, 0.20, 0.20],
                  [0.05, 0.05, 0.05, 0.60, 0.60, 0.20, 0.20],
                  [0.20, 0.20, 0.20, 0.02, 0.02, 0.70, 0.70],
                  [0.20, 0.20, 0.20, 0.02, 0.02, 0.70, 0.70]])
    to_test = expected_adjacency_matrix(pq, sizes)
    assert np.all(to_test == valid_expected_adjacency_matrix)


def test_expected_adjacency_matrix_no_self_loops():
    pq = np.array([[0.42, 0.09, 0.30],
                   [0.05, 0.60, 0.20],
                   [0.20, 0.02, 0.70]])
    sizes = [3, 2, 2]
    valid_expected_adjacency_matrix = \
        np.array([[0, 0.42, 0.42, 0.09, 0.09, 0.30, 0.30],
                  [0.42, 0, 0.42, 0.09, 0.09, 0.30, 0.30],
                  [0.42, 0.42, 0, 0.09, 0.09, 0.30, 0.30],
                  [0.05, 0.05, 0.05, 0, 0.60, 0.20, 0.20],
                  [0.05, 0.05, 0.05, 0.60, 0, 0.20, 0.20],
                  [0.20, 0.20, 0.20, 0.02, 0.02, 0, 0.70],
                  [0.20, 0.20, 0.20, 0.02, 0.02, 0.70, 0]])
    to_test = expected_adjacency_matrix(pq, sizes, self_loops=False)
    assert np.all(to_test == valid_expected_adjacency_matrix)


def test_get_membership_array():
    sizes = np.array([3, 2])
    valid_membership_array = np.array([0, 0, 0, 1, 1])
    to_test = get_membership_array(sizes)
    assert np.all(to_test == valid_membership_array)


def test_normalize_degree_propensity():
    sizes = np.array([2, 2])
    valid_nkappa = np.array([0.8, 0.2, 0.9, 0.1])
    test_nkappa = np.array([8, 2, 0.009, 0.001])
    test_nkappa = normalize_degree_propensity(test_nkappa, sizes)
    assert np.allclose(test_nkappa, valid_nkappa)


def test_normalize_degree_propensity_more_groups():
    N = 20
    sizes = np.array([5, 3, 2, 4, 2, 4])
    kappa_in_min = 3
    kappa_in_max = 10
    gamma_in = 2.5
    kappa_in = truncated_pareto(N, kappa_in_min, kappa_in_max, gamma_in)
    nkappa_in = normalize_degree_propensity(kappa_in, sizes)
    assert np.allclose(np.sum(nkappa_in), len(sizes))


if __name__ == "__main__":
    pytest.main()
