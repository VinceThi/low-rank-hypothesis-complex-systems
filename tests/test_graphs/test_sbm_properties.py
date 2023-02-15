# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.sbm_properties import *
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


if __name__ == "__main__":
    pytest.main()
