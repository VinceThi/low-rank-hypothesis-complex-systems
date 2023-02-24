# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.generate_soft_configuration_model import *
import pytest


def test_soft_configuration_model():
    N = 3
    g = 1
    alpha = np.array([0.8, 0.2, 0.3])
    beta = np.array([0.1, 0.6, 0.4])
    valid_expected_weight_matrix = \
        np.outer(alpha, beta)/(1 + np.outer(alpha, beta))

    EW = np.zeros((N, N))
    nb_instances = 100000
    for i in range(nb_instances):
        w = soft_configuration_model(N, alpha, beta, g, expected=False)
        EW = EW + w/nb_instances

    print(valid_expected_weight_matrix, "\n", EW)

    assert np.all((EW - valid_expected_weight_matrix) < 1e-2)


def test_weighted_soft_configuration_model():
    N = 3
    y = np.array([0.8, 0.2, 0.3])
    z = np.array([0.1, 0.6, 0.4])
    pij = np.outer(y, z)
    valid_expected_weight_matrix = pij/(1 - pij)

    EW = np.zeros((N, N))
    nb_instances = 100000
    for i in range(nb_instances):
        w = weighted_soft_configuration_model(y, z)
        EW = EW + w/nb_instances

    print(valid_expected_weight_matrix, "\n", EW)

    assert np.all((EW - valid_expected_weight_matrix) < 1e-2)


if __name__ == "__main__":
    pytest.main()
