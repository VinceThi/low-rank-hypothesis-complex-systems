# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import networkx as nx
import numpy as np
from scipy.stats import uniform


def soft_configuration_model(alpha, beta, g, expected=False):
    """
    Generator of the directed soft configuration model with self loops.

    :param N: Number of vertices
    :param alpha: positive number related to the expected in degrees
                 through the Lagrange multipliers
    :param beta: positive number related to the expected out degrees
                 through the Lagrange multipliers
    :param g: global parameter controlling the probabilities
    :param expected: (bool) if the expected adjacency matrix is returned or not

    :return:
    An instance w of the soft configuration model and the expected matrix if
    expected is True
    """

    # Builds the expected adjacency matrix (probabilities of connection)
    pij = g*np.outer(alpha, beta)/(1 + np.outer(alpha, beta))

    if np.any(pij > 1):
        raise ValueError("A probability of connection pij is greater than 1.")

    # Assigns which links exist.
    w = pij > uniform.rvs(size=pij.shape)

    if expected:
        return w, pij
    else:
        return w


def weighted_soft_configuration_model(y, z, expected=False):
    """
    Generator of the weighted directed soft configuration model with self loops

    It is the model 3 of Garlaschelli and Loffredo, PRL, 2009 when w*-> \infty

    :param y: N-dimensional array related to the expected in strengths ,
              pij = y_i z_j must be between 0 and 1
    :param z: N-dimensional array related to the expected out strengths,
              pij = y_i z_j must be between 0 and 1
    :param expected: (bool) if the expected adjacency matrix is returned or not

    :return:
    An instance of the weighted soft configuration model
    and the expected matrix if expected is True
    """

    # Builds the expected adjacency matrix (probabilities of connection)
    pij = np.outer(y, z)
    if np.any(pij > 1):
        raise ValueError("A probability of connection pij is greater than 1.")

    # Assigns weights
    w = np.random.default_rng().geometric(1 - pij) - 1

    if expected:
        return w, pij/(1 - pij)
    else:
        return w