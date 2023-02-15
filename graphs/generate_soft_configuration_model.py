# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import networkx as nx
import numpy as np
from scipy.stats import uniform


def soft_configuration_model(N, alpha, beta, selfloops=False, directed=True,
                             expected=False):
    """
    The density from which the expected degrees are drawn is a truncated pareto
    :param N: Number of vertices
    :param alpha:
    :param beta:
    :param selfloops: (bool) if the graph has selfloops
    :param directed: (bool) if the graph is directed or not
    :param expected: (bool) if the expected adjacency matrix is returned or not

    :return:
    An instance of the soft configuration model and the expected matrix if
    expected is True
    """

    # Builds the expected adjacency matrix (probabilities of connection)
    pij = np.outer(alpha, beta)/(np.ones((N, N)) + np.outer(alpha, beta))

    # Assigns which links exist.
    p = pij > uniform.rvs(size=pij.shape)
    if not selfloops:
        np.fill_diagonal(pij, 0)
    if not directed:
        p = np.tril(p) + np.tril(p).T  # Get an undirected graph
    if expected:
        return nx.from_numpy_array(p), pij
    else:
        return nx.from_numpy_array(p)
