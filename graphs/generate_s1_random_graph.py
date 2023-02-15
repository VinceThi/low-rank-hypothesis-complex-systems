# -*- coding: utf-8 -*-
# @author: Antoine Allard and Vincent Thibeault

import networkx as nx
import numpy as np
from scipy.stats import pareto, uniform


def s1_model(N, beta, kappa_min, kappa_max, gamma, directed=False,
             expected=False):
    """
    The density from which the expected degrees are drawn is a truncated pareto
    :param N: Number of vertices
    :param beta:  Parameter controling the clustering.
                 (1, infinity) -> lower to higher clustering
    :param kappa_min: Minimum expected degree
    :param kappa_max: Maximum expected degree
    :param gamma: shape parameter (gamma + 1) of the Pareto distribution
                  gamma = 2.5 => shape parameter = 1.5
    :param directed: (bool) if the graph is directed or not
    :param expected: (bool) if the expected adjacency matrix is returned or not

    :return:
    An instance of the S1 random graph model and the expected matrix if
    expected is True
    """
    kappas = [val for val in kappa_min * pareto.rvs(gamma - 1, size=N)
              if val < kappa_max]
    while len(kappas) < N:
        kappas.extend([
            val for val in kappa_min*pareto.rvs(gamma - 1, size=N-len(kappas))
            if val < kappa_max])

    # Angular positions drawn uniformly
    thetas = 2*np.pi*uniform.rvs(size=N)

    # Builds the expected adjacency matrix (probabilities of connection)
    mu = beta*np.sin(np.pi/beta)/(2*np.pi*np.average(kappas))
    pij = np.absolute(thetas.reshape(-1, 1) - thetas)
    pij = np.pi - np.absolute(np.pi - pij)  # option 1
    # pij = np.minimum(pij, 2*np.pi - pij) # option 2
    pij = 1/(1 + (len(kappas)*pij/(2*np.pi*mu*np.outer(kappas, kappas)))**beta)
    np.fill_diagonal(pij, 0)

    # Assigns which links exist.
    p = pij > uniform.rvs(size=pij.shape)
    np.fill_diagonal(p, False)         # Remove self-loops
    if not directed:
        p = np.tril(p) + np.tril(p).T  # Get an undirected graph
    if expected:
        return nx.from_numpy_array(p), pij
    else:
        return nx.from_numpy_array(p)
