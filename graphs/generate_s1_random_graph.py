# -*- coding: utf-8 -*-
# @author: Antoine Allard and Vincent Thibeault

import networkx as nx
import numpy as np
from scipy.stats import pareto, uniform


def s1_model(N, beta, kappa_min, kappa_max, gamma):
    """
    The density from which the expected degrees are drawn is a truncated pareto
    :param N:
    :param beta:  Parameter controling the clustering.
                 (1, infinity) -> lower to higher clustering
    :param kappa_min: Minimum expected degree
    :param kappa_max: Maximum expected degree
    :param gamma: shape parameter (gamma + 1) of the Pareto distribution
    :return:
    An instance of the S1 random graph model
    """
    kappas = [val for val in kappa_min * pareto.rvs(gamma - 1, size=N)
              if val < kappa_max]
    while len(kappas) < N:
        kappas.extend([
            val for val in kappa_min*pareto.rvs(gamma - 1, size=N-len(kappas))
            if val < kappa_max])

    # Angular positions drawn uniformly
    thetas = uniform.rvs(size=N)

    # Builds the "average" adjacency matrix (probabilities of connection)
    mu = beta*np.sin(np.pi/beta)/(2*np.pi*np.average(kappas))
    pij = np.absolute(thetas.reshape(-1, 1) - thetas)
    pij = np.pi - np.absolute(np.pi - pij)  # option 1
    # pij = np.minimum(pij, 2 * np.pi - pij) # option 2
    pij = 1/(1 + (len(kappas)*pij /
                  (2*np.pi*mu*np.outer(kappas, kappas)))**beta)
    np.fill_diagonal(pij, 0)

    # Assigns which links exist.
    p = pij > uniform.rvs(size=pij.shape)
    np.fill_diagonal(p, False)

    # Generate network from the lower diagonal.
    edgelist = np.nonzero(np.tril(p))
    edgelist = [t for t in zip(edgelist[0], edgelist[1])]

    return nx.Graph(edgelist)
