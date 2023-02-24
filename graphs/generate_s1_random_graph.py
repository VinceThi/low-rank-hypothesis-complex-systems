# -*- coding: utf-8 -*-
# @author: Antoine Allard and Vincent Thibeault

import numpy as np
from scipy.stats import uniform


def s1_model(beta, kappa_in, kappa_out, theta, directed=True, selfloops=True,
             expected=False):
    """
    Generator of the S1 random geometric model. The expected degrees arrays
     kappa_in and kappa_out must have the same average
    (see the function generate_nonnegative_arrays_with_same_average
    for one way to respect the constraint).
    :param beta:  (float) Inverse temperature. It controls the clustering.
                 (1, infinity) -> lower to higher clustering
    :param kappa_in: Array of expected in-degrees with average <kappa>
    :param kappa_out: Array of expected out-degrees with average <kappa>
    :param theta: Array of angular positions
    :param directed: (bool) if the graph is directed or not
    :param selfloops: (bool) if the graph has self-loops or not
    :param expected: (bool) if the expected adjacency matrix is returned or not

    :return:
    An instance of the S1 random graph model without self-loops and
    the expected matrix if expected is True
    """

    if not np.abs(np.average(kappa_in) - np.average(kappa_out)) < 1e-7:
        raise ValueError("The in- and out- expected degrees must have the "
                         "same average.")

    # Builds the expected adjacency matrix (probabilities of connection)
    mu = beta*np.sin(np.pi/beta)/(2*np.pi*np.average(kappa_in))
    pij = np.absolute(theta.reshape(-1, 1) - theta)
    pij = np.pi - np.absolute(np.pi - pij)  # option 1
    # pij = np.minimum(pij, 2*np.pi - pij) # option 2
    pij = 1/(1 + (len(kappa_in)*pij /
                  (2*np.pi*mu*np.outer(kappa_in, kappa_out)))**beta)

    # Assigns which links exist.
    W = pij > uniform.rvs(size=pij.shape)

    if not selfloops:
        np.fill_diagonal(W, 0)         # Remove self-loops

    if not directed:
        W = np.tril(W) + np.tril(W).T  # Get an undirected graph

    if expected:
        return W, pij
    else:
        return W


def generate_nonnegative_arrays_with_same_average(a, b):
    """Generate nonnegative arrays of same average given nonnegative arrays"""
    mean_a, mean_b = np.mean(a), np.mean(b)
    if mean_a > mean_b:
        b = b - mean_b + mean_a
    elif mean_a < mean_b:
        a = a - mean_a + mean_b
    return a, b
