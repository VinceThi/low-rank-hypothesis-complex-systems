# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from graphs.sbm_properties import expected_adjacency_matrix,\
    normalize_degree_propensity


def degree_corrected_stochastic_block_model(block_sizes, expected_nb_edges,
                                            nkappa_in, nkappa_out,
                                            selfloops=True, expected=False):
    """
    Generator of the (canonical, directed, Poisson) degree-corrected stochastic
     block model with N vertices and q blocks (groups, communities).
    :param block_sizes: q-dimensional array of the block sizes summing to N
    :param expected_nb_edges: (q, q) array where an element (r, s) is the
                              expected number of edges from group s to r.
    :param nkappa_in: N-dimensional array of propensity of vertex $i$ to
                      receive an edge respectively (proportional to the
                       expected in-degrees). If not normalized has in Eq.(15)
                       of Karrer, Newman, PRE (2011), the normalization is
                       imposed. See graphs/sbm_properties.py,
                        function normalize_degree_propensity
    :param nkappa_out: N-dimensional array of propensity of vertex $i$ to
                       give or receive an edge respectively (proportional to
                       the expected out-degrees). See graphs/sbm_properties.py,
                        function normalize_degree_propensity
    :param selfloops: (bool) if the graph has self-loops or not
    :param expected: (bool) if the expected adjacency matrix is returned or not

    :return:
    An instance [(N, N)-array] of the degree-corrected stochastic block model
    and the expected matrix if expected is True
    """

    # Builds the expected weight matrix
    Lambda = expected_adjacency_matrix(expected_nb_edges, block_sizes,
                                       self_loops=selfloops)

    if np.sum(nkappa_in) != len(block_sizes):
        nkappa_in = normalize_degree_propensity(nkappa_in, block_sizes)

    if np.sum(nkappa_out) != len(block_sizes):
        nkappa_out = normalize_degree_propensity(nkappa_out, block_sizes)

    pij = np.multiply(Lambda, np.outer(nkappa_in, nkappa_out))

    if not selfloops:
        np.fill_diagonal(pij, 0)

    # Assigns weights
    w = np.random.default_rng().poisson(pij)

    if expected:
        return w, pij
    else:
        return w
