#!/usr/bin/env python3
"""Parameter utilities for general modular graphs."""
# -*- coding: utf-8 -*-
# @author: Jean-Gabriel Young <jean.gabriel.young@gmail.com>
#          Vincent Thibeault  <vincent.thibeault.1@ulaval.ca>
import numpy as np


ensemble_types = ['simple_undirected', 'simple_directed',
                  'undirected', 'directed']


def get_m(A, sizes):
    """
    Get the number of edges counts between block pairs.

    :param A: simple undirected adjacency matrix from the SBM
              -> nx.stochastic_block_model(sizes, pq)
    :param sizes: list of the community sizes

    :return: m: (qxq array) number of edges counts between block pairs
    """
    q = len(sizes)    # number of blocks
    # N = len(A[:, 0])  # number of nodes
    m = np.zeros((q, q))
    mmu = 0
    for mu in range(q):
        nnu = 0
        for nu in range(q):
            if mu == nu:
                m[mu, nu] = \
                    np.sum(A[mmu:mmu + sizes[mu], nnu:nnu + sizes[nu]])/2
            else:
                m[mu, nu] = \
                    np.sum(A[mmu:mmu + sizes[mu], nnu:nnu + sizes[nu]])
            nnu += sizes[nu]
        mmu += sizes[mu]
    return m.astype(int)


def get_m_max(sizes, ensemble="simple_undirected"):
    """Get maximal edge counts between block pairs."""
    q = len(sizes)  # number of blocks
    m_max = np.zeros((q, q))
    for i in range(q):
        if ensemble == "simple_undirected":
            m_max[i, i] = sizes[i] * (sizes[i] - 1) / 2
        elif ensemble == "simple_directed":
            m_max[i, i] = sizes[i] * (sizes[i] - 1)
        elif ensemble == "undirected":
            m_max[i, i] = sizes[i] * (sizes[i] + 1) / 2
        else:  # ensemble == "directed":
            m_max[i, i] = sizes[i] ** 2
        for j in range(i + 1, q):
            m_max[i, j] = sizes[i] * sizes[j]
            m_max[j, i] = sizes[i] * sizes[j]
    return m_max.astype(int)


def get_probability_SBM(A, sizes, pq):
    """

    :param A: simple undirected adjacency matrix from the SBM
              -> nx.stochastic_block_model(sizes, pq)
    :param sizes: list of the community sizes
    :param pq: affinity matrix from which the adjacency matrix A was generated

    :return: The probability of getting the graph of adjacency matrix A
    """
    m = get_m(A, sizes)
    m_max = get_m_max(sizes)
    pq = np.array(pq)
    P = 1
    n = len(pq[0])
    for mu in range(n):
        for nu in range(n):
            if mu <= nu:
                P *= ((pq[mu, nu]**m[mu, nu]) *
                      (1 - pq[mu, nu])**(m_max[mu, nu] - m[mu, nu]))
    return P


# def get_beta(w, n, ensemble='simple_undirected'):
#     """Get the value of beta for the indicator matrix W and block sizes n."""
#     m_max = get_m_max(n, ensemble)
#     if ensemble == "simple_undirected" or ensemble == "undirected":
#         normalization = np.sum(np.triu(m_max))
#         return np.sum(np.triu(m_max * w / normalization))
#     elif ensemble == "simple_directed" or ensemble == "directed":
#         normalization = np.sum(m_max)
#         return np.sum(m_max * w / normalization)
#

def get_density(pq, sizes, ensemble='simple_undirected'):
    """

    :param pq: Affinity matrix (probability of connections within and between
               the blocks)
    :param sizes: size of each blocks
    :param ensemble: "simple_undirected", "undirected", "simple_directed",
                      or "directed"
    :return: The density of the stochastic block model

    See Eqs.(6)-(7) in J.-G Young et al. "Finite-size analysis of the
    detectability limit of the stochastic block model", PRE, 2017.
    """
    m_max = get_m_max(sizes, ensemble)
    if ensemble == "simple_undirected" or ensemble == "undirected":
        normalization = np.sum(np.triu(m_max))
        return np.sum(np.triu(m_max*pq/normalization))
    elif ensemble == "simple_directed" or ensemble == "directed":
        normalization = np.sum(m_max)
        return np.sum(m_max*pq/normalization)


def expected_adjacency_matrix(pq, sizes, self_loops=True):
    n = len(sizes)
    for mu in range(n):
        row_blocks = []
        for nu in range(n):
            row_blocks.append(pq[mu][nu]*np.ones((sizes[mu], sizes[nu])))
        if not mu:
            expected_adjacency_mat = np.block(row_blocks)
        else:
            row_blocks = np.concatenate(row_blocks, axis=1)
            expected_adjacency_mat = np.concatenate([expected_adjacency_mat,
                                                     row_blocks], axis=0)
    if not self_loops:
        np.fill_diagonal(expected_adjacency_mat, 0)
    return expected_adjacency_mat


def get_p(w, p_out, p_in):
    """Construct probability matrix from indicator matrix and densities."""
    return w*(p_in - p_out) + np.ones_like(w)*p_out


def to_probability_space(rho, delta, beta):
    """Map parameters from the density space to the probability space.
    Return
    ------
    (p_out, p_in) : tuple of float
       Internal and external densities.
    """
    return rho - delta * beta, rho + delta * (1 - beta)


def to_density_space(p_out, p_in, beta):
    """Map parameters from the probability space to the density space.
    Return
    ------
    (rho, delta) : tuple of float
       Density space coordinates.
    """
    return (beta * p_in + (1 - beta) * p_out), p_in - p_out


def in_allowed_region(rho, delta, beta):
    """Check whether a (rho,Delta) coordinate is in the allowed region."""
    if delta < 0:  # map to upper region under the rotation symmetry
        rho = 1 - rho
    if rho <= beta:
        return rho / beta >= abs(delta)
    else:
        return -rho / (1 - beta) + 1 / (1 - beta) >= abs(delta)


def get_delta_limits(beta, rho):
    """Get extremal values of Delta for fixed beta and rho."""
    lims = [0, 0]  # list since pairs are immutable
    if rho < (1 - beta):
        lims[0] = -rho / (1 - beta)
    else:
        lims[0] = rho / beta - 1 / beta
    if rho < beta:
        lims[1] = rho / beta
    else:
        lims[1] = -rho / (1 - beta) + 1 / (1 - beta)
    return lims[0], lims[1]


def get_rho_limits(beta, delta):
    """Get extremal values of rho for fixed beta and delta."""
    if delta < 0:
        # return (-rho / (1 - beta), rho / beta - 1 / beta)
        return -(1 - beta) * delta, beta * delta + 1
    else:  # delta >0
        return beta * delta, 1 - (1 - beta) * delta


def get_delta_line_limits(rho_array, beta):
    # Superior limits
    return rho_array/beta, (1 - rho_array)/(1 - beta)


def give_pq(rho, delta, beta):
    p_out = to_probability_space(rho, delta, beta)[0]
    p_in = to_probability_space(rho, delta, beta)[1]
    return [[p_in, p_out], [p_out, p_in]]


def uniform_cover_generator(beta, rho_spacing=0.05, delta_spacing=0.05):
    """
    Generate a list of parameters covering the allowed region uniformly.
    ensemble='simple_undirected'
    Notes
    -----
    Return values in the density space.
    """
    for rho in np.arange(0, 1 + rho_spacing, rho_spacing):
        for Delta in np.arange(-1, 1 + delta_spacing, delta_spacing):
            if in_allowed_region(rho, Delta, beta):
                yield (rho, Delta)


def phase_transition_generator(delta_list, rho, beta):
    """
    Generate (p_out, p_in) pairs for a list of delta values at fixed rho.
    The GMG will undergo a detectability phase transition as delta nears 0.
    """
    for delta in delta_list:
        yield to_probability_space(rho, delta, beta)
