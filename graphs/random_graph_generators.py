# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import networkx as nx
import numpy as np
# from scipy.stats import powerlaw
from graphs.s1_random_graph import s1_model


def random_graph_generators(graph_str, N):

    if graph_str == "gnp":
        generator = nx.fast_gnp_random_graph
        p = 0.1
        args = (N, p)

    elif graph_str == "SBM":
        generator = nx.stochastic_block_model
        pq = [[0.4, 0.1], [0.05, 0.2]]
        sizes = [1 - N// 4, N // 4]
        args = (N, pq,
                sizes)  # TODO verify

    elif graph_str == "DCSBM":
        generator = nx.degree_corrected_stochastic_block_model  # TODO

    elif graph_str == "watts_strogatz":
        generator = nx.watts_strogatz_graph
        k = 1
        p = 0
        args = (N, k, p)

    elif graph_str == "barabasi_albert":
        generator = nx.barabasi_albert_graph
        k = 1
        args = (N, k)

    elif graph_str == "hard_configuration":
        generator = nx.configuration_model
        deg_sequence = 2 * np.random.randint(0, 20, N)
        args = (N, deg_sequence)

    elif graph_str == "directed_hard_configuration":
        generator = nx.directed_configuration_model
        indeg_sequence = \
            2 * np.random.multinomial(500,
                                      np.random.dirichlet(np.ones(N) * 0.1))
        # outdeg_sequence = \
        #     2*np.random.multinomial(500, np.random.dirichlet(np.ones(N)*0.1))
        sum_indeg = np.sum(indeg_sequence)
        outdeg_sequence = 2 * np.random.randint(1, 5, N)
        sum_outdeg = np.sum(outdeg_sequence)
        if sum_indeg > sum_outdeg:
            np.array([0] * (N - sum_indeg))
            outdeg_sequence = outdeg_sequence
        args = (indeg_sequence, outdeg_sequence)

    elif graph_str == "chung_lu":
        generator = nx.expected_degree_graph
        w = np.random.uniform(0, 5,
                              size=N)  # powerlaw.rvs(2.5, size=N)
        args = (w,)

    elif graph_str == "s1":
        generator = s1_model
        beta = 2.5
        # ^ Controls the clustering, (1, inf) -> lower to higher clustering
        kappa_min = 1
        kappa_max = 8
        gamma = 2.5
        args = (N, beta, kappa_min, kappa_max, gamma)

    else:
        raise ValueError("No graph_str as this name, choose between "
                         "gnp, SBM, watts_strogatz, barabasi_albert,"
                         " hard_configuration, directed_hard_configuration,"
                         "chung_lu, s1")

    return generator, args
