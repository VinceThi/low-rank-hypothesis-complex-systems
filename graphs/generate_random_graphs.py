# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import networkx as nx
import numpy as np
import tenpy as tp
from scipy.stats import pareto
from graphs.random_matrix_generators import perturbed_gaussian,\
    tenpy_random_matrix
from graphs.generate_s1_random_graph import s1_model
from graphs.generate_soft_configuration_model import soft_configuration_model


def truncated_pareto(N, kappa_min, kappa_max, gamma):
    kappas = [val for val in kappa_min * pareto.rvs(gamma - 1, size=N)
              if val < kappa_max]
    while len(kappas) < N:
        kappas.extend([val for val
                       in kappa_min*pareto.rvs(gamma-1, size=N-len(kappas))
                       if val < kappa_max])
    return np.array(kappas)


def random_graph_generators(graph_str, N):

    """

    :param graph_str: name of the generator. Options:
        "gnp", "SBM", "watts_strogatz", "barabasi_albert",
        "configuration", "directed_configuration", "soft_configuration_model",
        "chung_lu", "s1", "random_regular", "GOE", "GUE", "COE", "CUE"
        "perturbed_gaussian
    :param N: number of vertices, must be a multiple of 2 and 5
    :return: generator and its (fixed) arguments

    Note that for the tenpy random matrices and "perturbed_gaussian"
     an instance W of the random matrix is obtained as W = generator(*args).

    Else, W = nx.to_numpy_array(generator(*args))
    """

    assert N % 5 == 0
    assert N % 2 == 0

    if graph_str == "gnp":
        generator = nx.fast_gnp_random_graph
        p = 0.1
        args = (N, p)

    elif graph_str == "random_regular":
        generator = nx.random_regular_graph
        d = 3
        args = (d, N)

    elif graph_str == "SBM":
        generator = nx.stochastic_block_model
        pq = [[0.42, 0.09, 0.30, 0.02, 0.003],
              [0.05, 0.60, 0.20, 0.10, 0.05],
              [0.20, 0.02, 0.70, 0.20, 0.02],
              [0.15, 0.05, 0.05, 0.80, 0.01],
              [0.01, 0.10, 0.05, 0.20, 0.50]]

        # print(np.linalg.matrix_rank(pq)) = 5
        sizes = [N//10, 2*N//5, N//10, N//5, N//5]
        #                          directed+self-loops
        args = (sizes, pq, None, None, True, True, True)

    elif graph_str == "DCSBM":
        raise ValueError("DCSBM is not coded yet."
                         " It is not on networkx, but it is on graph tool.")

    elif graph_str == "chung_lu":
        generator = nx.expected_degree_graph
        kappa_min = 3
        kappa_max = 20
        gamma = 2.5
        kappas = truncated_pareto(N, kappa_min, kappa_max, gamma)
        args = (kappas,)

    elif graph_str == "soft_configuration_model":
        generator = soft_configuration_model
        k = 8
        alpha = (np.ones((N, 1)) - np.random.uniform(0, 1, (N, 1)))/np.sqrt(k)
        beta = (np.ones((N, 1)) - np.random.uniform(0, 1, (N, 1)))/np.sqrt(k)
        # f"\nbar kin = {np.sum(meanA, axis=1)}",
        # f"\nbar kout = {np.sum(meanA, axis=0)}")
        args = (N, alpha, beta, )

    elif graph_str == "configuration":
        generator = nx.configuration_model
        deg_sequence = 2*np.random.randint(0, 20, N)
        args = (N, deg_sequence)

    elif graph_str == "directed_configuration":
        generator = nx.directed_configuration_model
        indeg_sequence = \
            2*np.random.multinomial(500, np.random.dirichlet(np.ones(N) * 0.1))
        # outdeg_sequence = \
        #     2*np.random.multinomial(500, np.random.dirichlet(np.ones(N)*0.1))
        sum_indeg = np.sum(indeg_sequence)
        outdeg_sequence = 2 * np.random.randint(1, 5, N)
        sum_outdeg = np.sum(outdeg_sequence)
        if sum_indeg > sum_outdeg:
            np.array([0] * (N - sum_indeg))
            outdeg_sequence = outdeg_sequence
        args = (indeg_sequence, outdeg_sequence)

    elif graph_str == "watts_strogatz":
        generator = nx.watts_strogatz_graph
        k = 3
        p = 0.1
        args = (N, k, p)

    elif graph_str == "barabasi_albert":
        generator = nx.barabasi_albert_graph
        m = 10
        args = (N, m)

    elif graph_str == "s1":
        generator = s1_model
        beta = 2.5
        # ^ Controls the clustering, (1, inf) -> lower to higher clustering
        kappa_min = 3
        kappa_max = 20
        gamma = 2.5
        args = (N, beta, kappa_min, kappa_max, gamma)

    elif graph_str == "perturbed_gaussian":
        generator = perturbed_gaussian
        rank = 5
        L = np.random.normal(0, 1/np.sqrt(N), (N, rank))
        R = np.random.normal(0, 1, (rank, N))
        g = 1    # Strength of the random part
        var = g**2/N
        args = (N, L, R, var)

    elif graph_str == "GOE":
        generator = tenpy_random_matrix
        args = (N, tp.linalg.random_matrix.GOE)

    elif graph_str == "GUE":
        generator = tenpy_random_matrix
        args = (N, tp.linalg.random_matrix.GUE)

    elif graph_str == "COE":
        generator = tenpy_random_matrix
        args = (N, tp.linalg.random_matrix.COE)

    elif graph_str == "CUE":
        generator = tenpy_random_matrix
        args = (N, tp.linalg.random_matrix.CUE)
    
    else:
        raise ValueError("No graph_str as this name, choose between "
                         "gnp, SBM, watts_strogatz, barabasi_albert,"
                         " hard_configuration, directed_hard_configuration,"
                         "chung_lu, s1, random_regular, GOE, GUE, COE, CUE, "
                         "perturbed_gaussian")

    return generator, args


def generate_fix_sum_random_vec(limit, num_elem, tries=10):
    v = np.random.randint(0, limit, num_elem)
    s = sum(v)
    if np.sum(np.round(v/s*limit)) == limit:
        return np.round(v/s*limit)
    elif np.sum(np.floor(v/s*limit)) == limit:
        return np.floor(v/s*limit)
    elif np.sum(np.ceil(v/s*limit)) == limit:
        return np.ceil(v/s*limit)
    else:
        return generate_fix_sum_random_vec(limit, num_elem, tries-1)


def random_graph_ensemble_generators(graph_str, N):

    assert N % 2 == 0

    if graph_str == "disconnected_self_loops":
        generator = np.eye
        args = (N, N)

    elif graph_str == "gnp":
        generator = nx.fast_gnp_random_graph
        p = np.random.random()
        args = (N, p)

    elif graph_str == "random_regular":
        generator = nx.random_regular_graph
        d = np.random.randint(1, 10)
        args = (d, N)

    elif graph_str == "SBM":
        generator = nx.stochastic_block_model
        nb_communities = np.random.randint(2, 20)
        pq = np.random.random((nb_communities, nb_communities))
        sizes = generate_fix_sum_random_vec(N, nb_communities)
        sizes = [int(x) for x in sizes]
        # numbers_with_sum(nb_communities, N)
        #                          directed+self-loops
        args = (sizes, pq, None, None, True, True, True)

    elif graph_str == "DCSBM":
        raise ValueError("DCSBM is not coded yet."
                         " It is not on networkx, but it is on graph tool.")

    elif graph_str == "watts_strogatz":
        generator = nx.watts_strogatz_graph
        k = np.random.randint(2, 20)
        p = np.random.random()
        args = (N, k, p)

    elif graph_str == "barabasi_albert":
        generator = nx.barabasi_albert_graph
        m = np.random.randint(2, 20)
        args = (N, m)

    elif graph_str == "hard_configuration":
        generator = nx.configuration_model
        deg_sequence = 2*np.random.randint(0, 20, N)
        args = (N, deg_sequence)

    elif graph_str == "directed_hard_configuration":
        generator = nx.directed_configuration_model
        indeg_sequence = \
            2*np.random.multinomial(500, np.random.dirichlet(np.ones(N)*0.1))
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
        w = np.random.uniform(1, 5, size=N)  # powerlaw.rvs(2.5, size=N)
        args = (w,)

    elif graph_str == "s1":
        generator = s1_model
        beta = np.random.randint(1, 5)  # 2.5
        # ^ Controls the clustering, (1, inf) -> lower to higher clustering
        kappa_min = np.random.randint(1, 10)  # 3
        kappa_max = np.random.randint(10, 30)  # 20
        gamma = np.random.uniform(1, 4)  # 2.5
        args = (N, beta, kappa_min, kappa_max, gamma)

    else:
        raise ValueError("No graph_str as this name, choose between "
                         "disconnected_self_loops, gnp, SBM, watts_strogatz,"
                         " barabasi_albert, hard_configuration,"
                         " directed_hard_configuration, chung_lu, s1",
                         "random_regular")

    return generator, args
