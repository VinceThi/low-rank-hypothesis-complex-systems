# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import networkx as nx
import numpy as np
import tenpy as tp
from scipy.stats import uniform
from graphs.generate_truncated_pareto import truncated_pareto
from graphs.generate_random_matrices import perturbed_gaussian,\
    tenpy_random_matrix
from graphs.generate_s1_random_graph import s1_model, \
    generate_nonnegative_arrays_with_same_average
from graphs.generate_soft_configuration_model import soft_configuration_model,\
    weighted_soft_configuration_model
from graphs.generate_degree_corrected_stochastic_block_model import\
    degree_corrected_stochastic_block_model
from graphs.sbm_properties import normalize_degree_propensity


def random_graph_generators(graph_str, N):

    """

    :param graph_str: name of the generator. Options:
        "gnp", "SBM", "watts_strogatz", "barabasi_albert",
        "configuration", "directed_configuration", "soft_configuration_model",
        "weighted_soft_configuration_model", "chung_lu", "s1",
        "random_regular", "GOE", "GUE", "COE", "CUE"
        "perturbed_gaussian
    :param N: number of vertices, must be a multiple of 2 and 5
    :return: generator and its (fixed) arguments

    Note that for the s1_model, soft_configuration_model, DCSBM
     weighted_soft_configuration_model, tenpy random matrices,
     "perturbed_gaussian"
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
        generator = degree_corrected_stochastic_block_model
        edge_propensities = N*np.array([[0.42, 0.09, 0.30, 0.02, 0.003],
                                        [0.05, 0.60, 0.20, 0.10, 0.05],
                                        [0.20, 0.02, 0.70, 0.20, 0.02],
                                        [0.15, 0.05, 0.05, 0.80, 0.01],
                                        [0.01, 0.10, 0.05, 0.20, 0.50]])
        sizes = [N//10, 2*N//5, N//10, N//5, N//5]

        kappa_in_min = 5
        kappa_in_max = 100
        gamma_in = 2.5
        kappa_in = truncated_pareto(N, kappa_in_min, kappa_in_max, gamma_in)
        nkappa_in = normalize_degree_propensity(kappa_in, sizes)
        kappa_out_min = 3
        kappa_out_max = 50
        gamma_out = 2
        kappa_out = truncated_pareto(N, kappa_out_min,
                                     kappa_out_max, gamma_out)
        nkappa_out = normalize_degree_propensity(kappa_out, sizes)

        args = (sizes, edge_propensities, nkappa_in, nkappa_out)

    elif graph_str == "chung_lu":
        generator = nx.expected_degree_graph
        kappa_min = 3
        kappa_max = 20
        gamma = 2.5
        kappas = truncated_pareto(N, kappa_min, kappa_max, gamma)
        args = (kappas,)

    elif graph_str == "soft_configuration_model":
        generator = soft_configuration_model
        alpha_min = 10
        beta_min = 5
        alpha_max = 50
        beta_max = 30
        gamma = 2.5
        alpha = truncated_pareto(N, alpha_min, alpha_max, gamma) / np.sqrt(N)
        beta = truncated_pareto(N, beta_min, beta_max, gamma) / np.sqrt(N)

        g = 1
        args = (N, alpha, beta, g)

    elif graph_str == "weighted_soft_configuration_model":
        generator = weighted_soft_configuration_model
        y = truncated_pareto(N, 0, 1, 2.5)
        z = truncated_pareto(N, 0, 1, 3)
        args = (y, z)

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
        kappa_in_min = 5
        kappa_in_max = 100
        gamma_in = 2.5
        kappa_in = truncated_pareto(N, kappa_in_min, kappa_in_max, gamma_in)
        kappa_out_min = 3
        kappa_out_max = 50
        gamma_out = 2
        kappa_out = truncated_pareto(N, kappa_out_min,
                                     kappa_out_max, gamma_out)
        kappa_in, kappa_out = \
            generate_nonnegative_arrays_with_same_average(kappa_in, kappa_out)

        theta = 2*np.pi*uniform.rvs(size=N)

        args = (beta, kappa_in, kappa_out, theta)

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
        beta = 2.5
        # ^ Controls the clustering, (1, inf) -> lower to higher clustering
        kappa_in_min = np.random.randint(3, 6, N)
        kappa_in_max = np.random.randint(20, 200, N)
        gamma_in = np.random.randint(1.1, 5, N)
        kappa_in = truncated_pareto(N, kappa_in_min, kappa_in_max, gamma_in)
        kappa_out_min = np.random.randint(3, 6, N)
        kappa_out_max = np.random.randint(20, 200, N)
        gamma_out = np.random.randint(1.1, 5, N)
        kappa_out = truncated_pareto(N, kappa_out_min,
                                     kappa_out_max, gamma_out)
        kappa_in, kappa_out = \
            generate_nonnegative_arrays_with_same_average(kappa_in, kappa_out)

        theta = 2*np.pi*uniform.rvs(size=N)

        args = (beta, kappa_in, kappa_out, theta)

    else:
        raise ValueError("No graph_str as this name, choose between "
                         "disconnected_self_loops, gnp, SBM, watts_strogatz,"
                         " barabasi_albert, hard_configuration,"
                         " directed_hard_configuration, chung_lu, s1",
                         "random_regular")

    return generator, args
