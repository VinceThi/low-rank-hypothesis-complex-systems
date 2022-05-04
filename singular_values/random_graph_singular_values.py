# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import networkx as nx
import numpy as np
from graphs.s1_random_graph import s1_model
from plots.plot_singular_values import plot_singular_values
import scipy.linalg as la

N = 1000

""" G(N,p)  """
# p = 0.005
# G = nx.gnp_random_graph(N, p)

""" Watts-Strogatz """
# k = 1
# p = 0
# G = nx.watts_strogatz_graph(N, k, p)

""" Barabasi-Albert """
# k = 1
# G = nx.barabasi_albert_graph(N, k)

""" Hard configuration model """
# deg_sequence = 2*np.random.randint(0, 20, N)
# G = nx.configuration_model(deg_sequence)

""" Hard directed configuration model """
# indeg_sequence =
#    2*np.random.multinomial(500, np.random.dirichlet(np.ones(N)*0.1))
# outdeg_sequence =
#    2*np.random.multinomial(500, np.random.dirichlet(np.ones(N)*0.1))
# print(indeg_sequence, outdeg_sequence)
# sum_indeg = np.sum(indeg_sequence)
# outdeg_sequence = 2*np.random.randint(1, 5, N)
# sum_outdeg = np.sum(outdeg_sequence)
# if sum_indeg > sum_outdeg:
#     np.array([0]*(N - sum_indeg))
#     outdeg_sequence = outdeg_sequence
# G = nx.directed_configuration_model(indeg_sequence, outdeg_sequence)


""" Chung-Lu model """
w = np.sqrt(N/(N+1))*(np.ones((N, 1)) - np.random.normal(0.1, 0.01, (N, 1)))
G = nx.expected_degree_graph(w)

""" S1 model """
# beta = 2.5  # Controls the clustering, (1, inf) -> lower to higher clustering
# kappa_min = 1
# kappa_max = 8
# gamma = 2.5
# G = s1_model(N, beta, kappa_min, kappa_max, gamma)

A = nx.to_numpy_array(G)
import matplotlib.pyplot as plt
plt.matshow(A)
plt.show()
singularValues = la.svdvals(A)

plot_singular_values(singularValues)
