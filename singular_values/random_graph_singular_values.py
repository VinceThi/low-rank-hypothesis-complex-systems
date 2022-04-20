# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import networkx as nx
from graphs.s1_random_graph import s1_model
from plots.plot_singular_values import plot_singular_values
import scipy.linalg as la

N = 1000

# k = 1
# p = 0
# G = nx.watts_strogatz_graph(N, k, p)

# k = 1
# G = nx.barabasi_albert_graph(N, k)

# deg_sequence = 2*np.random.randint(0, 20, N)
# G = nx.configuration_model(deg_sequence)

beta = 2.5  # Controls the clustering, (1, inf) -> lower to higher clustering
kappa_min = 1
kappa_max = 8
gamma = 2.5
G = s1_model(N, beta, kappa_min, kappa_max, gamma)

A = nx.to_numpy_array(G)
singularValues = la.svdvals(A)

plot_singular_values(singularValues)
