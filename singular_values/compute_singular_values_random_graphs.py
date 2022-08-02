# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.generate_random_graphs import random_graph_generators
import networkx as nx
from scipy.linalg import svdvals
import matplotlib.pyplot as plt
from plots.plot_singular_values import \
    plot_singular_values_histogram_random_networks, plot_singular_values


plot_histogram = True
plot_adjacency_matrix = False

N = 1000
nb_networks = 1000
nb_bins = 1000
graph_str = "s1"
G, args = random_graph_generators(graph_str, N)

if plot_adjacency_matrix:
    A = nx.to_numpy_array(G(*args))
    plt.matshow(A)
    plt.show()

if plot_histogram:
    plot_singular_values_histogram_random_networks(random_graph_generator=G,
                                                   random_graph_args=args,
                                                   nb_networks=nb_networks,
                                                   nb_bins=nb_bins)
else:
    A = nx.to_numpy_array(G(*args))
    singularValues = svdvals(A)
    plot_singular_values(singularValues)
