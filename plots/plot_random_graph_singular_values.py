# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.random_graph_generators import random_graph_generators
from plots.plot_singular_values import \
    plot_singular_values_histogram_random_networks


N = 500
nb_networks = 100
nb_bins = 100
graph_str = "s1"
G, args = random_graph_generators(graph_str, N)
# import matplotlib.pyplot as plt
# A = nx.to_numpy_array(G)
# plt.matshow(A)
# plt.show()

plot_singular_values_histogram_random_networks(random_graph_generator=G,
                                               random_graph_args=args,
                                               nb_networks=nb_networks,
                                               nb_bins=nb_bins)
