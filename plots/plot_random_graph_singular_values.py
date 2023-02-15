# -*- coding: utf-8 -*-\\
# @author: Vincent Thibeault

from graphs.generate_random_graphs import random_graph_generators
from scipy.linalg import svdvals
import networkx as nx
import numpy as np
from plots.plot_singular_values import plot_singular_values,\
    plot_singular_values_histogram_random_networks, \
    plot_singular_values_scree_random_networks
from plots.plot_weight_matrix import plot_weight_matrix


plot_histogram = False
plot_scree = True
plot_W = False
N = 1000
nb_networks = 10
graph_str = "SBM"
# Choices of graph_str
#  "gnp, SBM, watts_strogatz, barabasi_albert,"
# "configuration, directed_configuration,", "soft_configuration_model",
# "chung_lu, s1", "random_regular", "GOE", "GUE", "COE",
# "CUE", "perturbed_gaussian"

G, args = random_graph_generators(graph_str, N)


if plot_histogram:
    """ 
    Generate the data for the script "plot_fig_SI_random_graph_singular_values"
    """
    nb_bins = 1000
    plot_singular_values_histogram_random_networks(random_graph_generator=G,
                                                   random_graph_args=args,
                                                   nb_networks=nb_networks,
                                                   nb_bins=nb_bins)

elif plot_scree:
    plot_singular_values_scree_random_networks(random_graph_generator=G,
                                               random_graph_args=args,
                                               nb_vertices=N,
                                               nb_networks=nb_networks)

else:
    if graph_str in ["GOE", "GUE", "COE", "CUE", "perturbed_gaussian"]:
        W = G(*args)
    else:
        W = nx.to_numpy_array(G(*args))
    if plot_W:
        plot_weight_matrix(W)
    singularValues = svdvals(W)
    plot_singular_values(singularValues)
