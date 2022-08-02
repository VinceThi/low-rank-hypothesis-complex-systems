# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.generate_random_graphs import random_graph_generator


plot_histogram = True

with open(path_str + f"{dynamics_str}_data/vector_field_errors/" +
              path_error) as json_data:
        error_array = json.load(json_data)


N = 1000
nb_networks = 1000  
nb_bins = 1000
graph_str = "s1"
G, args = random_graph_generators(graph_str, N)
