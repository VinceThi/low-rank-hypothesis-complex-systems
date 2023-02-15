# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svdvals
from plots.config_rcparams import *
import time
import json
import tkinter.simpledialog
from tkinter import messagebox
import networkx as nx
from tqdm import tqdm
from graphs.sbm_properties import get_density, expected_adjacency_matrix

path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/singular_values/properties/" \
           "singular_values_random_graphs/"

""" Random graph parameters """
graph_str = "sbm"
N = 1000
nb_networks = 10     # 1000
directed = True
selfloops = True
pq0 = np.array([[0.10, 0.02, 0.01, 0.02, 0.003],
                [0.005, 0.10, 0.002, 0.05, 0.005],
                [0.002, 0.02, 0.05, 0.05, 0.02],
                [0.01, 0.005, 0.005, 0.10, 0.01],
                [0.01, 0.01, 0.005, 0.05, 0.05]])
sizes = [N//10, 2*N//5, N//10, N//5, N//5]

pq = 8*pq0   # pq = 8*pq0, 3*pq0

EW = expected_adjacency_matrix(pq, sizes, self_loops=selfloops)
norm_EW = norm(EW)
norm_R = np.zeros(nb_networks)

density = get_density(pq, sizes, ensemble='directed')

xlabel = "Index $i$"
ylabel = "Average rescaled singular\n values $\\sigma_i/\\sigma_1$"

singularValues = np.zeros((nb_networks, N))
for i in tqdm(range(0, nb_networks)):
    W = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq.tolist(),
                                                    directed=directed,
                                                    selfloops=selfloops))
    norm_R[i] = norm(W - EW)
    singularValues_instance = svdvals(W)
    singularValues[i, :] = singularValues_instance

norm_ratio = np.mean(norm_R)/norm_EW
print(norm_ratio)

mean_singularValues = np.mean(singularValues, axis=0)
std_singularValues = np.std(singularValues, axis=0)

fig = plt.figure(figsize=(6, 4))
plt.scatter(np.arange(1, len(singularValues_instance) + 1, 1),
            mean_singularValues, s=3, color=deep[0])
plt.fill_between(np.arange(1, len(singularValues_instance) + 1, 1),
                 mean_singularValues - std_singularValues,
                 mean_singularValues + std_singularValues,
                 color=deep[0], alpha=0.2)
plt.tick_params(axis='both', which='major')
plt.xlabel(xlabel)
plt.ylabel(ylabel, labelpad=20)
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters,"
                       " the data, and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    path = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "singular_values/properties/singular_values_random_graphs/"
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    parameters_dictionary = {"graph_str": graph_str, "sizes": sizes,
                             "N": N, "density": density, "pq": pq.tolist(),
                             "directed": directed, "selfloops": selfloops,
                             "nb_samples (nb_networks)": nb_networks,
                             "norm_R": norm_R.tolist()}

    fig.savefig(path + f'{timestr}_{file}_singular_values_histogram'
                       f'_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_singular_values_histogram'
                       f'_{graph_str}.png')

    with open(path + f'{timestr}_{file}_concatenated_singular_values'
                     f'_{graph_str}.json', 'w') \
            as outfile:
        json.dump(singularValues.tolist(), outfile)
    with open(path + f'{timestr}_{file}_singular_values_histogram'
                     f'_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
