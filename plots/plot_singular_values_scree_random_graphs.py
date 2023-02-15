# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import networkx as nx
from tqdm import tqdm
from graphs.generate_random_graphs import random_graph_generators
from scipy.linalg import svdvals

N = 1000

s = 3
fontsize = 12
letter_posx, letter_posy = -0.15, 1.05

validCategories = ['gnp', 'chung_lu', 'soft_configuration_model', 'SBM', 's1',
                   'barabasi_albert', 'watts_strogatz', 'random_regular',
                   'perturbed_gaussian']
name_dictionary = {'gnp': '$G(N, p)$', 'chung_lu': 'Chung-Lu',
                   'soft_configuration_model': 'Soft configuration model',
                   's1': '$S^1$', 'barabasi_albert': 'Barabási-Albert',
                   'SBM': 'Stochastic block model',
                   'watts_strogatz': 'Watts-Strogatz',
                   'random_regular': "Random regular",
                   'perturbed_gaussian': "Rank-perturbed Gaussian"}
colors = ["#C44E52", "#DD8452", "#55A868", "#DA8BC3", "#8172B3",
          "#937860", "#64B5CD", "#8C8C8C",  "#4C72B0"]
colorMap = dict(zip(validCategories, colors))


cat_nonrandom = ["Disconnected self-loops", 'Path', 'Hexagonal grid',
                 'Square grid', 'Cubic grid', 'Triangular grid',
                 'Wheel', 'Star']
graphs = [nx.from_numpy_matrix(np.eye(N)), nx.path_graph(N),
          nx.hexagonal_lattice_graph(32, 32), nx.grid_graph(dim=(32, 32)),
          nx.grid_graph(dim=(10, 10, 10)),
          nx.triangular_lattice_graph(32, 32), nx.wheel_graph(N),
          nx.star_graph(N)]

fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot(121)
ax1.text(letter_posx, letter_posy, "a", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)
ax1.plot([0, 1], [1, 0], linestyle='--', color="#ABABAB")
for i, g in enumerate(tqdm(graphs)):
    A = nx.to_numpy_array(g)
    singularValues = svdvals(A)
    rescaled_singular_values = singularValues / np.max(singularValues)
    indices = np.arange(1, len(rescaled_singular_values) + 1, 1)
    ax1.scatter(indices / len(indices), rescaled_singular_values, s=s,
                label=cat_nonrandom[i], color=colors[i])
plt.ylabel("Rescaled singular\n values $\\sigma_i/\\sigma_1$")
plt.xlabel("Rescaled index $i/N$")
plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=1.05)  # top=0.22)
plt.legend(loc=1, fontsize=fontsize_legend-5)

ax2 = plt.subplot(122)
ax2.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)
ax2.plot([0, 1], [1, 0], linestyle='--', color="#ABABAB")
for cat in tqdm(validCategories):

    if cat == "perturbed_gaussian":
        G, args = random_graph_generators(cat, N)
        A = G(*args)
        singularValues = svdvals(A)
        rescaled_singular_values = singularValues / np.max(singularValues)
        indices = np.arange(1, len(rescaled_singular_values) + 1, 1)
        ax2.scatter(indices / len(indices), rescaled_singular_values, s=s,
                    label=name_dictionary[cat], color=colorMap[cat])

    else:
        G, args = random_graph_generators(cat, N)
        A = nx.to_numpy_array(G(*args))
        singularValues = svdvals(A)
        rescaled_singular_values = singularValues/np.max(singularValues)
        indices = np.arange(1, len(rescaled_singular_values) + 1, 1)
        ax2.scatter(indices/len(indices), rescaled_singular_values, s=s,
                    label=name_dictionary[cat], color=colorMap[cat])
    # ticks = ax1.get_xticks()
    # ticks[ticks.tolist().index(0)] = 1
    # ticks = [i for i in ticks
    #          if -0.1 * len(singularValues) < i < 1.1 * len(singularValues)]
    # plt.xticks(ticks)
# plt.ylabel("Rescaled singular\n values $\\sigma_i/\\sigma_1$")
plt.xlabel("Rescaled index $i/N$")
plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=1.05)  # top=0.22)
plt.legend(loc=1, fontsize=fontsize_legend-5)

plt.show()
