# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.linalg import svdvals

N = 1000

s = 3
fontsize = 12
letter_posx, letter_posy = -0.15, 1.05

validCategories = ['barabasi_albert', 'watts_strogatz',
                   'random_regular']
m1, m2, m3 = 1, 2, 5
k1, k2, k3 = 2, 2, 10
p1, p2, p3 = 0.1, 0.6, 0.1
d1, d2, d3 = 3, 5, 10
random_graphs = [nx.barabasi_albert_graph(N, m1),
                 nx.barabasi_albert_graph(N, m2),
                 nx.barabasi_albert_graph(N, m3),
                 nx.connected_watts_strogatz_graph(N, k1, p1),
                 nx.connected_watts_strogatz_graph(N, k2, p2),
                 nx.connected_watts_strogatz_graph(N, k3, p3),
                 nx.random_regular_graph(d1, N),
                 nx.random_regular_graph(d2, N),
                 nx.random_regular_graph(d3, N)]
cat_random = [f"Barabási-Albert ($m = {m1}$)", f"Barabási-Albert ($m = {m2}$)",
              f"Barabási-Albert ($m = {m3}$)",
              f"Watts-Strogatz ($k = {k1}$, $p = {p1}$)",
              f"Watts-Strogatz ($k = {k2}$, $p = {p2}$)",
              f"Watts-Strogatz ($k = {k3}$, $p = {p3}$)",
              f"Random regular ($d = {d1})$", f"Random regular ($d = {d2})$",
              f"Random regular ($d = {d3})$"]
colors_brewer = ["#3182bd", "#9ecae1", "#deebf7", "#e6550d", "#fdae6b",
                 "#fee6ce", "#31a354", "#a1d99b", "#e5f5e0"]
colors = ["#C44E52", "#DD8452", "#55A868", "#DA8BC3", "#8172B3",
          "#937860", "#64B5CD", "#8C8C8C",  "#4C72B0"]
colorMap = dict(zip(validCategories, colors))


cat_nonrandom = ["Self-loops", 'Path', 'Hexagonal grid',
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
for i, g in enumerate(tqdm(graphs, position=0, desc="Graphs",
                           leave=True, ncols=80)):
    A = nx.to_numpy_array(g)
    singularValues = svdvals(A)
    rescaled_singular_values = singularValues / np.max(singularValues)
    indices = np.arange(1, len(rescaled_singular_values) + 1, 1)
    ax1.scatter(indices / len(indices), rescaled_singular_values, s=s,
                label=cat_nonrandom[i], color=colors[i])
plt.ylabel("Rescaled singular values $\\sigma_i/\\sigma_1$")
plt.xlabel("Rescaled index $i/N$")
plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=1.05)  # top=0.22)
plt.legend(loc=1, fontsize=fontsize_legend-5, bbox_to_anchor=(1, 0.95))

ax2 = plt.subplot(122)
ax2.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)
ax2.plot([0, 1], [1, 0], linestyle='--', color="#ABABAB")
for i, g in enumerate(tqdm(random_graphs, position=0, desc="Random graphs",
                           leave=True, ncols=80)):
        A = nx.to_numpy_array(g)
        singularValues = svdvals(A)
        rescaled_singular_values = singularValues/np.max(singularValues)
        indices = np.arange(1, len(rescaled_singular_values) + 1, 1)
        ax2.scatter(indices/len(indices), rescaled_singular_values, s=s,
                    label=cat_random[i], color=colors_brewer[i])
    # ticks = ax1.get_xticks()
    # ticks[ticks.tolist().index(0)] = 1
    # ticks = [i for i in ticks
    #          if -0.1 * len(singularValues) < i < 1.1 * len(singularValues)]
    # plt.xticks(ticks)
# plt.ylabel("Rescaled singular\n values $\\sigma_i/\\sigma_1$")
plt.xlabel("Rescaled index $i/N$")
plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=1.05)  # top=0.22)
leg = plt.legend(loc=1, fontsize=fontsize_legend-5, frameon=True,
                 framealpha=0.92)
leg.get_frame().set_linewidth(0.0)

plt.show()
