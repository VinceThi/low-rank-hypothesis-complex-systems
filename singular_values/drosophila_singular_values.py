# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
# from sklearn.utils.extmath import randomized_svd
# import networkx as nx
# import pandas as pd
# import scipy.linalg as la
import svgutils.compose as sc
import numpy as np
from optht import optht
from singular_values.compute_effective_ranks import computeERank,\
    computeStableRank, findCoudePosition


"""
path_str = "C:/Users/thivi/Documents/GitHub/network-synch/" \
               "synch_predictions/plots/data/"

df = pd.read_csv(path_str + 'drosophila_exported-traced-adjacencies-v1.1/'
                            'traced-total-connections.csv')
Graphtype = nx.DiGraph()
G_drosophila = nx.from_pandas_edgelist(df,
                                       source='bodyId_pre',
                                       target='bodyId_post',
                                       edge_attr='weight',
                                       create_using=Graphtype)
W = nx.to_numpy_array(G_drosophila)
N = G_drosophila.number_of_nodes()
print(f"N = {N}")
"""

networkName = "drosophila"         # singular_values/
singularValuesFilename = 'properties/' + networkName \
                         + '_singular_values.txt'
"""

numericalZero = 1e-13
singularValues = la.svdvals(W)
singularValues = singularValues[singularValues > numericalZero]

# n = N
# U, singularValues, Vh = randomized_svd(W, n)
with open(singularValuesFilename, 'wb') as singularValuesFile:
    singularValuesFile.write('# Singular values\n'.encode('ascii'))
    np.savetxt(singularValuesFile, singularValues)
"""

singularValues = np.loadtxt(singularValuesFilename)
numberSingularValues = len(singularValues)
stableRank = computeStableRank(singularValues)
gavishDonohoThreshold = optht(1, sv=singularValues, sigma=None)
coudePosition = findCoudePosition(singularValues)
erank = computeERank(singularValues)

print(f"number of singular values = {numberSingularValues}")
print(f"Stable rank = {stableRank}")
print(f"Gavish-Donoho threshold = {gavishDonohoThreshold}")
print(f"Elbow position = {coudePosition}")
print(f"Erank = {erank}")

# Warning: N = 21733, but we have 21687 singular values

fig, ax = plt.subplots(1, figsize=(3, 2.8))
# plt.axvline(x=21687, linestyle="--",
#             color=reduced_grey, label="Rank")
# plt.axvline(x=gavishDonohoThreshold, linestyle="--",
#             color=reduced_first_community_color,
#             label="Gavish-Donoho")
# plt.axvline(x=stableRank, linestyle="--",
#             color=reduced_second_community_color,
#             label="Stable rank")
# plt.axvline(x=coudePosition, linestyle="--",
#             color=reduced_third_community_color,
#             label="Elbow position")
# plt.axvline(x=erank, linestyle="--",
#             color=reduced_fourth_community_color,
#             label="erank")
ax.scatter(np.arange(1, len(singularValues) + 1, 1),
           singularValues / singularValues[0], s=10)
# ax.scatter(np.arange(1, len(singularValues) + 1, 1), singularValues, s=10)
plt.ylabel("Normalized singular\n values $\sigma_i/\sigma_1$")
# plt.ylabel("Singular values $\sigma_i$")
plt.xlabel("Index $i$")
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.xlim([1, 1.1*len(singularValues)])
# plt.ylim([1, 10*max(singularValues)])
# plt.legend(loc=4, fontsize=8)
# plt.tight_layout()
# ticks = ax.get_xticks()
# ticks[ticks.tolist().index(0)] = 1
# ticks = [i for i in ticks
#          if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
# plt.xticks(ticks)
# fig.savefig('drosophila_singular_values_cover.svg', transparent=True)
# sc.Figure("210cm", "200cm",
#           sc.Panel(sc.SVG("figures/drosophile.svg")
#                    .scale(2.5).move(140, 20)),
#           sc.Panel(sc.SVG("drosophila_singular_values_cover.svg")))\
#     .save("figures/drosophila_singular_values_compose.svg")
# SVG('figures/drosophila_singular_values_compose.svg')
plt.show()
