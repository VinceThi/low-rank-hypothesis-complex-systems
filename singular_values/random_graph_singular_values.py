# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import networkx as nx
import scipy.linalg as la
from optht import optht
from singular_values.compute_effective_ranks import computeERank,\
    computeStableRank, findCoudePosition
from plots.config_rcparams import *

N = 1000
k = 2
p = 0
G = nx.watts_strogatz_graph(N, k, p)
# G = nx.barabasi_albert_graph(N, k)
A = nx.to_numpy_array(G)
print(A)

numericalZero = 1e-13
singularValues = la.svdvals(A)
# singularValues = singularValues[singularValues > numericalZero]


cumulative_explained_variance = []
for r in range(1, len(singularValues)+1):
    # explained_variance.append(S[r]**2/np.sum(S**2))
    cumulative_explained_variance.append(
        np.sum(singularValues[0:r]**2)/np.sum(singularValues**2))

plt.figure(figsize=(8, 4))
ax1 = plt.subplot(121)
plt.scatter(np.arange(1, len(singularValues)+1, 1), singularValues)
plt.axvline(x=np.linalg.matrix_rank(A), linestyle="--",
            color=reduced_grey, label="Rank")
# Gavish donoho ne fonctionne pas ...? thres = 0 pour watts-strogatz...
plt.axvline(x=optht(1, sv=singularValues, sigma=None),
            linestyle="--", color=reduced_first_community_color,
            label="Gavish-Donoho threshold")
print(optht(1, sv=singularValues, sigma=None))
plt.axvline(x=computeStableRank(singularValues),
            linestyle="--", color=reduced_second_community_color,
            label="Stable rank")
plt.axvline(x=findCoudePosition(singularValues),
            linestyle="--", color=reduced_third_community_color,
            label="Elbow position")
plt.axvline(x=computeERank(singularValues),
            linestyle="--", color=reduced_fourth_community_color,
            label="erank")
plt.ylabel("Singular values $\\sigma_i$")
plt.xlabel("Index $i$")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
ticks = ax1.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])

ax2 = plt.subplot(122)
plt.scatter(np.arange(1, len(singularValues)+1, 1),
            cumulative_explained_variance, zorder=1)
plt.xlabel("Number of singular values $n$", fontsize=12)
plt.ylabel("Cumulative explained variance $F(n)$", fontsize=12)
# "Cumulative explained variance
#  $\\sum_{j=1}^n\\sigma_j^2/\\sum_{j=1}^N \sigma_j^2$"
ticks = ax2.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])
plt.ylim([-0.05, 1.05])
plt.tight_layout()
plt.show()
