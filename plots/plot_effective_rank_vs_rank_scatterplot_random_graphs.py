# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault
import seaborn as sns
from plots.config_rcparams import *
from graphs.random_graph_generators import *
from scipy.linalg import svdvals
from singular_values.compute_effective_ranks import \
    computeRank, computeStableRank


""" Plots the effective ranks to ratios
 vs the ranks to ratios of various random graphs. """

figureFilenamePDF = 'figures/pdf/effective_rank_vs_rank_random_graphs.pdf'
figureFilenamePNG = 'figures/png/effective_rank_vs_rank_random_graphs.png'


# Valid categories, in order of priority.
validCategories = ['gnp', 'chung_lu', 's1']
markers = ['*', 'h', '^', 'D', 's', 'v', 'd', 'X', '.', 's',
           'p', 'P', 'o', 'H']
colors = ["#C44E52", "#DD8452", "#55A868", "#DA8BC3", "#8172B3",
          "#937860", "#64B5CD", "#8C8C8C",  "#4C72B0"]
colorMap = dict(zip(validCategories, colors))
markerMap = dict(zip(validCategories, markers))

# rank = {cat: [] for cat in validCategories}
# effectiveRank = {cat: [] for cat in validCategories}
# for networkName in validCategories:
#
#     cat = [tag for tag in graphPropDF.loc[networkName]['tags'].split(',')
#            if tag in validCategories]
#     if len(cat) == 0:
#         cat = 'other'
#         print(networkName + ' does not have a category')
#     else:
#         cat = cat[0]
#
#     rank[cat].append(effectiveRanksDF.loc[networkName]['x'])
#     effectiveRank[cat].append(effectiveRanksDF.loc[networkName]['y'])

N = 100
nb_instances = 10

plt.figure(figsize=(4, 4))

for i, cat in enumerate(reversed(validCategories)):
    for j in range(nb_instances):
        G, args = random_graph_generators(cat, N)
        A = nx.to_numpy_array(G(args))
        singularValues = svdvals(A)
        rank = computeRank(singularValues)
        srank = computeStableRank(singularValues)
        sns.scatterplot(x=rank, y=srank,
                        facecolor='None',
                        edgecolor=colorMap[cat], alpha=0.7,
                        marker=markerMap[cat])
        cat = sorted(validCategories)[i]
        sns.scatterplot(x=[0], y=[1.5], facecolor='None',
                        edgecolor=colorMap[cat],
                        marker=markerMap[cat], label=cat.lower(),
                        zorder=len(validCategories)-i)

plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=0.22)

# plt.yscale('log')
# plt.xscale('log')

# g.ax_joint.set_xlabel('Rank to dimension ratio')
# g.ax_joint.set_ylabel('Effective rank to dimension ratio')

plt.xlabel('Rank to dimension ratio')
plt.ylabel('Effective rank to dimension ratio')
plt.show()

# plt.savefig(figureFilenamePDF)  # , bbox_inches='tight')
# plt.savefig(figureFilenamePNG)  # , bbox_inches='tight')
