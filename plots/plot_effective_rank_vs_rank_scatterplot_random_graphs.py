# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault
import seaborn as sns
from plots.config_rcparams import *
from graphs.generate_random_graphs import *
from scipy.linalg import svdvals
from singular_values.compute_effective_ranks import \
    computeRank, computeStableRank


""" Plots the effective ranks to ratios
 vs the ranks to ratios of various random graphs. """

figureFilenamePDF = 'figures/pdf/effective_rank_vs_rank_random_graphs.pdf'
figureFilenamePNG = 'figures/png/effective_rank_vs_rank_random_graphs.png'


# Valid categories, in order of priority.
validCategories = ["disconnected_self_loops", 'gnp', 'chung_lu', 'SBM', 's1',
                   'barabasi_albert', 'watts_strogatz', 'random_regular']
name_dictionary = {"disconnected_self_loops": "Disconnected self-loops",
                   'gnp': '$G(N, p)$', 'chung_lu': 'Chung-Lu', 's1': '$S^1$',
                   'barabasi_albert': 'Barabási-Albert',
                   'SBM': 'Stochastic block model',
                   'watts_strogatz': 'Watts-Strogatz',
                   'random_regular': "Random regular"}
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
nb_instances = 500
fontsize = 12
letter_posx, letter_posy = -0.1, 1.08

plt.figure(figsize=(8, 4))

ax2 = plt.subplot(122)
ax2.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)
ax2.plot([0, 1], [0, 1], linestyle='--', color="#ABABAB")
for i, cat in enumerate(validCategories):
    print(cat)
    for j in range(nb_instances):
        G, args = random_graph_ensemble_generators(cat, N)
        A = nx.to_numpy_array(G(*args))
        singularValues = svdvals(A)
        rank = computeRank(singularValues)
        srank = computeStableRank(singularValues)
        sns.scatterplot(x=[rank/N], y=[srank/N],
                        facecolor='None',
                        edgecolor=colorMap[cat], alpha=0.7,
                        marker=markerMap[cat])
        # cat = sorted(validCategories)[i]
        if not j:
            label_random_graph = name_dictionary[cat]
        else:
            label_random_graph = None
        sns.scatterplot(x=[0], y=[1.5], facecolor='None',
                        edgecolor=colorMap[cat],
                        marker=markerMap[cat], label=label_random_graph)
# label=cat.lower(),
# zorder=len(validCategories)-i)

plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=1.05)  # top=0.22)

plt.xlabel('rank/N', fontsize=fontsize)
plt.ylabel('srank/N', fontsize=fontsize)

plt.legend(loc=2, fontsize=fontsize_legend-5)


ax1 = plt.subplot(121)
# validCategories = ['gnp', 'chung_lu', 'SBM', 's1', 'barabasi_albert',
#                    'watts_strogatz', 'random_regular', 'empty_graph']
ax1.text(letter_posx, letter_posy, "a", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)
ax1.plot([0, 1], [1, 0], linestyle='--', color="#ABABAB")
for cat in validCategories:
    G, args = random_graph_generators(cat, N)
    A = nx.to_numpy_array(G(*args))
    singularValues = svdvals(A)
    rescaled_singular_values = singularValues/np.max(singularValues)
    indices = np.arange(1, len(rescaled_singular_values) + 1, 1)
    ax1.scatter(indices/len(indices), rescaled_singular_values, s=10,
                label=name_dictionary[cat], color=colorMap[cat])
    plt.ylabel("Rescaled singular\n values $\\sigma_i/\\sigma_1$")
    plt.xlabel("Rescaled index $i/N$")
    ticks = ax1.get_xticks()
    # ticks[ticks.tolist().index(0)] = 1
    # ticks = [i for i in ticks
    #          if -0.1 * len(singularValues) < i < 1.1 * len(singularValues)]
    plt.xticks(ticks)

plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=1.05)  # top=0.22)
plt.legend(loc=1, fontsize=fontsize_legend-5)

plt.show()

# plt.savefig(figureFilenamePDF)  # , bbox_inches='tight')
# plt.savefig(figureFilenamePNG)  # , bbox_inches='tight')
