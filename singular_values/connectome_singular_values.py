# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import scipy.linalg as la
from plots.plot_singular_values import plot_singular_values
from sklearn.utils.extmath import randomized_svd
from graphs.get_real_networks import get_connectome_weight_matrix
from singular_values.compute_effective_ranks import *

compute_singvals = False

networkName = "mouse_voxel"  # "drosophila"
singularValuesFilename = 'properties/' + networkName \
                         + '_singular_values.txt'

if compute_singvals:
    W = get_connectome_weight_matrix(networkName)
    N = len(W[0])
    # U, singularValues, Vh = randomized_svd(W, 200)
    singularValues = la.svdvals(W)

    with open(singularValuesFilename, 'wb') as singularValuesFile:
        singularValuesFile.write('# Singular values\n'.encode('ascii'))
        np.savetxt(singularValuesFile, singularValues)
else:
    if networkName == "drosophila":
        N = 21733
    elif networkName == "mouse_voxel":
        N = 15314

    singularValues = np.loadtxt(singularValuesFilename)

print(computeEffectiveRanks(singularValues, networkName, N))

plot_singular_values(singularValues, effective_ranks=0)


""" To keep for the svg save ... """
# rank = computeRank(singularValues)
# stableRank = computeStableRank(singularValues)
# gavishDonohoThreshold = optht(1, sv=singularValues, sigma=None)
# elbowRank = findEffectiveRankElbow(singularValues)
# erank = computeERank(singularValues)
#
# print(f"Rank = {rank}")
# print(f"Stable rank = {stableRank}")
# print(f"Gavish-Donoho threshold = {gavishDonohoThreshold}")
# print(f"Elbow = {elbowRank}")
# print(f"Erank = {erank}")
#
# fig, ax = plt.subplots(1, figsize=(3, 2.8))
# plt.axvline(x=21687, linestyle="--",
#             color=reduced_grey, label="Rank")
# plt.axvline(x=gavishDonohoThreshold, linestyle="--",
#             color=reduced_first_community_color,
#             label="Gavish-Donoho")
# plt.axvline(x=stableRank, linestyle="--",
#             color=reduced_second_community_color,
#             label="Stable rank")
# plt.axvline(x=elbowRank, linestyle="--",
#             color=reduced_third_community_color,
#             label="Elbow")
# plt.axvline(x=erank, linestyle="--",
#             color=reduced_fourth_community_color,
#             label="erank")
# ax.scatter(np.arange(1, len(singularValues) + 1, 1),
#            singularValues / singularValues[0], s=10)
# # ax.scatter(np.arange(1, len(singularValues) + 1, 1), singularValues, s=10)
# plt.ylabel("Normalized singular\n values $\sigma_i/\sigma_1$")
# # plt.ylabel("Singular values $\sigma_i$")
# plt.xlabel("Index $i$")
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# # plt.xlim([1, 1.1*len(singularValues)])
# # plt.ylim([1, 10*max(singularValues)])
# plt.legend(loc=4, fontsize=8)
# # plt.tight_layout()
# ticks = ax.get_xticks()
# ticks[ticks.tolist().index(0)] = 1
# ticks = [i for i in ticks
#          if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
# plt.xticks(ticks)
# # fig.savefig('drosophila_singular_values_cover.svg', transparent=True)
# # sc.Figure("210cm", "200cm",
# #           sc.Panel(sc.SVG("figures/drosophile.svg")
# #                    .scale(2.5).move(140, 20)),
# #           sc.Panel(sc.SVG("drosophila_singular_values_cover.svg")))\
# #     .save("figures/drosophila_singular_values_compose.svg")
# # SVG('figures/drosophila_singular_values_compose.svg')
# plt.show()
