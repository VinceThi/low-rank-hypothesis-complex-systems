# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import scipy.linalg as la
# from plots.plot_singular_values import plot_singular_values, \
#     plot_singular_values_histogram
# from sklearn.utils.extmath import randomized_svd
from plots.config_rcparams import *
from graphs.get_real_networks import get_connectome_weight_matrix
from singular_values.compute_effective_ranks import *
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes


networkName = "drosophila"
# "zebrafish_meso", "drosophila","mouse_voxel", ...
singularValuesFilename = 'C:/Users/thivi/Documents/GitHub/' \
                         'low-rank-hypothesis-complex-systems/' \
                         'singular_values/properties/' + networkName \
                         + '_singular_values.txt'

if networkName == "drosophila":
    N = 21733
    singularValues = np.loadtxt(singularValuesFilename)
elif networkName == "mouse_voxel":
    N = 15314
    singularValues = np.loadtxt(singularValuesFilename)

else:
    W = get_connectome_weight_matrix(networkName)
    N = len(W[0])
    # U, singularValues, Vh = randomized_svd(W, 200)
    singularValues = la.svdvals(W)

    with open(singularValuesFilename, 'wb') as singularValuesFile:
        singularValuesFile.write('# Singular values\n'.encode('ascii'))
        np.savetxt(singularValuesFile, singularValues)

print(computeEffectiveRanks(singularValues, networkName, N))

""" Compute effective ranks """
rank, thrank, shrank, erank, elbow, energy, srank, nrank =\
    [computeRank(singularValues),
     computeOptimalThreshold(singularValues),
     computeOptimalShrinkage(singularValues),
     computeERank(singularValues),
     findEffectiveRankElbow(singularValues),
     computeEffectiveRankEnergyRatio(singularValues,
                                     threshold=0.9),
     computeStableRank(singularValues),
     computeNuclearRank(singularValues)]


""" Plot singular values for the drosophila melanogaster's connectome """
plot_inset = 0
plot_first_version = 0

if plot_inset:
    ymin = 10**(-2)
    fig, ax = plt.subplots(1, figsize=(4, 2))
    normalized_singular_values = singularValues / singularValues[0]
    ax.scatter(np.arange(1, len(singularValues) + 1, 1),
               normalized_singular_values, s=10, color=deep[0])
    plt.vlines(x=srank, ymin=ymin, ymax=normalized_singular_values[11],
               linestyle="--", color=deep[1], label="srank")
    plt.vlines(x=nrank, ymin=ymin, ymax=normalized_singular_values[172],
               linestyle="--", color=deep[2], label="nrank")
    plt.vlines(x=elbow, ymin=ymin, ymax=normalized_singular_values[723],
               linestyle="--", color=deep[4], label="elbow")
    plt.vlines(x=energy, ymin=ymin, ymax=normalized_singular_values[1061],
               linestyle="--", color=deep[3], label="energy")
    plt.vlines(x=thrank, ymin=ymin, ymax=normalized_singular_values[4867],
               linestyle="--", color=deep[6], label="thrank")
    plt.vlines(x=shrank, ymin=ymin, ymax=normalized_singular_values[5542],
               linestyle="--", color=deep[7], label="shrank")
    plt.vlines(x=erank, ymin=ymin, ymax=normalized_singular_values[6702],
               linestyle="--", color=deep[5], label="erank")
    plt.vlines(x=rank, ymin=normalized_singular_values[rank - 1], ymax=1,
               linestyle="--", color=deep[9], label="rank")
    plt.xticks([11, 173, 724, 1062])
    ax.set_xlim([1, 1100])
    ax.set_ylim([ymin, 1.5])
    ax.set_yscale('log')
    plt.tick_params(axis='y', which='both', left=True,
                    right=False, labelbottom=False)
    plt.show()

elif plot_first_version:  # Version of the first submission
    # Warning: the zero singular values are at 10^(-7)
    ymin = 10**(-7)
    fig, ax = plt.subplots(1, figsize=(5, 5))
    normalized_singular_values = singularValues / singularValues[0]
    ax.scatter(np.arange(1, len(singularValues)+1, 1),
               normalized_singular_values, s=10, color=deep[0])
    zero_singvals_indices = np.arange(21687, 21733 + 1, 1)
    ax.scatter(zero_singvals_indices,
               1.1*ymin*np.ones(len(zero_singvals_indices)), s=10,
               color=deep[0])
    plt.vlines(x=srank, ymin=ymin, ymax=normalized_singular_values[11],
               linestyle="--", color=deep[1], label="srank = 12")
    plt.vlines(x=nrank, ymin=ymin, ymax=normalized_singular_values[172],
               linestyle="--", color=deep[2], label="nrank = 173")
    plt.vlines(x=elbow, ymin=ymin, ymax=normalized_singular_values[723],
               linestyle="--", color=deep[4], label="elbow = 724")
    plt.vlines(x=energy, ymin=ymin, ymax=normalized_singular_values[1061],
               linestyle="--", color=deep[3], label="energy = 1062")
    plt.vlines(x=thrank, ymin=ymin, ymax=normalized_singular_values[4867],
               linestyle="--", color=deep[6], label="thrank = 4868")
    plt.vlines(x=shrank, ymin=ymin, ymax=normalized_singular_values[5542],
               linestyle="--", color=deep[7], label="shrank = 5543")
    plt.vlines(x=erank, ymin=ymin, ymax=normalized_singular_values[6702],
               linestyle="--", color=deep[5], label="erank = 6703")
    plt.vlines(x=rank, ymin=normalized_singular_values[rank - 1], ymax=1,
               linestyle="--", color=deep[9], label="rank = 21687")

    # ax.text(11.5811 / N + 300, normalized_singular_values[10],
    #         "11", fontsize=8, color=deep[1])
    #
    # ax.text(173.483 / N + 400, normalized_singular_values[172],
    #         "173", fontsize=8, color=deep[2])
    #
    # ax.text(724+10, normalized_singular_values[723]+0.005,
    #         "724", fontsize=8, color=deep[0])
    #
    # ax.text(1062+100, normalized_singular_values[1061],
    #         "1062", fontsize=8, color=deep[0])
    #
    # ax.text(4868-500, normalized_singular_values[4867]+0.001,
    #         "4868", fontsize=8, color=dark_grey)
    #
    # ax.text(5543-400, normalized_singular_values[5542]+0.001,
    #         "5543", fontsize=8, color=dark_grey)
    #
    # ax.text(6702-300, normalized_singular_values[6701]+0.001,
    #         "6701", fontsize=8, color=dark_grey)
    #
    # ax.text(21687-2000, normalized_singular_values[21686]-10**(-6),
    #         "21687", fontsize=8, color=dark_grey)

    plt.ylabel("Normalized singular values $\sigma_i/\sigma_1$")
    plt.xlabel("Index $i$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.62, 0.04, 0, 0))
    ticks = ax.get_xticks()
    ticks[ticks.tolist().index(0)] = 1
    ticks = [i for i in ticks
             if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
    plt.xticks(ticks)
    ax.set_ylim([ymin, 1.5])
    ax.set_yscale('log')
    plt.tick_params(axis='y', which='both', left=True,
                    right=False, labelbottom=False)
    # - To keep for the svg save ...
    # fig.savefig('drosophila_singular_values_cover.svg', transparent=True)
    # sc.Figure("210cm", "200cm",
    #           sc.Panel(sc.SVG("figures/drosophile.svg")
    #                    .scale(2.5).move(140, 20)),
    #           sc.Panel(sc.SVG("drosophila_singular_values_cover.svg")))\
    #     .save("figures/drosophila_singular_values_compose.svg")
    # SVG('figures/drosophila_singular_values_compose.svg')

    plt.show()

else:
    # Warning: the zero singular values are at 10^(-7)
    ymin = 10**(-7)
    fig, ax = plt.subplots(1, figsize=(2.2, 2))
    normalized_singular_values = singularValues / singularValues[0]
    s = 2
    max_scatter = 50
    indices = np.arange(1, len(singularValues) + 1, 1)
    ax.scatter(indices[:max_scatter], normalized_singular_values[:max_scatter],
               s=s, color=deep[0])
    ax.plot(indices[max_scatter:21686],
            normalized_singular_values[max_scatter:21686],
            linewidth=2.2, color=deep[0])
    zero_singvals_indices = np.arange(21687, 21733 + 1, 1)
    ax.scatter(zero_singvals_indices,
               1.1*ymin * np.ones(len(zero_singvals_indices)), s=s,
               color=deep[0])
    plt.vlines(x=srank, ymin=ymin, ymax=normalized_singular_values[11],
               linestyle="--", color=deep[1], label="srank = 12")
    plt.vlines(x=nrank, ymin=ymin, ymax=normalized_singular_values[172],
               linestyle="--", color=deep[2], label="nrank = 173")
    plt.vlines(x=elbow, ymin=ymin, ymax=normalized_singular_values[723],
               linestyle="--", color=deep[4], label="elbow = 724")
    plt.vlines(x=energy, ymin=ymin, ymax=normalized_singular_values[1061],
               linestyle="--", color=deep[3], label="energy = 1062")
    plt.vlines(x=thrank, ymin=ymin, ymax=normalized_singular_values[4867],
               linestyle="--", color=deep[6], label="thrank = 4868")
    plt.vlines(x=shrank, ymin=ymin, ymax=normalized_singular_values[5542],
               linestyle="--", color=deep[7], label="shrank = 5543")
    plt.vlines(x=erank, ymin=ymin, ymax=normalized_singular_values[6702],
               linestyle="--", color=deep[5], label="erank = 6703")
    plt.vlines(x=rank, ymin=normalized_singular_values[rank - 1], ymax=1,
               linestyle="--", color=deep[9], label="rank = 21687")

    # ax.text(11.5811 / N + 300, normalized_singular_values[10],
    #         "11", fontsize=8, color=deep[1])
    #
    # ax.text(173.483 / N + 400, normalized_singular_values[172],
    #         "173", fontsize=8, color=deep[2])
    #
    # ax.text(724+10, normalized_singular_values[723]+0.005,
    #         "724", fontsize=8, color=deep[0])
    #
    # ax.text(1062+100, normalized_singular_values[1061],
    #         "1062", fontsize=8, color=deep[0])
    #
    # ax.text(4868-500, normalized_singular_values[4867]+0.001,
    #         "4868", fontsize=8, color=dark_grey)
    #
    # ax.text(5543-400, normalized_singular_values[5542]+0.001,
    #         "5543", fontsize=8, color=dark_grey)
    #
    # ax.text(6702-300, normalized_singular_values[6701]+0.001,
    #         "6701", fontsize=8, color=dark_grey)
    #
    # ax.text(21687-2000, normalized_singular_values[21686]-10**(-6),
    #         "21687", fontsize=8, color=dark_grey)

    ylab = plt.ylabel("$\\frac{\sigma_i}{\sigma_1}$", fontsize=18)
    ylab.set_rotation(0)
    plt.xlabel("$i$")
    # ax.legend(loc="lower center", bbox_to_anchor=(0.62, 0.04, 0, 0))
    ticks = ax.get_xticks()
    ticks[ticks.tolist().index(0)] = 1
    ticks = [i for i in ticks
             if -0.1 * len(singularValues) < i < 1.1 * len(singularValues)]
    ticks.remove(10000)
    ax.set_xticks(ticks)
    ax.xaxis.set_label_coords(0.4, -0.09)
    plt.tick_params(axis='y', which='both', left=True,
                    right=False, labelbottom=False)
    ax.set_yscale('log')
    ax.yaxis.set_label_coords(-0.2, 0.35)
    ax.set_ylim([ymin, 1.5])
    ax.set_yticks([1, 10**(-7)])
    plt.tight_layout()
    # - To keep for the svg save ...
    # fig.savefig('drosophila_singular_values_cover.svg', transparent=True)
    # sc.Figure("210cm", "200cm",
    #           sc.Panel(sc.SVG("figures/drosophile.svg")
    #                    .scale(2.5).move(140, 20)),
    #           sc.Panel(sc.SVG("drosophila_singular_values_cover.svg")))\
    #     .save("figures/drosophila_singular_values_compose.svg")
    # SVG('figures/drosophila_singular_values_compose.svg')

    plt.show()

