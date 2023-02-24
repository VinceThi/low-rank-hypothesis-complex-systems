# -*- coding: utf-8 -*-\\
# @author: Vincent Thibeault

from singular_values.compute_effective_ranks import\
    computeEffectiveRanksRandomGraphs
from graphs.generate_random_graphs import random_graph_generators
import json
import numpy as np
from plots.config_rcparams import *

compute_ranks = True
path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/singular_values/properties/" \
           "singular_values_random_graphs/"


N = 1000
nb_graphs = 100
graph_str = "perturbed_gaussian"
# Choices of graph_str
# gnp, SBM, watts_strogatz, barabasi_albert,
#  hard_configuration, directed_hard_configuration,
# chung_lu, s1, random_regular, GOE, GUE, COE,
# CUE, perturbed_gaussian, soft_configuration_model,
# weighted_soft_configuration_model
G, args = random_graph_generators(graph_str, N)

if compute_ranks:
    computeEffectiveRanksRandomGraphs(G, args, nb_graphs=nb_graphs)

else:
    path = "2023_02_03_09h31min01sec_100_samples_concatenated" \
           "_effective_ranks_perturbed_gaussian.json"
    with open(path_str + path) as json_data:
        effective_ranks = np.array(json.load(json_data))

    rank = effective_ranks[0, :]
    thrank = effective_ranks[1, :]
    shrank = effective_ranks[2, :]
    erank = effective_ranks[3, :]
    elbow = effective_ranks[4, :]
    energy = effective_ranks[5, :]
    srank = effective_ranks[6, :]
    nrank = effective_ranks[7, :]

    ylabel = "Density"
    bar_color = "#064878"
    # nb_bins = 1000
    letter_posx, letter_posy = -0.27, 1.08
    fontsize_legend = 10
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) =\
        plt.subplots(nrows=2, ncols=4, figsize=(10, 5))

    weights = np.ones_like(srank) / float(len(srank))
    ax1.hist(srank,  # bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    ax1.tick_params(axis='both', which='major')
    ax1.set_xlabel("srank")
    ax1.set_ylabel(ylabel, labelpad=10)
    ax1.text(letter_posx, letter_posy, "a", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=ax1.transAxes)
    # ax1.set_ylim([0, 0.065])
    ax1.set_xlim([2, 2.1])
    # ax1.set_xticks([0, 10, 20, 30])

    weights = np.ones_like(nrank) / float(len(nrank))
    ax2.hist(nrank,  # bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    ax2.tick_params(axis='both', which='major')
    ax2.set_xlabel("nrank")
    ax2.text(letter_posx, letter_posy, "b", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=ax2.transAxes)
    # ax2.set_ylim([0, 0.055])
    ax2.set_xlim([24, 26])
    # ax2.set_xticks([0, 10, 20, 30])

    weights = np.ones_like(elbow) / float(len(elbow))
    ax3.hist(elbow,  # bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    ax3.tick_params(axis='both', which='major')
    ax3.set_xlabel("elbow")
    ax3.text(letter_posx, letter_posy, "c", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=ax3.transAxes)
    ax3.set_ylim([0, 1.05])
    # ax3.set_xlim([24, 26])
    # ax3.set_xticks([0, 10, 20, 30])

    weights = np.ones_like(energy) / float(len(energy))
    ax4.hist(energy,  # bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    ax4.tick_params(axis='both', which='major')
    ax4.set_xlabel("energy")
    ax4.text(letter_posx, letter_posy, "d", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=ax4.transAxes)
    # ax4.set_ylim([0, 0.055])
    # ax4.set_xlim([24, 26])

    weights = np.ones_like(thrank) / float(len(thrank))
    ax5.hist(thrank,  # bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    ax5.tick_params(axis='both', which='major')
    ax5.set_xlabel("thrank")
    ax5.set_ylabel(ylabel, labelpad=10)
    ax5.text(letter_posx, letter_posy, "e", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=ax5.transAxes)
    ax5.set_ylim([0, 1.05])
    # ax5.set_xlim([24, 26])

    weights = np.ones_like(shrank) / float(len(shrank))
    ax6.hist(shrank,  # bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    ax6.tick_params(axis='both', which='major')
    ax6.set_xlabel("shrank")
    ax6.text(letter_posx, letter_posy, "f", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=ax6.transAxes)
    ax6.set_ylim([0, 1.05])
    # ax6.set_xlim([24, 26])

    weights = np.ones_like(erank) / float(len(erank))
    ax7.hist(erank,  # bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    ax7.tick_params(axis='both', which='major')
    ax7.set_xlabel("erank")
    ax7.text(letter_posx, letter_posy, "g", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=ax7.transAxes)
    # ax7.set_ylim([0, 1.05])
    ax7.set_xlim([690, 700])

    weights = np.ones_like(rank) / float(len(rank))
    ax8.hist(rank,  # bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    ax8.tick_params(axis='both', which='major')
    ax8.set_xlabel("rank")
    ax8.text(letter_posx, letter_posy, "h", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=ax8.transAxes)
    ax8.set_ylim([0, 1.05])
    ax8.set_xlim([999, 1001])

    plt.show()
