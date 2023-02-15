# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_networks import *
import json
from plots.config_rcparams import *


path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "singular_values/properties/"

""" Soft configuration model """
graph_str_scm = "soft_configuration_model"
spec_path_str_scm = "2023_02_14_13h30min07sec_100graphs_50s"
effrank_indicator_list_scm = [1, 0, 1, 0, 1, 0, 0, 0]

""" SBM """
graph_str_SBM = "sbm"
spec_path_str_SBM_svdvals = "2023_02_10_12h24min20sec_100graphs_50s"
spec_path_str_SBM = "2023_02_10_12h24min20sec_100graphs_50s"
effrank_indicator_list_SBM = [1, 0, 1, 0, 1, 0, 0, 0]

""" S1 """
graph_str_S1 = "s1"
spec_path_str_S1 = "2023_02_10_12h24min44sec_100graphs_50s"
effrank_indicator_list_S1 = [1, 0, 1, 0, 1, 0, 0, 0]

""" Perturbed gaussian """
graph_str_gauss = "perturbed_gaussian"
spec_path_str_gauss = "2023_02_15_09h12min32sec_1000graphs_50s"
effrank_indicator_list_gauss = [1, 0, 1, 0, 1, 0, 0, 0]


def plot_effective_ranks_random_graphs(ax, graph_str, spec_path_str,
                                       effrank_indicator_list):
    effrank_str_list = ["srank", "nrank", "elbow", "energy", "thrank",
                        "shrank", "erank", "rank"]

    with open(path_str + graph_str
              + f"/{spec_path_str}_{graph_str}_parameters_dictionary.json")\
            as json_data:
        parameters_dictionary = json.load(json_data)
    N = parameters_dictionary["N"]

    with open(path_str + graph_str
              + f"/{spec_path_str}_norm_R_{graph_str}.json") as json_data:
        norm_R = np.array(json.load(json_data))
    with open(path_str + graph_str +
              f"/{spec_path_str}_norm_EW_{graph_str}.json") as json_data:
        norm_EW = np.array(json.load(json_data))

    norm_ratio = np.mean(norm_R, axis=0) / norm_EW

    xlabel = "$\\langle ||R||_F \\rangle\,/\,||\\langle W \\rangle||_F$"
    alpha = 0.2
    s = 3
    j = 0
    for i in range(len(effrank_str_list)):
        if effrank_indicator_list[i]:
            with open(path_str + graph_str +
                      f"/{spec_path_str}_{effrank_str_list[i]}"
                      f"_{graph_str}.json") as json_data:
                effrank = np.array(json.load(json_data))
            mean_effrank = np.mean(effrank, axis=0) / N
            std_effrank = np.std(effrank, axis=0) / N
            ax.scatter(norm_ratio, mean_effrank, color=deep[j], s=s,
                       label=effrank_str_list[i] + "/N")
            ax.fill_between(norm_ratio, mean_effrank - std_effrank,
                            mean_effrank + std_effrank, color=deep[j],
                            alpha=alpha)
            j += 1
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel(xlabel)
    ax.legend(loc=1, fontsize=fontsize_legend)


""" Plot """

letter_posx, letter_posy = -0.27, 1.08
fontsize_legend = 8
fontsize_title = 9.5
title_pad = 0

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) =\
    plt.subplots(nrows=2, ncols=4, figsize=(10, 5))

ax1.set_title(" Rank-perturbed Gaussian", fontsize=fontsize_title,
              pad=title_pad)
ax1.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)

ax2.set_title("Stochastic block model", fontsize=fontsize_title,
              pad=title_pad)
ax2.tick_params(axis='both', which='major')
# ax2.set_ylabel("nrank/N")
# ax2.set_xlabel(xlabel)
ax2.text(letter_posx, letter_posy, "c", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)

ax3.set_title("Soft configuration model", fontsize=fontsize_title,
              pad=title_pad)
ax3.tick_params(axis='both', which='major')
# ax3.set_ylabel("elbow/N")
# ax3.set_xlabel(xlabel)
ax3.text(letter_posx, letter_posy, "d", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax3.transAxes)

ax4.set_title("Random geometric $S^1$", fontsize=fontsize_title,
              pad=title_pad)
ax4.tick_params(axis='both', which='major')
# ax4.set_ylabel("energy/N")
# ax4.set_xlabel(xlabel)
ax4.text(letter_posx, letter_posy, "e", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax4.transAxes)

plot_effective_ranks_random_graphs(ax5, graph_str_gauss, spec_path_str_gauss,
                                   effrank_indicator_list_gauss)
ax5.tick_params(axis='both', which='major')
# ax5.set_ylabel("thrank/N")
# ax5.set_xlabel(xlabel)
ax5.text(letter_posx, letter_posy, "f", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax5.transAxes)

plot_effective_ranks_random_graphs(ax6, graph_str_SBM, spec_path_str_SBM,
                                   effrank_indicator_list_SBM)
ax6.set_xlim([0.5, 2])
ax6.tick_params(axis='both', which='major')
# ax6.set_ylabel("shrank/N")
# ax6.set_xlabel(xlabel)
ax6.text(letter_posx, letter_posy, "g", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax6.transAxes)

plot_effective_ranks_random_graphs(ax7, graph_str_scm, spec_path_str_scm,
                                   effrank_indicator_list_scm)
ax7.tick_params(axis='both', which='major')
# ax7.set_ylabel("erank/N")
# ax7.set_xlabel(xlabel)
ax7.text(letter_posx, letter_posy, "h", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax7.transAxes)

plot_effective_ranks_random_graphs(ax8, graph_str_S1, spec_path_str_S1,
                                   effrank_indicator_list_S1)
ax8.tick_params(axis='both', which='major')
# ax8.set_ylabel("rank/N")
# ax8.set_xlabel(xlabel)
ax8.text(letter_posx, letter_posy, "i", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax8.transAxes)

plt.show()
