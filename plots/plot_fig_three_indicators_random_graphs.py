# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_networks import *
import json
from plots.config_rcparams import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "singular_values/properties/"


""" Soft configuration model """
graph_str_scm = "soft_configuration_model"
spec_path_str_scm = "2023_02_21_14h24min17sec_50graphs_10s"
spec_path_str_scm_svdvals_1 = "2023_02_21_14h59min04sec_100graphs_g_1_6_r0_48"  # g inversed
spec_path_str_scm_svdvals_2 = "2023_02_21_14h58min14sec_100graphs_g_0_2_r0_55"  # g inversed
effrank_indicator_list_scm = [1, 0, 1, 0, 1, 1, 0, 0]

""" SBM """
graph_str_SBM = "sbm"
spec_path_str_SBM_svdvals_1 = "2023_02_20_14h58min44sec_8pq0_normratio_0_066"
spec_path_str_SBM_svdvals_2 = "2023_02_20_14h26min10sec_3pq0_normratio_0_1515"
spec_path_str_SBM = "2023_02_21_13h59min53sec_50graphs_10s"
effrank_indicator_list_SBM = [1, 0, 1, 0, 1, 1, 0, 0]

""" S1 """
graph_str_S1 = "s1"
spec_path_str_S1_svdvals_1 = "2023_02_20_21h06min58sec_100graphs" \
                             "_temp0_1_normratio_0_17"
spec_path_str_S1_svdvals_2 = "2023_02_20_21h07min42sec_100graphs" \
                             "_temp0_8_normratio_0_48"
spec_path_str_S1 = "2023_02_21_13h57min39sec_50graphs_10s"
effrank_indicator_list_S1 = [1, 0, 1, 0, 1, 1, 0, 0]

""" Perturbed gaussian """
graph_str_gauss = "perturbed_gaussian"
spec_path_str_gauss = "2023_02_21_13h58min54sec_50graphs_10s"
spec_path_str_gauss_svdvals_1 = "2023_02_21_14h40min09sec_100graphs_g_0_1"
spec_path_str_gauss_svdvals_2 = "2023_02_21_14h40min38sec_100graphs_g_2"
effrank_indicator_list_gauss = [1, 0, 1, 0, 1, 1, 0, 0]


def plot_singular_values_random_graphs(ax, graph_str, spec_path_str):
    # ylabel = "Average rescaled singular\n values $\\sigma_i/\\sigma_1$"
    # ylabel = "Average singular values"

    with open(path_str + "singular_values_random_graphs" +
              f"/{spec_path_str}_singular_values_"
              f"{graph_str}_parameters_dictionary.json")\
            as json_data:
        parameters_dictionary = json.load(json_data)
    N = parameters_dictionary["N"]
    nb_networks = parameters_dictionary["nb_samples (nb_networks)"]
    norm_R = np.array(parameters_dictionary["norm_R"])

    with open(path_str + "singular_values_random_graphs"
              + f"/{spec_path_str}_singular_values_W_{graph_str}.json")\
            as json_data:
        singularValues = np.array(json.load(json_data))
    with open(path_str + "singular_values_random_graphs"
              + f"/{spec_path_str}_singular_values_EW_{graph_str}.json")\
            as json_data:
        singularValues_EW = np.array(json.load(json_data))
    with open(path_str + "singular_values_random_graphs"
              + f"/{spec_path_str}_singular_values_R_{graph_str}.json")\
            as json_data:
        singularValues_R = np.array(json.load(json_data))

    mean_norm_W = np.mean(singularValues[:, 0])
    mean_singularValues = np.mean(singularValues, axis=0)
    bar_singularValues = np.std(singularValues, axis=0)

    mean_singularValues_R = np.mean(singularValues_R, axis=0)
    bar_singularValues_R = np.std(singularValues_R, axis=0)

    indices = np.arange(1, N + 1, 1)
    # ax.scatter(indices, mean_singularValues/mean_norm_W, s=30, color=deep[0],
    #             label="$\\langle\\sigma_i(W)\\rangle\,/"
    #                   "\,\\langle\,||W||_2\,\\rangle$")
    # ax.fill_between(indices,
    #                  (mean_singularValues - std_singularValues)/mean_norm_W,
    #                  (mean_singularValues + std_singularValues)/mean_norm_W,
    #                  color=deep[0], alpha=0.2)
    ax.errorbar(x=indices, y=mean_singularValues / mean_norm_W,
                yerr=bar_singularValues / mean_norm_W, fmt="o", color=deep[0],
                zorder=-30, markersize=5, capsize=1, elinewidth=1,
                label="$\\langle\\sigma_i(W)\\rangle\,/"
                      "\,\\langle\,||W||_2\,\\rangle$")
    # ax.scatter(indices, mean_singularValues_R/mean_norm_W,s=2, color=deep[9],
    #            label="$\\langle\\sigma_i(R)\\rangle\,/"
    #                  "\,\\langle\,||W||_2\,\\rangle$", zorder=2)
    # ax.fill_between(indices,
    #           (mean_singularValues_R - bar_singularValues_R)/mean_norm_W,
    #           (mean_singularValues_R + bar_singularValues_R)/mean_norm_W,
    #           color=deep[2], alpha=0.2)
    ax.scatter(indices, singularValues_EW / mean_norm_W,
               label="$\\sigma_i(\\langle W \\rangle)\,/"
                     "\,\\langle\,||W||_2\,\\rangle$",
               s=5, color=deep[1], zorder=0)

    ax.errorbar(x=indices, y=mean_singularValues_R / mean_norm_W,
                yerr=bar_singularValues_R / mean_norm_W, fmt="o",
                color=deep[9],
                zorder=30, markersize=1.5, capsize=1, elinewidth=1,
                label="$\\langle\\sigma_i(R)\\rangle\,/"
                      "\,\\langle\,||W||_2\,\\rangle$")
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [1, 2, 0]
    # plt.legend([handles[idx]for idx in order],[labels[idx] for idx in order],
    #            loc=4, bbox_to_anchor=(0, 0.1, 1, 1))
    plt.legend(loc=4, bbox_to_anchor=(0, 0.1, 1, 1))
    ticks = ax.get_xticks()
    ticks[ticks.tolist().index(0)] = 1
    ax.set_xticks(ticks[ticks > 0])
    plt.tick_params(axis='both', which='major')
    # plt.ylim([0.01*np.min(singularValues), 1.5])
    # ax1.set_yscale('log')
    # plt.tick_params(axis='y', which='both', left=True,
    #                 right=False, labelbottom=False)
    ax.set_xlabel("Index $i$")
    # plt.ylabel(ylabel)  # , labelpad=20)
    ax.set_xlim([-50, N + 50])

    axins = inset_axes(ax, width="50%", height="50%",
                       bbox_to_anchor=(-0.1, -0.05, 1, 1),
                       bbox_transform=ax.transAxes, loc=1)
    axins.plot(indices, np.mean(norm_R) * np.ones(N) / mean_norm_W,
               linewidth=1, linestyle="--", color=deep[9])
    axins.text(0.92, 0.97, "$\\frac{\\langle\,||R||_2\,\\rangle}"
                           "{\\langle\,||W||_2\,\\rangle}$", fontweight="bold",
               horizontalalignment="center", verticalalignment="top",
               transform=ax.transAxes, fontsize=8)
    axins.scatter(indices,
                  np.mean(
                      np.abs(singularValues - np.outer(np.ones(nb_networks),
                                                       singularValues_EW)),
                      axis=0) / mean_norm_W,
                  color=deep[7], s=2)
    # axins.errorbar(x=indices,
    # y=np.mean(np.abs(singularValues-singularValues_EW),
    #                                     axis=0)/mean_norm_W,
    #                yerr=np.std(np.abs(singularValues-singularValues_EW),
    #                            axis=0)/mean_norm_W, fmt="o", color=deep[7],
    #                markersize=1, capsize=0, elinewidth=1)
    axins.set_ylabel("$\\langle\,\,|\,\\sigma_i(W)"
                     " - \\sigma_i(\\langle W\\rangle)\,|\,\,\\rangle"
                     "\,\,/\,\,\\langle\,||W||_2\,\\rangle$",
                     fontsize=6)
    axins.set_xlabel("Index $i$", fontsize=6)
    ticks = axins.get_xticks()
    ticks[ticks.tolist().index(0)] = 1
    axins.set_xticks(ticks[ticks > 0])
    axins.set_xlim([-50, N + 50])

    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(0.5)
    axins.tick_params(axis='both', which='major', labelsize=8,
                      width=0.5, length=2)


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
              f"/{spec_path_str}_norm_W_{graph_str}.json") as json_data:
        norm_W = np.array(json.load(json_data))

    # norm_ratio = np.mean(norm_R, axis=0) / norm_EW

    norm_ratio = np.mean(norm_R, axis=0) / np.mean(norm_W, axis=0)

    xlabel = "$\\langle ||R||_2 \\rangle\,/\,\\langle ||W||_2 \\rangle$"
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

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) =\
    plt.subplots(nrows=3, ncols=4, figsize=(12, 8))    # figsize=(10, 8)

plot_singular_values_random_graphs(ax1, graph_str_scm,
                                   spec_path_str_scm_svdvals_1)
ax1.set_title("Soft configuration", fontsize=fontsize_title,
              pad=title_pad)
ax1.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)

plot_singular_values_random_graphs(ax2, graph_str_SBM,
                                   spec_path_str_SBM_svdvals_1)
ax2.set_title("Stochastic block", fontsize=fontsize_title,
              pad=title_pad)
ax2.text(letter_posx, letter_posy, "c", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)

plot_singular_values_random_graphs(ax3, graph_str_S1,
                                   spec_path_str_S1_svdvals_1)
ax3.set_title("$S^1$ random geometric", fontsize=fontsize_title,
              pad=title_pad)
ax3.text(letter_posx, letter_posy, "d", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax3.transAxes)

plot_singular_values_random_graphs(ax4, graph_str_gauss,
                                   spec_path_str_gauss_svdvals_1)
ax4.set_title(" Rank-perturbed Gaussian", fontsize=fontsize_title,
              pad=title_pad)
ax4.text(letter_posx, letter_posy, "e", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax4.transAxes)

plot_singular_values_random_graphs(ax5, graph_str_scm,
                                   spec_path_str_scm_svdvals_2)
ax5.text(letter_posx, letter_posy, "f", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax5.transAxes)

plot_singular_values_random_graphs(ax6, graph_str_SBM,
                                   spec_path_str_SBM_svdvals_2)
ax6.text(letter_posx, letter_posy, "g", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax6.transAxes)

plot_singular_values_random_graphs(ax7, graph_str_S1,
                                   spec_path_str_S1_svdvals_2)
ax7.text(letter_posx, letter_posy, "h", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax7.transAxes)

plot_singular_values_random_graphs(ax8, graph_str_gauss,
                                   spec_path_str_gauss_svdvals_2)
ax8.text(letter_posx, letter_posy, "i", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax8.transAxes)


plot_effective_ranks_random_graphs(ax9, graph_str_scm, spec_path_str_scm,
                                   effrank_indicator_list_scm)
ax9.text(letter_posx, letter_posy, "k", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax9.transAxes)
# ax9.set_yticks([0, 0.01, 0.02])
# ax9.set_ylim([0, 0.025])

plot_effective_ranks_random_graphs(ax10, graph_str_SBM, spec_path_str_SBM,
                                   effrank_indicator_list_SBM)
ax10.text(letter_posx, letter_posy, "l", fontweight="bold",
          horizontalalignment="center", verticalalignment="top",
          transform=ax10.transAxes)

plot_effective_ranks_random_graphs(ax11, graph_str_S1, spec_path_str_S1,
                                   effrank_indicator_list_S1)
ax11.tick_params(axis='both', which='major')
ax11.text(letter_posx, letter_posy, "m", fontweight="bold",
          horizontalalignment="center", verticalalignment="top",
          transform=ax11.transAxes)
ax11.set_yticks([0, 0.1, 0.2])

plot_effective_ranks_random_graphs(ax12, graph_str_gauss, spec_path_str_gauss,
                                   effrank_indicator_list_gauss)
ax12.text(letter_posx, letter_posy, "n", fontweight="bold",
          horizontalalignment="center", verticalalignment="top",
          transform=ax12.transAxes)
ax12.set_yticks([0, 0.01, 0.02])

plt.show()
