# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_networks import *
import json
from plots.config_rcparams import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from singular_values.compute_singular_values_dscm_upper_bounds import\
    upper_bound_singvals_infinite_sum_weighted
from tqdm import tqdm

path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "singular_values/properties/"

""" Perturbed gaussian """
graph_str_gauss = "perturbed_gaussian"
spec_path_str_gauss = "2023_03_02_13h53min20sec_100graphs_30s"
spec_path_str_gauss_svdvals_1 = "2023_03_02_22h59min37sec_100graphs_g1_r0_08"
spec_path_str_gauss_svdvals_2 = "2023_03_02_23h00min31sec_100graphs_g3_r0_25"
effrank_indicator_list_gauss = [1, 1, 1, 0, 0, 0, 0, 0]

""" Degree-corrected stochastic block model """
graph_str_SBM = "sbm"
spec_path_str_SBM_svdvals_1 = "2023_03_02_23h17min27sec_100graphs_g100_r0_091"
spec_path_str_SBM_svdvals_2 = "2023_03_02_23h18min38sec_100graphs_g10_r0_285"
spec_path_str_SBM = "2023_03_02_13h52min15sec_100graphs_30s"
effrank_indicator_list_SBM = [1, 1, 1, 0, 0, 0, 0, 0]

""" Soft configuration model """
graph_str_scm = "soft_configuration_model"
spec_path_str_scm_svdvals_1 = "2023_03_04_15h05min39sec_100graphs_g0_6_r0_096"
spec_path_str_scm_svdvals_2 = "2023_03_04_15h06min27sec_100graphs_g0_15_r0_268"
spec_path_str_scm = "2023_03_02_13h53min07sec_100graphs_30s"
effrank_indicator_list_scm = [1, 1, 1, 0, 0, 0, 0, 0]

""" S1 """
graph_str_S1 = "s1"
spec_path_str_S1_svdvals_1 = "2023_03_02_23h11min06sec_100graphs_g0_2_r0_0657"
spec_path_str_S1_svdvals_2 = "2023_03_02_23h11min57sec_100graphs_g0_95_r0_25"
spec_path_str_S1 = "2023_03_02_22h58min15sec_100graphs_30s"
effrank_indicator_list_S1 = [1, 1, 1, 0, 0, 0, 0, 0]

singvals_color1 = "#c6dbef"
singvals_color2 = deep[9]  # "#6baed6"
singvals_color3 = "#08519c"


def plot_singular_values_random_graphs(ax, graph_str, spec_path_str, ins_ylim,
                                       ins_yticks, R_posy, markerstyle,
                                       max_scatter, plot_upper_bound=True):
    """ max_scatter is the value at which we plot the singular values
        with solid lines instead of scattering them, to reduce the size
        of the pdf
    """

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

    max_singvals = 200
    if graph_str == "soft_configuration_model" and plot_upper_bound:
        max_singvals = 23
        max_scatter = 23

    singularValues_trunc = singularValues[:, :max_singvals]
    singularValues_EW_trunc = singularValues_EW[:max_singvals]
    singularValues_R_trunc = singularValues_R[:, :max_singvals]

    mean_singularValues_trunc = np.mean(singularValues_trunc, axis=0)
    bar_singularValues_trunc = np.std(singularValues_trunc, axis=0)

    mean_singularValues_R_trunc = np.mean(singularValues_R_trunc, axis=0)
    bar_singularValues_R_trunc = np.std(singularValues_R_trunc, axis=0)

    indices = np.arange(1, max_singvals + 1, 1)

    ax.errorbar(x=indices[:max_scatter],
                y=mean_singularValues_trunc[:max_scatter] / mean_norm_W,
                yerr=bar_singularValues_trunc[:max_scatter] / mean_norm_W,
                fmt="o", color=singvals_color3,
                zorder=-30, markersize=5, capsize=1, elinewidth=1,
                label="$\\overline{\\sigma_i(W)}$")
    ax.errorbar(x=indices[max_scatter:],
                y=mean_singularValues_trunc[max_scatter:]/mean_norm_W,
                yerr=bar_singularValues_trunc[max_scatter:] / mean_norm_W,
                color=singvals_color3,
                zorder=-30, capsize=1, elinewidth=1, linewidth=6.5)

    ax.scatter(indices[:max_scatter],
               singularValues_EW_trunc[:max_scatter] / mean_norm_W,
               label="$\\overline{\\sigma_i(\\langle W \\rangle)}$",
               s=5, color=singvals_color2, zorder=30)
    ax.plot(indices[max_scatter:],
            singularValues_EW_trunc[max_scatter:] / mean_norm_W,
            linewidth=3.5, color=singvals_color2, zorder=30)

    ax.errorbar(x=indices[:max_scatter],
                y=mean_singularValues_R_trunc[:max_scatter] / mean_norm_W,
                yerr=bar_singularValues_R_trunc[:max_scatter] / mean_norm_W,
                fmt="o", color=singvals_color1, zorder=0, markersize=1,
                capsize=1, elinewidth=1, label="$\\overline{\\sigma_i(R)}$")
    ax.errorbar(x=indices[max_scatter:],
                y=mean_singularValues_R_trunc[max_scatter:] / mean_norm_W,
                yerr=bar_singularValues_R_trunc[max_scatter:] / mean_norm_W,
                color=singvals_color1, zorder=0, capsize=1, elinewidth=1,
                linewidth=2)

    if graph_str == "soft_configuration_model" and plot_upper_bound:
        y = parameters_dictionary["y"]
        z = parameters_dictionary["z"]
        upper_bound_singvals = np.zeros(N)
        for i in tqdm(indices):
            upper_bound_singvals[i - 1] = \
                upper_bound_singvals_infinite_sum_weighted(i, y, z, 1e-12)
        # ax.scatter(indices, upper_bound_singvals[:max_singvals]/mean_norm_W,
        #            color=dark_grey, s=1, zorder=50,
        #            label="Theoretical\nupper\nbound")
        if np.mean(norm_R)/mean_norm_W < 0.2:
            ax.text(2.9, 0.2,
                    "$\\rightarrow\,\,$Bound on"
                    " $\\overline{\\sigma_i(\\langle W \\rangle)}$",  # \n"
                    # "$\,\,\\qquad$(Theorem 1)",
                    fontsize=8)
        ax.plot(indices, upper_bound_singvals[:max_singvals] / mean_norm_W,
                color=dark_grey, zorder=50, linestyle="--", linewidth=1)

    ticks = ax.get_xticks()
    ticks[ticks.tolist().index(0)] = 1
    ticks = ticks[ticks > 0].tolist()
    if graph_str == "soft_configuration_model" and plot_upper_bound:
        ticks.remove(10)
        ticks.remove(30)
        # ticks.append(30)
        ax.set_xlim([0.1, max_singvals + 0.3])
    else:
        ticks.remove(100)
        ticks.remove(300)
        ax.set_xlim([-5, max_singvals + 5])
    ax.set_xticks(ticks)
    ax.tick_params(axis='both', which='major')
    # plt.ylim([0.01*np.min(singularValues), 1.5])
    # ax1.set_yscale('log')
    # plt.tick_params(axis='y', which='both', left=True,
    #                 right=False, labelbottom=False)
    ax.set_xlabel("Index $i$")
    ax.xaxis.set_label_coords(0.5, -0.08)
    # plt.ylabel(ylabel)  # , labelpad=20)
    # ax.set_xlim([-50, N + 50])
    ax.set_ylim([-0.05, 1.05])
    ax.set_yticks([0, 0.5, 1])

    axins = inset_axes(ax, width="45%", height="45%",
                       bbox_to_anchor=(-0.1, -0.05, 1, 1.05),
                       bbox_transform=ax.transAxes, loc=1)
    hat_R = np.mean(norm_R)/mean_norm_W
    print(hat_R)
    # indices = np.arange(1, N + 1, 1)
    axins.plot(indices, hat_R*np.ones(max_singvals),
               linewidth=1, linestyle="--", color=singvals_color1)
    axins.text(1.2, R_posy+0.1, "$\\overline{||R||}_2$", fontweight="bold",
               horizontalalignment="center", verticalalignment="top",
               transform=axins.transAxes, fontsize=8)
    if graph_str == "soft_configuration_model" and plot_upper_bound:
        axins.scatter(22, hat_R, marker=markerstyle,
                      color=singvals_color1, s=20)
    else:
        axins.scatter(190, hat_R, marker=markerstyle,
                      color=singvals_color1, s=20)
    Delta = singularValues_trunc - np.outer(np.ones(nb_networks),
                                            singularValues_EW_trunc)
    barDelta = np.mean(np.abs(Delta), axis=0) / mean_norm_W
    axins.scatter(indices[:max_scatter+10], barDelta[:max_scatter+10],
                  edgecolors='none', color=deep[7], s=4)
    axins.plot(indices[max_scatter+10:], barDelta[max_scatter+10:],
               color=deep[7], linewidth=2)

    ylab = axins.set_ylabel("$\\bar{\,\,\\Delta}_{\,i}$", fontsize=8)
    axins.yaxis.set_label_coords(-0.24, 0.4)
    ylab.set_rotation(0)
    axins.set_xlabel("$i$", fontsize=8)
    axins.xaxis.set_label_coords(0.5, -0.1)
    if graph_str == "soft_configuration_model" and plot_upper_bound:
        axins.set_xlim([0.1, max_singvals + 0.9])
    else:
        axins.set_xlim([-5, max_singvals + 5])
    ticks = axins.get_xticks()
    ticks[ticks.tolist().index(0)] = 1
    ticks_list = ticks[ticks > 0].tolist()
    if graph_str == "soft_configuration_model" and plot_upper_bound:
        ticks_list.remove(10)
        ticks_list.remove(30)
        # ticks_list.remove(40)
        # ticks_list.append(30)
    else:
        ticks_list.remove(100)
        ticks_list.remove(300)
    axins.set_xticks(ticks_list)
    # axins.set_xlim([-50, N + 50])
    axins.set_ylim(ins_ylim)
    axins.set_yticks(ins_yticks)

    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(0.5)
    axins.tick_params(axis='both', which='major', labelsize=8,
                      width=0.5, length=2)


def plot_effective_ranks_random_graphs(ax, graph_str, spec_path_str,
                                       effrank_indicator_list):
    effrank_str_list = ["srank", "nrank", "elbow", "energy", "thrank",
                        "shrank", "erank", "rank"]
    # color = ["#fc8d62", "#8da0cb", "#a6d854", "#ffff99"]
    color = ["#fdbf6f", "#b2df8a", "#cab2d6", "#fb9a99", "#a6cee3"]
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

    norm_ratio = np.mean(norm_R, axis=0) / np.mean(norm_W, axis=0)

    xlabel = "$\\overline{||R||}_2$"
    # ""$\\langle ||R||_2 \\rangle\,/\,\\langle ||W||_2 \\rangle$"
    alpha = 0.2
    # s = 3
    j = 0
    for i in range(len(effrank_str_list)):
        if effrank_indicator_list[i]:
            with open(path_str + graph_str +
                      f"/{spec_path_str}_{effrank_str_list[i]}"
                      f"_{graph_str}.json") as json_data:
                effrank = np.array(json.load(json_data))
            mean_effrank = np.mean(effrank, axis=0) / N
            std_effrank = np.std(effrank, axis=0) / N
            # ax.scatter(norm_ratio, mean_effrank, color=deep[j+1], s=s,
            #            label=effrank_str_list[i] + "/N")
            ax.plot(norm_ratio, mean_effrank, color=color[j], linewidth=1.5,
                    label="$\\langle$" + effrank_str_list[i]
                          + "$\\rangle$" + "/N")
            ax.fill_between(norm_ratio, mean_effrank - std_effrank,
                            mean_effrank + std_effrank, color=color[j],
                            alpha=alpha)
            j += 1
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel(xlabel)
    ax.set_ylim([-0.006, 0.15])
    ax.set_yticks([0, 0.1])


""" Plot """

letter_posx, letter_posy = -0.25, 1.08
fontsize_legend = 8
fontsize_title = 9.5
title_pad = 0
markerpos = -0.006
max_scatter = 8


fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) =\
    plt.subplots(nrows=3, ncols=4, figsize=(10, 6.5))    # figsize=(10, 8)

plot_singular_values_random_graphs(ax1, graph_str_gauss,
                                   spec_path_str_gauss_svdvals_1, [0, 0.1],
                                   [0, 0.1], 0.88, "s", max_scatter)
# ax1.set_title(" Rank-perturbed Gaussian", fontsize=fontsize_title,
#               pad=title_pad)
ax1.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)
ax1.legend(loc=2)
handles, labels = ax1.get_legend_handles_labels()
# order = [2, 0, 3, 1]
order = [1, 0, 2]
u = ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               fontsize=8, bbox_to_anchor=(1.1, 0.155), loc=4,
               handletextpad=-0.04, ncol=3, columnspacing=0.06, frameon=True,
               borderpad=0.1)
frame = u.get_frame()
# frame.set_edgecolor("#4d4d4d")
frame.set_color("#f0f0f0")


plot_singular_values_random_graphs(ax2, graph_str_SBM,
                                   spec_path_str_SBM_svdvals_1, [0, 0.1],
                                   [0, 0.1], 0.93, "s", max_scatter)
# ax2.set_title("Stochastic block", fontsize=fontsize_title,
#               pad=title_pad)
ax2.text(letter_posx, letter_posy, "c", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)

plot_singular_values_random_graphs(ax3, graph_str_S1,
                                   spec_path_str_S1_svdvals_1, [0, 0.1],
                                   [0, 0.1], 0.7, "s", 52)
# ax3.set_title("$S^1$ random geometric", fontsize=fontsize_title,
#              pad=title_pad)
ax3.text(letter_posx, letter_posy, "d", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax3.transAxes)


plot_singular_values_random_graphs(ax4, graph_str_scm,
                                   spec_path_str_scm_svdvals_1, [0, 0.1],
                                   [0, 0.1], 0.97, "s", max_scatter)
# ax4.set_title("Soft configuration", fontsize=fontsize_title,
#               pad=title_pad)
ax4.text(letter_posx, letter_posy, "e", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax4.transAxes)

plot_singular_values_random_graphs(ax5, graph_str_gauss,
                                   spec_path_str_gauss_svdvals_2, [0, 0.3],
                                   [0, 0.3], 0.85, "*", max_scatter)
ax5.text(letter_posx, letter_posy, "f", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax5.transAxes)

plot_singular_values_random_graphs(ax6, graph_str_SBM,
                                   spec_path_str_SBM_svdvals_2, [0, 0.3],
                                   [0, 0.3], 0.97, "*", max_scatter)
ax6.text(letter_posx, letter_posy, "g", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax6.transAxes)


plot_singular_values_random_graphs(ax7, graph_str_S1,
                                   spec_path_str_S1_svdvals_2, [0, 0.3],
                                   [0, 0.3], 0.87, "*", 52)
ax7.text(letter_posx, letter_posy, "h", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax7.transAxes)


plot_singular_values_random_graphs(ax8, graph_str_scm,
                                   spec_path_str_scm_svdvals_2, [0, 0.3],
                                   [0, 0.3], 0.92, "*", max_scatter)
ax8.text(letter_posx, letter_posy, "i", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax8.transAxes)


plot_effective_ranks_random_graphs(ax9, graph_str_gauss, spec_path_str_gauss,
                                   effrank_indicator_list_gauss)
ax9.scatter([0.0846], [markerpos], color=singvals_color1, zorder=10,
            clip_on=False, marker="s")
ax9.scatter([0.250], [markerpos], color=singvals_color1, zorder=10,
            clip_on=False, marker="*")
ax9.text(letter_posx, letter_posy, "k", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax9.transAxes)
# ax9.set_yticks([0, 0.01, 0.02])
# ax9.set_ylim([0, 0.025])
# ax9.set_yticks([0, 0.01, 0.02])
ax9.set_xticks([0, 0.1, 0.2, 0.3])
# ax9.set_yticks([0, 0.1, 0.2])
# ax9.set_ylim([-0.02, 0.15])
w = ax9.legend(loc=2, fontsize=8, handletextpad=0.7,
               bbox_to_anchor=(0, 1.05), handlelength=0.8, frameon=True)
frame2 = w.get_frame()
frame2.set_color("#f0f0f0")

plot_effective_ranks_random_graphs(ax10, graph_str_SBM, spec_path_str_SBM,
                                   effrank_indicator_list_SBM)
ax10.scatter([0.0908], [markerpos], color=singvals_color1, zorder=10,
             clip_on=False, marker="s")
ax10.scatter([0.285], [markerpos], color=singvals_color1, zorder=10,
             clip_on=False, marker="*")
ax10.text(letter_posx, letter_posy, "l", fontweight="bold",
          horizontalalignment="center", verticalalignment="top",
          transform=ax10.transAxes)
ax10.set_xticks([0.1, 0.2, 0.3])
# ax10.set_yticks([0, 0.1, 0.2])
# ax10.set_ylim([0, 0.2])


plot_effective_ranks_random_graphs(ax11, graph_str_S1, spec_path_str_S1,
                                   effrank_indicator_list_S1)
ax11.scatter([0.0657], [markerpos], color=singvals_color1, zorder=10,
             clip_on=False, marker="s")
ax11.scatter([0.2504], [markerpos], color=singvals_color1, zorder=10,
             clip_on=False, marker="*")
ax11.set_xticks([0, 0.1, 0.2, 0.3])
# ax12.set_yticks([0, 0.1, 0.2])
ax11.set_xlim([-0.006, 0.306])
ax11.text(letter_posx, letter_posy, "m", fontweight="bold",
          horizontalalignment="center", verticalalignment="top",
          transform=ax11.transAxes)


plot_effective_ranks_random_graphs(ax12, graph_str_scm, spec_path_str_scm,
                                   effrank_indicator_list_scm)
ax12.scatter([0.0964], [markerpos], color=singvals_color1, zorder=10,
             clip_on=False, marker="s")
ax12.scatter([0.268], [markerpos], color=singvals_color1, zorder=10,
             clip_on=False, marker="*")
ax12.set_xticks([0.1, 0.2, 0.3])
# ax12.set_yticks([0, 0.1, 0.2])
# ax12.set_ylim([0, 0.2])
ax12.text(letter_posx, letter_posy, "n", fontweight="bold",
          horizontalalignment="center", verticalalignment="top",
          transform=ax12.transAxes)

plt.tight_layout()

plt.show()
