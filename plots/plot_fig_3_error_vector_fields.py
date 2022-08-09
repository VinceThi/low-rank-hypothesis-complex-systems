# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_networks import *
from plots.config_rcparams import *
from scipy.linalg import svdvals
from singular_values.optimal_shrinkage import optimal_shrinkage
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable

path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "simulations/simulations_data/"

""" QMF SIS """
path_error_sis = "2022_03_29_11h27min22sec_1000_samples_randomD_RMSE" \
                 "_vector_field_qmf_sis_high_school_proximity.json"
path_upper_bound_sis = "2022_03_29_11h27min22sec_1000_samples_randomD" \
                       "_upper_bound_RMSE_vector_field_qmf_sis_high_school" \
                       "_proximity.json"
graph_str = "high_school_proximity"
A_sis = get_epidemiological_weight_matrix(graph_str)
S_sis = svdvals(A_sis)
N_arange_sis = np.arange(1, len(A_sis[0])+1, 1)


""" Microbial """
path_error_microbial = "2022_07_31_10h04min31sec_1000_samples_max_x_Px" \
                       "_RMSE_vector_field_microbial_gut.json"
path_upper_bound_microbial = "2022_07_31_10h04min31sec_1000_samples_max_x_Px_"\
                             "upper_bound_RMSE_vector_field_microbial_gut.json"
graph_str = "gut"
A_microbial = get_microbiome_weight_matrix(graph_str)
S_microbial = svdvals(A_microbial)
N_arange_microbial = np.arange(1, len(A_microbial[0])+1, 1)


""" RNN """
path_error_rnn = "2022_07_12_15h51min54sec_1000_samples_frobenius_shrinkage" \
                 "_RMSE_vector_field_rnn_mouse_control_rnn.json"
path_upper_bound_rnn = "2022_07_12_15h51min54sec_1000_samples_frobenius" \
                       "_shrinkage_upper_bound_RMSE_vector_field_rnn" \
                       "_mouse_control_rnn.json"
graph_str = "mouse_control_rnn"
A_rnn = get_learned_weight_matrix(graph_str)
N = len(A_rnn[0])  # Dimension of the complete dynamics  669
U, S, Vh = np.linalg.svd(A_rnn)
shrink_s = optimal_shrinkage(S, 1, 'frobenius')
A_rnn = U@np.diag(shrink_s)@Vh
N_arange_rnn = np.arange(1, len(A_rnn[0])+1, 1)


""" Wilson-Cowan """
path_error_wc = "2022_03_25_14h45min47sec_1000_samples_RMSE_vector_field" \
                "_wilson_cowan_celegans_signed.json"
path_upper_bound_wc = "2022_03_25_14h45min47sec_1000_samples_upper_bound" \
                      "_RMSE_vector_field_wilson_cowan_celegans_signed.json"
graph_str = "celegans_signed"
A_wc = get_connectome_weight_matrix(graph_str)
S_wc = svdvals(A_wc)
N_arange_wc = np.arange(1, len(A_wc[0])+1, 1)


# Global parameters
ymin = 0.5*10**(-5)
ymax = 500


def plot_singvals(ax, singularValues, ylabel_bool=False):
    """ Warning: For the sake of vizualisation, we set the 0 values to the
        minimum value of the log scale and add a 0 value on the y-axis. Symlog
        is supposed to deal with this problem but it didn't give the expected
        result in the log region (even by setting linthresh correctly). """

    normalized_singularValues = singularValues / singularValues[0]
    """ For the sake of visualization, we set the zeros to ymin 
            (symlog did not give the desired result) """
    normalized_singularValues[normalized_singularValues < 1e-13] = ymin
    # ax.plot(np.arange(1, len(singularValues) + 1, 1),
    #         normalized_singularValues,
    #         color=first_community_color,
    #         label="Normalized singular values $\\sigma_n/\\sigma_1$")
    ax.scatter(np.arange(1, len(singularValues) + 1, 1),
               normalized_singularValues, s=1,
               color=first_community_color, zorder=10)
    if ylabel_bool:
        plt.ylabel("$\\frac{\\sigma_n}{\\sigma_1}$", rotation=0, fontsize=16,
                   color=first_community_color)
        ax.yaxis.labelpad = 20


def plot_error(ax, dynamics_str, path_error, path_upper_bound, N_arange,
               xlabel_bool=True):

    """ Warning: For the sake of vizualisation, we set the 0 values to the
    minimum value of the log scale and add a 0 value on the y-axis. Symlog
    is supposed to deal with this problem but it didn't give the expected
    result in the log region (even by setting linthresh correctly). """

    with open(path_str + f"{dynamics_str}_data/vector_field_errors/" +
              path_error) as json_data:
        error_array = json.load(json_data)

    with open(path_str+f"{dynamics_str}_data/vector_field_errors/" +
              path_upper_bound) as json_data:
        error_upper_bound_array = json.load(json_data)

    """ In the folowing 'if' we add the error 0 at n = N to the error and
    the upper bound array of rnn, wilson_cowan, and microbial. """
    if dynamics_str == "rnn":
        nb_samples, nb_sing = np.shape(error_array)
        error_array = \
            np.hstack((error_array, np.zeros((nb_samples,
                                              len(N_arange_rnn) - nb_sing))))
        error_upper_bound_array = \
            np.hstack((error_upper_bound_array,
                       np.zeros((nb_samples, len(N_arange_rnn) - nb_sing))))

    if dynamics_str == "wilson_cowan":
        nb_samples, nb_sing = np.shape(error_array)
        error_array = \
            np.hstack((error_array, np.zeros((nb_samples,
                                              len(N_arange_wc) - nb_sing))))
        error_upper_bound_array = \
            np.hstack((error_upper_bound_array,
                       np.zeros((nb_samples, len(N_arange_wc) - nb_sing))))

    if dynamics_str == "microbial":
        nb_samples, nb_sing = np.shape(error_array)
        error_array = \
            np.hstack((error_array, np.zeros((nb_samples,
                                              len(N_arange_microbial)
                                              - nb_sing))))
        error_upper_bound_array = \
            np.hstack((error_upper_bound_array,
                       np.zeros((nb_samples, len(N_arange_microbial)
                                 - nb_sing))))

    mean_error = np.mean(error_array, axis=0)
    mean_log10_error = np.log10(mean_error)
    relative_std_semilogplot_error = \
        np.std(error_array, axis=0) / (np.log(10) * mean_error)
    fill_between_error_1 = 10**(
            mean_log10_error - relative_std_semilogplot_error)
    fill_between_error_2 = 10**(
            mean_log10_error + relative_std_semilogplot_error)

    mean_upper_bound_error = np.mean(error_upper_bound_array, axis=0)

    mean_log10_upper_bound_error = np.log10(mean_upper_bound_error)
    relative_std_semilogplot_upper_bound_error = \
        np.std(error_upper_bound_array, axis=0) / \
        (np.log(10) * mean_upper_bound_error)
    fill_between_ub1 = 10**(mean_log10_upper_bound_error -
                            relative_std_semilogplot_upper_bound_error)
    fill_between_ub2 = 10**(mean_log10_upper_bound_error +
                            relative_std_semilogplot_upper_bound_error)

    """ For the sake of visualization, we set the zeros to ymin 
        (symlog did not give the desired result) """
    mean_error[mean_error < 1e-13] = ymin
    mean_upper_bound_error[mean_upper_bound_error < 1e-13] = ymin
    fill_between_error_1[fill_between_error_1 < 1e-13] = ymin
    fill_between_error_2[fill_between_error_2 < 1e-13] = ymin
    fill_between_ub1[fill_between_ub1 < 1e-13] = ymin
    fill_between_ub2[fill_between_error_2 < 1e-13] = ymin

    ax.scatter(N_arange, mean_error, s=5, color=deep[3],
               label="Average alignment error"
                     " $\\langle \\mathcal{E} \\rangle_x$", zorder=0)
    ax.scatter(N_arange, mean_upper_bound_error, s=2, color=dark_grey,
               label="Average upper-bound"
                     " on $\\langle \\mathcal{E} \\rangle_x$")
    ax.fill_between(N_arange, fill_between_error_1, fill_between_error_2,
                    color=deep[3], alpha=0.5)
    ax.fill_between(N_arange, fill_between_ub1, fill_between_ub2,
                    color=dark_grey, alpha=0.5)

    # if ylabel_bool:
    #    ybox1 = TextArea("$\\langle \\mathcal{E}_f \\rangle_x$ ",
    #                     textprops=dict(color=deep[3], size=15, rotation=90,
    #                                    ha='left', va='bottom'))
    #    ybox2 = TextArea("$\\leq$ ",
    #                     textprops=dict(color="#3f3f3f", size=15, rotation=90,
    #                                    ha='left', va='bottom'))
    #    ybox3 = TextArea("$\\langle \\mathcal{E}_{\\mathrm{ub}} \\rangle_x$",
    #                     textprops=dict(color="#3f3f3f", size=15, rotation=90,
    #                                    ha='left', va='bottom'))
    #    ybox4 = TextArea(" ",
    #                     textprops=dict(color="#3f3f3f", size=15, rotation=90,
    #                                    ha='left', va='bottom'))
    #    ybox5 = TextArea("$\\sigma_n/\\sigma_1$",
    #                     textprops=dict(color=first_community_color, size=13,
    #                                    rotation=90,
    #                                    ha='left', va='bottom'))
    #
    #    ybox = VPacker(children=[ybox5, ybox4, ybox3, ybox2, ybox1],
    #                   align="bottom", pad=0, sep=2)
    #
    #    anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0.,
    #                                      frameon=False,
    #                                      bbox_to_anchor=(-0.35, -0.05),
    #                                      bbox_transform=ax.transAxes,
    #                                      borderpad=0.)
    #
    #    ax.add_artist(anchored_ybox)
    #    # plt.ylabel("$\\langle \\mathcal{E}_f \\rangle_x \leq $",
    #    #  color=deep[3])
    #    # rotation=0,  # fontsize=16,
    #    ax.yaxis.labelpad = 20

    if dynamics_str is not "rnn":
        if xlabel_bool:
            plt.xlabel('Dimension $n$')
    #     ticks = ax.get_xticks()
    #     print(ticks)
    #     ticks[ticks.tolist().index(0)] = 1
    #     ticks = [i for i in ticks
    #              if -0.1*len(N_arange) < i < 1.1*len(N_arange)]
    #     print(ticks)
    #     plt.xticks(ticks)
    if dynamics_str == "qmf_sis" or dynamics_str == "wilson_cowan":
        ticks = ax.get_xticks().tolist()
        ticks[ticks.index(0)] = 1
        ticks += [100, 200, 300]
        # ticks = [i for i in ticks
        #          if -0.1*len(N_arange) < i < 1.1*len(N_arange)]
        plt.xticks(ticks)
        plt.text(-len(mean_error) / 5.8, 0.0000035, "0 -")

    elif dynamics_str == "microbial":
        ticks = ax.get_xticks().tolist()
        ticks[ticks.index(0)] = 1
        ticks += [250, 500, 750]
        # ticks = [i for i in ticks
        #          if -0.1*len(N_arange) < i < 1.1*len(N_arange)]
        plt.xticks(ticks)
        plt.text(-len(mean_error) / 5.6, 0.0000035, "0 -")

    else:
        plt.text(-len(mean_error) / 5.8, 0.0000035, "0 -")

    ax.set_ylim([0.75*ymin, ymax])
    ax.set_yscale('log', nonposy="clip")
    plt.tick_params(axis='y', which='both', left=True,
                    right=False, labelbottom=False)
    plt.minorticks_off()


""" -----------------------  Figure 3 ------------------------------------- """
# fig = plt.figure(figsize=(6, 5.5))  # 5, 4.5))
fig = plt.figure(figsize=(11, 3))

title_pad = -12
letter_posx, letter_posy = -0.27, 1.08

# --------------------------- SIS ---------------------------------------------
ax1 = plt.subplot(141)
ax1.set_title("Epidemiological", fontsize=fontsize_legend,
              pad=title_pad)
plot_error(ax1, "qmf_sis", path_error_sis, path_upper_bound_sis, N_arange_sis)
plot_singvals(ax1, S_sis)
ax1.set_xlabel('Dimension $n$')
ax1.text(letter_posx, letter_posy, "a", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)


# ----------------------- Wilson-Cowan-----------------------------------------
ax2 = plt.subplot(142)
ax2.set_title("Neuronal", fontsize=fontsize_legend,
              pad=title_pad)
plot_error(ax2, "wilson_cowan", path_error_wc,
           path_upper_bound_wc, N_arange_wc)
plot_singvals(ax2, S_wc)
ax2.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)


# --------------------------- RNN ---------------------------------------------
ax3 = plt.subplot(143)
divider = make_axes_locatable(ax3)
ax32 = divider.new_horizontal(size="100%", pad=0.1)
fig.add_axes(ax32)
plot_error(ax3, "rnn", path_error_rnn,
           path_upper_bound_rnn, N_arange_rnn)
plot_singvals(ax3, shrink_s)
ax3.set_xlim(-10, 120)
plot_error(ax32, "rnn", path_error_rnn,
           path_upper_bound_rnn, N_arange_rnn)
plot_singvals(ax32, shrink_s)
ax3.text(letter_posx-0.2, letter_posy, "c", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax3.transAxes)
ax32.set_xlim(630, 670)
ax32.tick_params(left=False, labelleft=False)
ax32.set_yticks([])
ax32.spines['left'].set_visible(False)
d = .025  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax32.transAxes, color=dark_grey, clip_on=False)
ax32.plot((-d, +d), (-d, +d), zorder=10, linewidth=1.5, **kwargs)
kwargs.update(transform=ax3.transAxes)
ax3.plot((1 - d, 1 + d), (-d, d), zorder=10, linewidth=1.5, **kwargs)
ax32.text(605, 100, "Chaotic RNN", fontsize=12)
ax3.text(-120/3, 0.0000035, "0 -")
ax3.set_xlabel('Dimension $n$')
ax3.xaxis.set_label_coords(1.1, -0.16)
# ticks = ax3.get_xticks()
# ticks[ticks.tolist().index(0)] = 1
# ticks = [i for i in ticks
#          if -0.1*200 < i < 1.1*200]
ax3.set_xticks([1, 100])
ax32.set_xticks([650])
# ax3.set_title("$\\qquad\\qquad\\qquad$Chaotic RNN",
#               fontsize=fontsize_legend, pad=title_pad)


# ------------------------ Microbial ------------------------------------------
ax4 = plt.subplot(144)
ax4.set_title("Microbial", fontsize=fontsize_legend,
              pad=title_pad)
plot_error(ax4, "microbial", path_error_microbial,
           path_upper_bound_microbial, N_arange_microbial)
plot_singvals(ax4, S_microbial)
ax4.text(letter_posx, letter_posy, "d", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax4.transAxes)


# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc=(0.3, 0.95))

plt.tight_layout()
# fig.subplots_adjust(bottom=0, top=100)
# plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
# fig.tight_layout(rect=[0, 0, .8, 1])
# .subplots_adjust(bottom=0.1, right=0.8, top=0.9)

plt.show()
