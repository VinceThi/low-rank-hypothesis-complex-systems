# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_networks import *
from scipy.linalg import svdvals
from dynamics.error_vector_fields import rmse, mse
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from singular_values.optimal_shrinkage import optimal_shrinkage
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plots.config_rcparams import * 
from mpl_toolkits.mplot3d import Axes3D
from math import log10, floor


path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "simulations/simulations_data/"

""" QMF SIS """
path_error_sis = "2022_03_29_11h27min22sec_1000_samples_randomD_RMSE" \
                 "_vector_field_qmf_sis_high_school_proximity.json"
path_upper_bound_sis = "2022_03_29_11h27min22sec_1000_samples_randomD" \
                       "_upper_bound_RMSE_vector_field_qmf_sis_high_school" \
                       "_proximity.json"
path_parameters_sis = "2023_01_14_15h10min58sec_n_1_1000pts_qmf_sis" \
                      "_high_school_proximity_parameters_dictionary.json"
path_complete_bifurcation_sis = \
    "2023_01_14_15h10min58sec_n_1_1000pts_x_equilibrium_points" \
    "_list_complete_qmf_sis_high_school_proximity.json"
path_reduced_bifurcation_n1_sis = \
    "2023_01_14_15h10min58sec_n_1_1000pts_redx_equilibrium_points" \
    "_list_reduced_qmf_sis_high_school_proximity.json"
path_reduced_bifurcation_n7_sis = \
    "2023_01_14_15h28min20sec_n_7_1000pts_redx_equilibrium_points" \
    "_list_reduced_qmf_sis_high_school_proximity.json"
path_reduced_bifurcation_n104_sis = \
    "2023_01_14_15h28min41sec_n_104_1000pts_redx_equilibrium_points" \
    "_list_reduced_qmf_sis_high_school_proximity.json"
graph_str = "high_school_proximity"
A_sis = get_epidemiological_weight_matrix(graph_str)
S_sis = svdvals(A_sis)
N_arange_sis = np.arange(1, len(A_sis[0])+1, 1)


""" Microbial """
path_error_microbial = "2022_07_31_10h04min31sec_1000_samples_max_x_Px" \
                       "_RMSE_vector_field_microbial_gut.json"
path_upper_bound_microbial = "2022_07_31_10h04min31sec_1000_samples_max_x_Px_"\
                             "upper_bound_RMSE_vector_field_microbial_gut.json"
path_parameters_microbial_n76 = "2023_01_13_21h32min04sec_n_76_100ptsx2" \
                                "_microbial_gut_parameters_dictionary.json"
path_parameters_microbial_n203 = "2023_01_13_21h33min47sec_n_203_100ptsx2" \
                                 "_microbial_gut_parameters_dictionary.json"
path_parameters_microbial_n400 = "2023_01_14_10h44min04sec_n_400_100ptsx2_" \
                                 "microbial_gut_parameters_dictionary.json"
path_complete_forward_bifurcation_microbial =\
    "2023_01_13_21h32min04sec_n_76_100ptsx2_x_forward_equilibrium" \
    "_points_list_complete_microbial_gut.json"
path_complete_backward_bifurcation_microbial =\
    "2023_01_13_21h32min04sec_n_76_100ptsx2_x_backward_equilibrium" \
    "_points_list_complete_microbial_gut.json"
path_reduced_forward_bifurcation_n76_microbial =\
    "2023_01_13_21h32min04sec_n_76_100ptsx2_redx_forward_equilibrium" \
    "_points_list_reduced_microbial_gut.json"
path_reduced_backward_bifurcation_n76_microbial = \
    "2023_01_13_21h32min04sec_n_76_100ptsx2_redx_backward_equilibrium" \
    "_points_list_reduced_microbial_gut.json"
path_reduced_forward_bifurcation_n203_microbial =\
    "2023_01_13_21h33min47sec_n_203_100ptsx2_redx_forward_equilibrium" \
    "_points_list_reduced_microbial_gut.json"
path_reduced_backward_bifurcation_n203_microbial =\
    "2023_01_13_21h33min47sec_n_203_100ptsx2_redx_backward_equilibrium" \
    "_points_list_reduced_microbial_gut.json"
path_reduced_forward_bifurcation_n400_microbial =\
    "2023_01_14_10h44min04sec_n_400_100ptsx2_redx_forward_equilibrium" \
    "_points_list_reduced_microbial_gut.json"
path_reduced_backward_bifurcation_n400_microbial = \
    "2023_01_14_10h44min04sec_n_400_100ptsx2_redx_backward_equilibrium" \
    "_points_list_reduced_microbial_gut.json"
graph_str = "gut"
A_microbial = get_microbiome_weight_matrix(graph_str)
S_microbial = svdvals(A_microbial)
N_arange_microbial = np.arange(1, len(A_microbial[0])+1, 1)


""" Wilson-Cowan """
path_error_wc = "2022_03_25_14h45min47sec_1000_samples_RMSE_vector_field" \
                "_wilson_cowan_celegans_signed.json"
path_upper_bound_wc = "2022_03_25_14h45min47sec_1000_samples_upper_bound" \
                      "_RMSE_vector_field_wilson_cowan_celegans_signed.json"
path_parameters_wc_n9 = "2023_01_10_15h50min51sec_n_9_2000pts_wilson_cowan" \
                        "_celegans_signed_parameters_dictionary.json"
path_parameters_wc_n30 = "2023_01_10_16h16min37sec_n_30_1000pts_wilson_cowan" \
                         "_celegans_signed_parameters_dictionary.json"
path_parameters_wc_n80 = "2023_01_10_16h25min33sec_n_80_1000pts_wilson_cowan" \
                         "_celegans_signed_parameters_dictionary.json"
path_complete_forward_bifurcation_wc =\
    "2023_01_10_15h50min51sec_n_9_2000pts_x_forward_equilibrium_points" \
    "_list_complete_wilson_cowan_celegans_signed.json"
path_complete_backward_bifurcation_wc =\
    "2023_01_10_15h50min51sec_n_9_2000pts_x_backward_equilibrium_points" \
    "_list_complete_wilson_cowan_celegans_signed.json"
path_reduced_forward_bifurcation_n9_wc =\
    "2023_01_10_15h50min51sec_n_9_2000pts_redx_forward_equilibrium_points" \
    "_list_reduced_wilson_cowan_celegans_signed.json"
path_reduced_backward_bifurcation_n9_wc = \
    "2023_01_10_15h50min51sec_n_9_2000pts_redx_backward_equilibrium_points" \
    "_list_reduced_wilson_cowan_celegans_signed.json"
path_reduced_forward_bifurcation_n30_wc =\
    "2023_01_10_16h16min37sec_n_30_1000pts_redx_forward_equilibrium_points" \
    "_list_reduced_wilson_cowan_celegans_signed.json"
path_reduced_backward_bifurcation_n30_wc =\
    "2023_01_10_16h16min37sec_n_30_1000pts_redx_backward_equilibrium_points" \
    "_list_reduced_wilson_cowan_celegans_signed.json"
path_reduced_forward_bifurcation_n80_wc =\
    "2023_01_10_16h25min33sec_n_80_1000pts_redx_forward_equilibrium_points" \
    "_list_reduced_wilson_cowan_celegans_signed.json"
path_reduced_backward_bifurcation_n80_wc =\
    "2023_01_10_16h25min33sec_n_80_1000pts_redx_backward_equilibrium_points" \
    "_list_reduced_wilson_cowan_celegans_signed.json"
graph_str = "celegans_signed"
A_wc = get_connectome_weight_matrix(graph_str)
S_wc = svdvals(A_wc)
N_arange_wc = np.arange(1, len(A_wc[0])+1, 1)

""" RNN """
path_error_rnn = "2022_07_12_15h51min54sec_1000_samples_frobenius_shrinkage" \
                 "_RMSE_vector_field_rnn_mouse_control_rnn.json"
path_upper_bound_rnn = "2022_07_12_15h51min54sec_1000_samples_frobenius" \
                       "_shrinkage_upper_bound_RMSE_vector_field_rnn" \
                       "_mouse_control_rnn.json"
path_x_limit_cycle_rnn = "2023_01_08_12h48min42sec_nice_limit_cycle" \
                         "_x_time_series_complete_rnn_mouse_control_rnn.json"
path_tc_limit_cycle_rnn = "2023_01_08_12h48min42sec_nice_limit_cycle" \
                          "_time_points_complete_rnn_mouse_control_rnn.json"
path_redx1_limit_cycle_rnn = "2023_01_17_15h57min58sec_n_80_redx_time_series" \
                             "_reduced_rnn_mouse_control_rnn.json"
path_tr1_limit_cycle_rnn = "2023_01_17_15h57min58sec_n_80_time_points" \
                           "_reduced_rnn_mouse_control_rnn.json"
path_redx2_limit_cycle_rnn = "2023_01_08_12h48min42sec_nice_limit_cycle_redx" \
                             "_time_series_reduced_rnn_mouse_control_rnn.json"
path_tr2_limit_cycle_rnn = "2023_01_08_12h48min42sec_nice_limit_cycle_time" \
                           "_points_reduced_rnn_mouse_control_rnn.json"
path_redx3_limit_cycle_rnn = "2023_01_10_11h44min17sec_n_100_redx_time_" \
                             "series_reduced_rnn_mouse_control_rnn.json"
path_tr3_limit_cycle_rnn = "2023_01_10_11h44min17sec_n_100_time_points" \
                           "_reduced_rnn_mouse_control_rnn.json"
graph_str = "mouse_control_rnn"
A_rnn = get_learned_weight_matrix(graph_str)
N = len(A_rnn[0])  # Dimension of the complete dynamics  669
U, S, Vh = np.linalg.svd(A_rnn)
shrink_s = optimal_shrinkage(S, 1, 'frobenius')
A_rnn = U@np.diag(shrink_s)@Vh
N_arange_rnn = np.arange(1, len(A_rnn[0])+1, 1)

# Global parameters for error vector fields
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

    # if dynamics_str is not "rnn":
    #    if xlabel_bool:
    #        plt.xlabel('Dimension $n$')
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
        plt.text(-len(mean_error) / 6, 0.0000035, "0 -")

    elif dynamics_str == "microbial":
        ticks = ax.get_xticks().tolist()
        ticks[ticks.index(0)] = 1
        ticks += [250, 500, 750]
        # ticks = [i for i in ticks
        #          if -0.1*len(N_arange) < i < 1.1*len(N_arange)]
        plt.xticks(ticks)
        plt.text(-len(mean_error) / 6.1, 0.0000035, "0 -")

    else:
        plt.text(-len(mean_error) / 6.1, 0.0000035, "0 -")

    ax.set_ylim([0.75*ymin, ymax])
    ax.set_yscale('log', nonposy="clip")
    plt.tick_params(axis='y', which='both', left=True,
                    right=False, labelbottom=False)
    plt.minorticks_off()
    if xlabel_bool:
            plt.xlabel('Dimension $n$')


""" -----------------------  Figure 3 ------------------------------------- """

fig = plt.figure(figsize=(11.5, 5.5))

title_pad = -12
letter_posx, letter_posy = -0.27, 1.08
xlabel_ypos = 0.000002
fontsize_legend = 12
linewidth = 1.5
s = 3.5


def round_to_1(num):
    return round(num, -int(floor(log10(abs(num)))))


# --------------------------- SIS ---------------------------------------------
""" Error """
ax1 = plt.subplot(241)
ax1.set_title("Epidemiological", fontsize=fontsize_legend,
              pad=title_pad)
plot_error(ax1, "qmf_sis", path_error_sis, path_upper_bound_sis, N_arange_sis)
plot_singvals(ax1, S_sis)
# ax1.set_xlabel('Dimension $n$')
# ax1.text(348, xlabel_ypos, "$n$", fontsize=12)
ax1.text(letter_posx, letter_posy, "a", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)
# ax1.vlines(x=1, ymin=0, ymax=0.0001, linestyle="--", color=reduced_grey)
# ax1.vlines(x=104, ymin=0, ymax=0.0001, linestyle="--", color=reduced_grey)

""" Bifurcations """
ax5 = plt.subplot(245)
dynamics_str = "qmf_sis"
with open(path_str + f"{dynamics_str}_data/" +
          path_parameters_sis) as json_data:
    parameters_dictionary = json.load(json_data)
with open(path_str + f"{dynamics_str}_data/"
          + path_complete_bifurcation_sis) as json_data:
    X1c = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_bifurcation_n1_sis) as json_data:
    X1r1 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_bifurcation_n7_sis) as json_data:
    X1r2 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_bifurcation_n104_sis) as json_data:
    X1r3 = np.array(json.load(json_data))

coupling_constants = parameters_dictionary["coupling_constants"]

ax5.scatter(coupling_constants, X1c,
            color=deep[0], label="Exact", s=s)
ax5.plot(coupling_constants, X1r1, color=deep[4],
         label="$n = 1 \,\,(e = \mathrm{rmse} = $" +
               f"${round_to_1(rmse(X1c, X1r1))})$",
         linewidth=linewidth)
ax5.plot(coupling_constants, X1r2, color=deep[2],
         label="$n = 7 \\approx \mathrm{srank} \,\,(e = $" +
               f"${round_to_1(rmse(X1c, X1r2))})$",
         linewidth=linewidth)
ax5.plot(coupling_constants, X1r3, color=deep[1],
         label="$n = 104 = \mathrm{energy} \,\,(e = $" +
               f"${round_to_1(rmse(X1c, X1r3))})$",
         linewidth=linewidth)
ylab = plt.ylabel('$\\mathcal{X}^*$')
ylab.set_rotation(0)
plt.xlabel("Infection rate")
plt.ylim([-0.04, 1.04])
plt.tick_params(axis='both', which='major')
plt.legend(loc=2, fontsize=8, handlelength=1)
plt.yticks([0, 0.5, 1])
ax5.text(letter_posx, letter_posy, "e", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax5.transAxes)

# axins = inset_axes(ax5, width="30%", height="30%",
#                    bbox_to_anchor=(-0.05, .09, 1, 1),
#                    bbox_transform=ax5.transAxes, loc=4)
# axins.scatter(coupling_constants,
#               X1c_equilibrium_points,
#               color=deep[0], label="Exact", s=s)
# axins.plot(coupling_constants,
#            X1r1_equilibrium_points, color=deep[4],
#            label="$n = 1$", linewidth=linewidth)
# axins.plot(coupling_constants,
#            X1r2_equilibrium_points, color=deep[2],
#            label="$n = 7 \\approx \mathrm{srank}$", linewidth=linewidth)
# axins.plot(coupling_constants,
#            X1r3_equilibrium_points, color=deep[1],
#            label="$n = 104 = \mathrm{energy}$", linewidth=linewidth)
# axins.set_xlim([0.999, 1.011])
# axins.set_ylim([0.027, 0.033])
# axins.set_xticks([1, 1.01])
# axins.set_yticks([0.03])
# # axins2.spines['top'].set_visible(True)
# # axins2.spines['right'].set_visible(True)
# axins.tick_params(axis='both', which='major', labelsize=8)


# ----------------------- Wilson-Cowan-----------------------------------------
""" Errors """
ax2 = plt.subplot(242)
ax2.set_title("Neuronal", fontsize=fontsize_legend,
              pad=title_pad)
plot_error(ax2, "wilson_cowan", path_error_wc,
           path_upper_bound_wc, N_arange_wc)
plot_singvals(ax2, S_wc)
ax2.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)
# ax2.text(320, xlabel_ypos, "$n$", fontsize=12)
# ax2.vlines(x=39, ymin=0, ymax=0.0001, linestyle="--", color=reduced_grey)
# ax2.vlines(x=80, ymin=0, ymax=0.0001, linestyle="--", color=reduced_grey)


""" Bifurcations"""
ax6 = plt.subplot(246)
dynamics_str = "wilson_cowan"
with open(path_str + f"{dynamics_str}_data/" +
          path_parameters_wc_n9) as json_data:
    parameters_dictionary_wc_n9 = json.load(json_data)
with open(path_str + f"{dynamics_str}_data/" +
          path_parameters_wc_n30) as json_data:
    parameters_dictionary_wc_n30 = json.load(json_data)
with open(path_str + f"{dynamics_str}_data/" +
          path_parameters_wc_n80) as json_data:
    parameters_dictionary_wc_n80 = json.load(json_data)

with open(path_str + f"{dynamics_str}_data/"
          + path_complete_forward_bifurcation_wc) as json_data:
    Xcf = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_complete_backward_bifurcation_wc) as json_data:
    Xcb = np.array(json.load(json_data))

with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_forward_bifurcation_n9_wc) as json_data:
    Xrf_n9 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_backward_bifurcation_n9_wc) as json_data:
    Xrb_n9 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_forward_bifurcation_n30_wc) as json_data:
    Xrf_n30 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_backward_bifurcation_n30_wc) as json_data:
    Xrb_n30 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_forward_bifurcation_n80_wc) as json_data:
    Xrf_n80 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_backward_bifurcation_n80_wc) as json_data:
    Xrb_n80 = np.array(json.load(json_data))

Xcfb = np.concatenate([Xcf, Xcb])
Xrfb9 = np.concatenate([Xrf_n9, Xrb_n9])
Xrfb30 = np.concatenate([Xrf_n30, Xrb_n30])
Xrfb80 = np.concatenate([Xrf_n80, Xrb_n80])

# The coupling constants are the same for every n
coupling_constants_n9 =\
    np.array(parameters_dictionary_wc_n9["coupling_constants"])
# coupling_constants_n30 =\
#     np.array(parameters_dictionary_wc_n30["coupling_constants"])
# coupling_constants_n80 = \
#     np.array(parameters_dictionary_wc_n80["coupling_constants"])
# print(len(coupling_constants_n9), len(coupling_constants_n30), len(coupling_constants_n80))
ax6.text(letter_posx, letter_posy, "f", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax6.transAxes)
plt.scatter(coupling_constants_n9, Xcf, color=deep[0], label="Exact", s=s)
plt.scatter(coupling_constants_n9, Xcb, color=deep[0], s=s)
plt.plot(coupling_constants_n9[:1860], Xrf_n9[Xrf_n9 < 0.13], color=deep[4],
         label="$n = 9 \\approx \mathrm{srank} \,\,(e = $" +
               f"${round_to_1(rmse(Xcfb, Xrfb9))})$", linewidth=linewidth)
plt.plot(coupling_constants_n9[1860:], Xrf_n9[Xrf_n9 >= 0.13], color=deep[4],
         linewidth=linewidth)
plt.plot(coupling_constants_n9[:1849], Xrb_n9[Xrb_n9 < 0.13], color=deep[4],
         linewidth=linewidth)
plt.plot(coupling_constants_n9[1849:], Xrb_n9[Xrb_n9 >= 0.13], color=deep[4],
         linewidth=linewidth)
plt.plot(coupling_constants_n30[:866], Xrf_n30[Xrf_n30 < 0.12], color=deep[2],
         label="$n = 30 \\approx \mathrm{nrank} \,\,(e = $" +
               f"${round_to_1(rmse(Xcfb, Xrfb30))})$", linewidth=linewidth)
plt.plot(coupling_constants_n30[866:], Xrf_n30[Xrf_n30 >= 0.12], color=deep[2],
         linewidth=linewidth)
plt.plot(coupling_constants_n30[:822], Xrb_n30[Xrb_n30 < 0.12], color=deep[2],
         linewidth=linewidth)
plt.plot(coupling_constants_n30[822:], Xrb_n30[Xrb_n30 >= 0.12], color=deep[2],
         linewidth=linewidth)
plt.plot(coupling_constants_n80[:834], Xrf_n80[Xrf_n80 < 0.11], color=deep[1],
         label="$n = 80 = \mathrm{shrank}$ \,\,(e = $" +
               f"${round_to_1(rmse(Xcfb, Xrfb80))})", linewidth=linewidth)
plt.plot(coupling_constants_n80[834:], Xrf_n80[Xrf_n80 >= 0.11], color=deep[1],
         linewidth=linewidth)
plt.plot(coupling_constants_n80[:795], Xrb_n80[Xrb_n80 < 0.11], color=deep[1],
         linewidth=linewidth)
plt.plot(coupling_constants_n80[795:], Xrb_n80[Xrb_n80 >= 0.11], color=deep[1],
         linewidth=linewidth)
plt.xticks([0.02, 0.06, 0.10, 0.14])
ylab = plt.ylabel('$\\mathcal{X}^*$')
ylab.set_rotation(0)
plt.xlabel('Synaptic weight')
plt.ylim([-0.001, 0.26])
plt.yticks([0, 0.1, 0.2])
plt.legend(loc=2, fontsize=8, handlelength=1)

axins = inset_axes(ax6, width="35%", height="25%",
                   bbox_to_anchor=(.2, .32, 1, 1),
                   bbox_transform=ax6.transAxes, loc=3)
axins.scatter(coupling_constants_n9, Xcf, color=deep[0], s=s)
axins.scatter(coupling_constants_n9, Xcb, color=deep[0], s=s)
axins.plot(coupling_constants_n80[:834], Xrf_n80[Xrf_n80 < 0.11],
           color=deep[2], label="$n = 80 = \mathrm{shrank}$",
           linewidth=linewidth)
plt.scatter(coupling_constants_n9, Xcf, color=deep[0], label="Exact", s=s)
plt.scatter(coupling_constants_n9, Xcb, color=deep[0], s=s)
# plt.plot(coupling_constants_n9[:1860], Xrf_n9[Xrf_n9 < 0.13], color=deep[4],
#          label="$n = 9 \\approx \mathrm{srank}$", linewidth=linewidth)
# plt.plot(coupling_constants_n9[1860:], Xrf_n9[Xrf_n9 >= 0.13], color=deep[4],
#          linewidth=linewidth)
# plt.plot(coupling_constants_n9[:1849], Xrb_n9[Xrb_n9 < 0.13], color=deep[4],
#          linewidth=linewidth)
# plt.plot(coupling_constants_n9[1849:], Xrb_n9[Xrb_n9 >= 0.13], color=deep[4],
#          linewidth=linewidth)
plt.plot(coupling_constants_n30[:866], Xrf_n30[Xrf_n30 < 0.12], color=deep[2],   
         label="$n = 30 \\approx \mathrm{nrank}$", linewidth=linewidth)          
plt.plot(coupling_constants_n30[866:], Xrf_n30[Xrf_n30 >= 0.12], color=deep[2],  
         linewidth=linewidth)                                                    
plt.plot(coupling_constants_n30[:822], Xrb_n30[Xrb_n30 < 0.12], color=deep[2],   
         linewidth=linewidth)                                                    
plt.plot(coupling_constants_n30[822:], Xrb_n30[Xrb_n30 >= 0.12], color=deep[2],  
         linewidth=linewidth)                                                    
plt.plot(coupling_constants_n80[:834], Xrf_n80[Xrf_n80 < 0.11], color=deep[1],
         label="$n = 80 = \mathrm{shrank}$", linewidth=linewidth)
plt.plot(coupling_constants_n80[834:], Xrf_n80[Xrf_n80 >= 0.11], color=deep[1],
         linewidth=linewidth)
plt.plot(coupling_constants_n80[:795], Xrb_n80[Xrb_n80 < 0.11], color=deep[1],
         linewidth=linewidth)
plt.plot(coupling_constants_n80[795:], Xrb_n80[Xrb_n80 >= 0.11], color=deep[1],
         linewidth=linewidth)
axins.set_xlim([0.112, 0.124])
axins.set_ylim([0.075, 0.158])
axins.set_xticks([0.113, 0.123])
axins.set_yticks([0.11, 0.15])
axins.spines['top'].set_visible(True)
axins.spines['right'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    axins.spines[axis].set_linewidth(0.5)
axins.tick_params(axis='both', which='major', labelsize=8,
                  width=0.5, length=2)


axins2 = inset_axes(ax6, width="18%", height="18%",
                    bbox_to_anchor=(0.03, .07, 1, 1),
                    bbox_transform=ax6.transAxes, loc=4)
axins2.plot(coupling_constants_n9[:1860], Xrf_n9[Xrf_n9 < 0.13], color=deep[4],
            label="$n = 9 \\approx \mathrm{srank}$")
axins2.plot(coupling_constants_n9[1860:], Xrf_n9[Xrf_n9 >= 0.13],
            color=deep[4], linewidth=linewidth)
axins2.plot(coupling_constants_n9[:1849], Xrb_n9[Xrb_n9 < 0.13],
            color=deep[4], linewidth=linewidth)
axins2.plot(coupling_constants_n9[1849:], Xrb_n9[Xrb_n9 >= 0.13],
            color=deep[4], linewidth=linewidth)
axins2.set_xlim([0.129, 0.132])
axins2.set_ylim([0.108, 0.148])
axins2.set_xticks([0.13])
axins2.set_yticks([0.11, 0.13])
axins2.spines['top'].set_visible(True)
axins2.spines['right'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    axins2.spines[axis].set_linewidth(0.5)
axins2.tick_params(axis='both', which='major', labelsize=8,
                   width=0.5, length=2)

# ------------------------ Microbial ------------------------------------------
""" Errors """
ax3 = plt.subplot(243)
ax3.set_title("Microbial", fontsize=fontsize_legend,
              pad=title_pad)
plot_error(ax3, "microbial", path_error_microbial,
           path_upper_bound_microbial, N_arange_microbial)
plot_singvals(ax3, S_microbial)
# ax3.text(890, xlabel_ypos, "$n$", fontsize=12)
ax3.text(letter_posx, letter_posy, "c", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax3.transAxes)

""" Bifurcations """
ax7 = plt.subplot(247)
ax7.text(letter_posx, letter_posy, "g", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax7.transAxes)
dynamics_str = "microbial"
with open(path_str + f"{dynamics_str}_data/" +
          path_parameters_microbial_n76) as json_data:
    parameters_dictionary_microbial_n76 = json.load(json_data)
with open(path_str + f"{dynamics_str}_data/" +
          path_parameters_microbial_n203) as json_data:
    parameters_dictionary_microbial_n203 = json.load(json_data)
with open(path_str + f"{dynamics_str}_data/" +
          path_parameters_microbial_n400) as json_data:
    parameters_dictionary_microbial_n400 = json.load(json_data)

with open(path_str + f"{dynamics_str}_data/"
          + path_complete_forward_bifurcation_microbial) as json_data:
    Xcf = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_complete_backward_bifurcation_microbial) as json_data:
    Xcb = np.array(json.load(json_data))

with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_forward_bifurcation_n76_microbial) as json_data:
    Xrf_n76 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_backward_bifurcation_n76_microbial) as json_data:
    Xrb_n76 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_forward_bifurcation_n203_microbial) as json_data:
    Xrf_n203 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_backward_bifurcation_n203_microbial) as json_data:
    Xrb_n203 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_forward_bifurcation_n400_microbial) as json_data:
    Xrf_n400 = np.array(json.load(json_data))
with open(path_str + f"{dynamics_str}_data/"
          + path_reduced_backward_bifurcation_n400_microbial) as json_data:
    Xrb_n400 = np.array(json.load(json_data))
coupling_constants =\
    np.array(parameters_dictionary_microbial_n76["coupling_constants_forward"])
# same coupling constants linspace for  in every case
plt.scatter(coupling_constants[:44], Xcf[Xcf < 2], color=deep[0],
            label="Exact", s=s)
plt.scatter(coupling_constants, Xcb, color=deep[0], s=s)
plt.vlines(x=coupling_constants[44], ymin=0.1315, ymax=6.8, linestyle="--",
           linewidth=1, color=deep[0])
plt.plot(coupling_constants[:76], Xrf_n76[Xrf_n76 < 2], color=deep[4],
         label="$n = 76 \\approx \mathrm{erank}$", linewidth=linewidth)
plt.plot(coupling_constants, Xrb_n76, color=deep[4], linewidth=linewidth)
plt.vlines(x=coupling_constants[76], ymin=0.139, ymax=5.097, linestyle="--",
           linewidth=1, color=deep[4])
plt.plot(coupling_constants[:67], Xrf_n203[Xrf_n203 < 2], color=deep[2],
         label="$n = 203 = \mathrm{shrank}$", linewidth=linewidth)
plt.plot(coupling_constants, Xrb_n203, color=deep[2], linewidth=linewidth)
plt.vlines(x=coupling_constants[67], ymin=0.14, ymax=6.45, linestyle="--",
           linewidth=1, color=deep[2])
plt.plot(coupling_constants[:66], Xrf_n400[Xrf_n400 < 2], color=deep[1],
         label="$n = 400$", linewidth=linewidth)
plt.plot(coupling_constants, Xrb_n400, color=deep[1], linewidth=linewidth)
plt.vlines(x=coupling_constants[66], ymin=0.1408, ymax=7.39, linestyle="--",
           linewidth=1, color=deep[1])
plt.xticks([1.5, 2, 2.5, 3])
ylab = plt.ylabel('$\\mathcal{X}^*$')
ylab.set_rotation(0)
plt.ylim([-0.3, 11.3])
plt.yticks([0, 4, 8])
plt.legend(loc=2, fontsize=8, handlelength=1)
plt.xlabel('Microbial interaction weight')

# --------------------------- RNN ---------------------------------------------
""" Errors """
ax4 = plt.subplot(244)
divider = make_axes_locatable(ax4)
ax42 = divider.new_horizontal(size="100%", pad=0.1)
fig.add_axes(ax42)
plot_error(ax4, "rnn", path_error_rnn,
           path_upper_bound_rnn, N_arange_rnn, xlabel_bool=False)
plot_singvals(ax4, shrink_s)
ax4.set_xlim(-10, 120)
plot_error(ax42, "rnn", path_error_rnn,
           path_upper_bound_rnn, N_arange_rnn, xlabel_bool=False)
plot_singvals(ax42, shrink_s)
ax4.text(letter_posx-0.2, letter_posy, "d", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax4.transAxes)
ax42.set_xlim(630, 670)
ax42.tick_params(left=False, labelleft=False)
ax42.set_yticks([])
ax42.spines['left'].set_visible(False)
d = .025  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax42.transAxes, color=dark_grey, clip_on=False)
ax42.plot((-d, +d), (-d, +d), zorder=10, linewidth=1.5, **kwargs)
kwargs.update(transform=ax4.transAxes)
ax4.plot((1 - d, 1 + d), (-d, d), zorder=10, linewidth=1.5, **kwargs)
# ax42.text(575, 1000, "Recurrent neural network", fontsize=12)
# ax42.text(595, 100, "Recurrent neural network", fontsize=10)
ax42.text(605, 100, "Recurrent neural", fontsize=fontsize_legend)
# ax42.text(605, 100, "$\quad\quad$RNN", fontsize=12)
ax4.text(-120/3.2, 0.0000035, "0 -")
ax42.text(620, 0.00001, "Exact rank reduction\nat $n=\mathrm{rank}(W) = 102$",
          fontsize=8)
# ax42.text(672, xlabel_ypos, "$n$", fontsize=12)
ax4.set_xlabel('Dimension $n$')
ax4.xaxis.set_label_coords(1.1, -0.19)
# ticks = ax3.get_xticks()
# ticks[ticks.tolist().index(0)] = 1
# ticks = [i for i in ticks
#          if -0.1*200 < i < 1.1*200]
ax4.set_xticks([1, 100])
ax42.set_xticks([650])
# ax3.set_title("$\\qquad\\qquad\\qquad$Chaotic RNN",
#               fontsize=fontsize_legend, pad=title_pad)
# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc=(0.3, 0.95))

""" Bifurcations """
ax8 = fig.add_subplot(248, projection='3d')
with open(path_str + f"rnn_data/" + path_x_limit_cycle_rnn) as json_data:
    x = np.array(json.load(json_data))
with open(path_str + f"rnn_data/" + path_redx1_limit_cycle_rnn) as json_data:
    redx1 = np.array(json.load(json_data))
with open(path_str + f"rnn_data/" + path_redx2_limit_cycle_rnn) as json_data:
    redx2 = np.array(json.load(json_data))
with open(path_str + f"rnn_data/" + path_redx3_limit_cycle_rnn) as json_data:
    redx3 = np.array(json.load(json_data))
with open(path_str + f"rnn_data/" + path_tc_limit_cycle_rnn) as json_data:
    tc = np.array(json.load(json_data))
with open(path_str + f"rnn_data/" + path_tr1_limit_cycle_rnn) as json_data:
    tr1 = np.array(json.load(json_data))
with open(path_str + f"rnn_data/" + path_tr2_limit_cycle_rnn) as json_data:
    tr2 = np.array(json.load(json_data))
with open(path_str + f"rnn_data/" + path_tr3_limit_cycle_rnn) as json_data:
    tr3 = np.array(json.load(json_data))
time_cut_c = 0.5
time_cut = 0.5
Un, Sn, M = computeTruncatedSVD_more_positive(A_rnn, 100)
X1c, X2c, X3c = M[0, :]@x, M[1, :]@x, M[2, :]@x
X1r1, X2r1, X3r1 = redx1[0, :], redx1[1, :], redx1[2, :]
X1r2, X2r2, X3r2 = redx2[0, :], redx2[1, :], redx2[2, :]
X1r3, X2r3, X3r3 = redx3[0, :], redx3[1, :], redx3[2, :]
ax8.scatter(X1c[int(time_cut_c*len(tc)):],
            X2c[int(time_cut_c*len(tc)):],
            X3c[int(time_cut_c*len(tc)):], color=deep[0], s=s,  # linewidth=2,
            label="Exact", zorder=0)
ax8.plot(X1r1[int(time_cut*len(tr1)):],
         X2r1[int(time_cut*len(tr1)):],
         X3r1[int(time_cut*len(tr1)):], color=deep[4],
         linewidth=0.5, label="$n = 80 \\approx \\mathrm{erank}$", zorder=3)
ax8.plot(X1r2[int(time_cut*len(tr2)):],
         X2r2[int(time_cut*len(tr2)):],
         X3r2[int(time_cut*len(tr2)):], color=deep[2],
         linewidth=1, label="$n = 90$", zorder=2)
ax8.plot(X1r3[int(time_cut*len(tr3)):],
         X2r3[int(time_cut*len(tr3)):],
         X3r3[int(time_cut*len(tr3)):], color=deep[1],
         linewidth=1, label="$n = 100$", zorder=1)
ax8.view_init(45, -135)
ax8.set_xlabel("$X_1$")
ax8.set_ylabel("$X_2$")
ax8.zaxis.set_rotate_label(False)
ax8.set_zlabel("$X_3$", rotation=0)
ax8.set_xticks([-0.2, 0, 0.2])
ax8.set_yticks([-0.2, 0, 0.2])
ax8.set_zticks([-0.2, 0, 0.2])
# ax8.tick_params(axis='both', which='major', labelsize=8)
plt.legend(loc=(-0.2, 0.7), fontsize=8, handlelength=1)
ax8.text2D(letter_posx, letter_posy, "h", fontweight="bold",
           horizontalalignment="center", verticalalignment="top",
           transform=ax8.transAxes)
plt.tight_layout()
# fig.subplots_adjust(bottom=0, top=100)
# plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
# fig.tight_layout(rect=[0, 0, .8, 1])
# .subplots_adjust(bottom=0.1, right=0.8, top=0.9)

plt.show()
