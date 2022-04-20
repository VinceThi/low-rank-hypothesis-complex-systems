# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_networks import *
from plots.config_rcparams import *
from scipy.linalg import svdvals
from singular_values.optimal_shrinkage import optimal_shrinkage
import json
from singular_values.compute_effective_ranks import computeRank

# TODO: the N_arange or not perfectly chosen right now

path_str = "C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
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
print(len(N_arange_sis), computeRank(S_sis))

""" RNN """
path_error_rnn = "2022_03_25_16h19min32sec_1000_samples_RMSE_vector_field" \
                 "_rnn_mouse_control_rnn.json"
path_upper_bound_rnn = "2022_03_25_16h19min32sec_1000_samples_upper_bound" \
                       "_RMSE_vector_field_rnn_mouse_control_rnn.json"
graph_str = "mouse_control_rnn"
A_rnn = get_learned_weight_matrix(graph_str)
N = len(A_rnn[0])  # Dimension of the complete dynamics
U, S, Vh = np.linalg.svd(A_rnn)
shrink_s = optimal_shrinkage(S, 1, 'operator')
A_rnn = U@np.diag(shrink_s)@Vh
# N_arange_rnn = np.arange(1, computeRank(shrink_s), 1)
N_arange_rnn = np.arange(1, len(A_rnn[0]), 1)

""" Wilson-Cowan """
path_error_wc = "2022_03_25_14h45min47sec_1000_samples_RMSE_vector_field" \
                "_wilson_cowan_celegans_signed.json"
path_upper_bound_wc = "2022_03_25_14h45min47sec_1000_samples_upper_bound" \
                      "_RMSE_vector_field_wilson_cowan_celegans_signed.json"
graph_str = "celegans_signed"
A_wc = get_connectome_weight_matrix(graph_str)
S_wc = svdvals(A_wc)
N_arange_wc = np.arange(1, len(A_wc[0]), 1)
print(len(N_arange_wc))


def plot_singvals(ax, singularValues, ylabel_bool=False):
    ax.scatter(np.arange(1, len(singularValues) + 1, 1),
               singularValues/singularValues[0], s=10,
               color=first_community_color)
    # plt.ylabel("Normalized singular\n values $\\sigma_i/\\sigma_1$")
    if ylabel_bool:
        plt.ylabel("$\\frac{\\sigma_n}{\\sigma_1}$", rotation=0, fontsize=16,
                   color=first_community_color)
        ax.yaxis.labelpad = 20
    ticks = ax.get_xticks()
    ticks[ticks.tolist().index(0)] = 1
    ticks = [i for i in ticks
             if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
    plt.xticks(ticks)
    # plt.ylim([0.01*np.min(singularValues > 1e-13), 1.5])
    ax.set_yscale('log')
    plt.ylim([0.5*10**(-4), 2])
    plt.tick_params(axis='y', which='both', left=True,
                    right=False, labelbottom=False)
    plt.minorticks_off()


def plot_error(ax, dynamics_str, path_error, path_upper_bound, N_arange,
               ylabel_bool=False):

    with open(path_str + f"{dynamics_str}_data/vector_field_errors/" +
              path_error) as json_data:
        error_array = json.load(json_data)

    with open(path_str+f"{dynamics_str}_data/vector_field_errors/" +
              path_upper_bound) as json_data:
        error_upper_bound_array = json.load(json_data)

    if dynamics_str == "rnn":
        nb_samples, nb_sing = np.shape(error_array)
        error_array = \
            np.hstack((error_array, np.zeros((nb_samples,
                                              len(N_arange_rnn) - nb_sing))))
        error_upper_bound_array = \
            np.hstack((error_upper_bound_array,
                       np.zeros((nb_samples, len(N_arange_rnn) - nb_sing))))

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

    # median_error = np.percentile(error_array, q=50, axis=0)
    # median_log10_error = np.log10(median_error)
    # p16_semilog = np.percentile(error_array, q=16, axis=0) / \
    #     (np.log(10)*median_error)
    # p84_semilog = np.percentile(error_array, q=84, axis=0) / \
    #     (np.log(10)*median_error)
    # fill_between_error_1 = 10**(median_log10_error - p16_semilog)
    # fill_between_error_2 = 10**(p84_semilog - median_log10_error)
    #
    # median_upper_bound_error = np.median(error_upper_bound_array, axis=0)
    # median_log10_upper_bound_error = np.log10(median_upper_bound_error)
    # p16_semilog = np.percentile(error_array, q=16, axis=0) / \
    #     (np.log(10) * median_upper_bound_error)
    # p84_semilog = np.percentile(error_array, q=84, axis=0) / \
    #     (np.log(10) * median_upper_bound_error)
    # fill_between_ub1 = 10**(median_log10_upper_bound_error - p16_semilog)
    # fill_between_ub2 = 10**(p84_semilog - median_log10_upper_bound_error)

    # print(np.shape(N_arange), np.shape(mean_error))
    ax.scatter(N_arange, mean_error, s=5, color=deep[3],
               label="RMSE $\\mathcal{E}_f\,(x)$")
    ax.plot(N_arange, mean_upper_bound_error, color=dark_grey,
            label="Upper bound")
    ax.fill_between(N_arange, fill_between_error_1, fill_between_error_2,
                    color=deep[3], alpha=0.5)
    ax.fill_between(N_arange, fill_between_ub1, fill_between_ub2,
                    color=dark_grey, alpha=0.5)
    plt.xlabel('Dimension $n$')
    if ylabel_bool:
        plt.ylabel("$\\mathcal{E}_f\,(x)$", rotation=0,  # fontsize=16,
                   color=deep[3])
        ax.yaxis.labelpad = 20
    # ticks = ax.get_xticks()
    # ticks[ticks.tolist().index(0)] = 1
    # plt.xticks(ticks[ticks > 0])
    # plt.xlim([-0.01 * len(N_arange), 1.01 * len(N_arange)])
    ticks = ax.get_xticks()
    ticks[ticks.tolist().index(0)] = 1
    ticks = [i for i in ticks
             if -0.1*len(N_arange) < i < 1.1*len(N_arange)]
    plt.xticks(ticks)

    # plt.ylim([0.9 * np.min(fill_between_error_1),
    #           1.1 * np.max(fill_between_ub2)])
    plt.ylim([0.5*10**(-4), 500])
    ax.set_yscale('log')
    plt.tick_params(axis='y', which='both', left=True,
                    right=False, labelbottom=False)
    plt.minorticks_off()
    # y_major = matplotlib.ticker.LogLocator(base=10.0, numticks=5)
    # ax.yaxis.set_major_locator(y_major)
    # y_minor = matplotlib.ticker.LogLocator(base=10.0,
    #                                        subs=np.arange(1.0, 10.0)*0.1,
    #                                        numticks=10)
    # ax.yaxis.set_minor_locator(y_minor)
    # ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


plt.figure(figsize=(9, 5))

ax1 = plt.subplot(231)
plot_singvals(ax1, S_sis, ylabel_bool=True)

ax4 = plt.subplot(234)
plot_error(ax4, "qmf_sis", path_error_sis,
           path_upper_bound_sis, N_arange_sis, ylabel_bool=True)

ax2 = plt.subplot(232)
plot_singvals(ax2, S_wc)

ax5 = plt.subplot(235)
plot_error(ax5, "wilson_cowan", path_error_wc,
           path_upper_bound_wc, N_arange_wc)

ax3 = plt.subplot(233)
plot_singvals(ax3, shrink_s)

ax6 = plt.subplot(236)
plot_error(ax6, "rnn", path_error_rnn,
           path_upper_bound_rnn, N_arange_rnn)
# ax4 = plt.subplot(224)

plt.tight_layout()
plt.show()
