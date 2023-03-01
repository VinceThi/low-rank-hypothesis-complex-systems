# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svdvals
from plots.config_rcparams import *
import time
import json
import tkinter.simpledialog
from tkinter import messagebox
from tqdm import tqdm
from plots.plot_weight_matrix import plot_weight_matrix
from graphs.sbm_properties import normalize_degree_propensity
from graphs.generate_truncated_pareto import truncated_pareto
from graphs.generate_degree_corrected_stochastic_block_model import \
    degree_corrected_stochastic_block_model
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/singular_values/properties/" \
           "singular_values_random_graphs/"

plot_weight_mat = False
plot_degrees = False

""" Random graph parameters """
graph_str = "sbm"
N = 1000
nb_networks = 1    # 1000
directed = True
selfloops = True
norm_choice = 2
expected_nb_edges = N*np.array([[0.30, 0.10, 0.10, 0.02, 0.13],
                                [0.05, 0.80, 0.02, 0.09, 0.10],
                                [0.02, 0.02, 0.20, 0.05, 0.02],
                                [0.10, 0.05, 0.05, 0.40, 0.01],
                                [0.10, 0.09, 0.05, 0.05, 0.60]])

g = 100   # 100, 5
expected_nb_edges = g*expected_nb_edges

sizes = np.array([N//10, 2*N//5, N//10, N//5, N//5])

kappa_in_min = 5
kappa_in_max = 100
gamma_in = 2.5
kappa_in = truncated_pareto(N, kappa_in_min, kappa_in_max, gamma_in)
nkappa_in = normalize_degree_propensity(kappa_in, sizes)

kappa_out_min = 3
kappa_out_max = 50
gamma_out = 2
kappa_out = truncated_pareto(N, kappa_out_min, kappa_out_max, gamma_out)
nkappa_out = normalize_degree_propensity(kappa_out, sizes)


norm_R = np.zeros(nb_networks)

""" Get singular values """
singularValues = np.zeros((nb_networks, N))
singularValues_R = np.zeros((nb_networks, N))
for i in tqdm(range(0, nb_networks)):
    W, EW = degree_corrected_stochastic_block_model(
        sizes, expected_nb_edges, nkappa_in, nkappa_out,
        selfloops=True, expected=True)
    if plot_weight_mat:
        plot_weight_matrix(W)
    if plot_degrees:
        plt.figure(figsize=(6, 4))
        plt.hist(np.sum(EW, axis=1), bins=100, label="$\\kappa_{in}$",
                 density=True)
        plt.hist(np.sum(EW, axis=0), bins=100, label="$\\kappa_{out}$",
                 density=True)
        plt.ylabel("Density")
        plt.legend(loc=1)
        plt.show()

    R = W - EW
    norm_R[i] = norm(R, ord=norm_choice)
    singularValues[i, :] = svdvals(W)
    singularValues_R[i, :] = svdvals(R)

singularValues_EW = svdvals(EW)
norm_EW = norm(EW, ord=norm_choice)


""" Plot singular values and Weyl's theorem"""

xlabel = "Index $i$"
# ylabel = "Average rescaled singular\n values $\\sigma_i/\\sigma_1$"
# ylabel = "Average singular values"

mean_norm_W = np.mean(singularValues[:, 0])
mean_norm_R = np.mean(norm_R)
norm_ratio = mean_norm_R/mean_norm_W

mean_singularValues = np.mean(singularValues, axis=0)
bar_singularValues = np.std(singularValues, axis=0)

mean_singularValues_R = np.mean(singularValues_R, axis=0)
bar_singularValues_R = np.std(singularValues_R,  axis=0)

fig = plt.figure(figsize=(4, 4))
ax1 = plt.subplot(111)
indices = np.arange(1, N + 1, 1)
# ax1.scatter(indices, mean_singularValues/mean_norm_W, s=30, color=deep[0],
#             label="$\\langle\\sigma_i(W)\\rangle\,/"
#                   "\,\\langle\,||W||_2\,\\rangle$")
# ax1.fill_between(indices,
#                  (mean_singularValues - std_singularValues)/mean_norm_W,
#                  (mean_singularValues + std_singularValues)/mean_norm_W,
#                  color=deep[0], alpha=0.2)
ax1.errorbar(x=indices, y=mean_singularValues/mean_norm_W,
             yerr=bar_singularValues/mean_norm_W, fmt="o", color=deep[0],
             zorder=-30, markersize=5, capsize=1, elinewidth=1,
             label="$\\langle\\sigma_i(W)\\rangle\,/"
                   "\,\\langle\,||W||_2\,\\rangle$")
# ax1.scatter(indices, mean_singularValues_R/mean_norm_W, s=2, color=deep[9],
#             label="$\\langle\\sigma_i(R)\\rangle\,/"
#                   "\,\\langle\,||W||_2\,\\rangle$", zorder=2)
# ax1.fill_between(indices,
#                  (mean_singularValues_R - bar_singularValues_R)/mean_norm_W,
#                  (mean_singularValues_R + bar_singularValues_R)/mean_norm_W,
#                  color=deep[2], alpha=0.2)
ax1.scatter(indices, singularValues_EW/mean_norm_W,
            label="$\\sigma_i(\\langle W \\rangle)\,/"
                  "\,\\langle\,||W||_2\,\\rangle$",
            s=5, color=deep[1], zorder=0)

ax1.errorbar(x=indices, y=mean_singularValues_R/mean_norm_W,
             yerr=bar_singularValues_R/mean_norm_W, fmt="o", color=deep[9],
             zorder=30, markersize=1.5, capsize=1, elinewidth=1,
             label="$\\langle\\sigma_i(R)\\rangle\,/"
                   "\,\\langle\,||W||_2\,\\rangle$")
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
           loc=4, bbox_to_anchor=(0, 0.1, 1, 1))
ticks = ax1.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])
plt.tick_params(axis='both', which='major')
# plt.ylim([0.01*np.min(singularValues), 1.5])
# ax1.set_yscale('log')
# plt.tick_params(axis='y', which='both', left=True,
#                 right=False, labelbottom=False)
plt.xlabel(xlabel)
# plt.ylabel(ylabel)  # , labelpad=20)
plt.xlim([-50, N+50])

axins = inset_axes(ax1, width="50%", height="50%",
                   bbox_to_anchor=(-0.1, 0, 1, 1),
                   bbox_transform=ax1.transAxes, loc=1)
axins.plot(indices, mean_norm_R*np.ones(N)/mean_norm_W,
           linewidth=1, linestyle="--", color=deep[9])
axins.text(0.92, 0.97, "$\\frac{\\langle\,||R||_2\,\\rangle}"
                       "{\\langle\,||W||_2\,\\rangle}$", fontweight="bold",
           horizontalalignment="center", verticalalignment="top",
           transform=ax1.transAxes, fontsize=8)
axins.scatter(indices,
              np.mean(np.abs(singularValues-np.outer(np.ones(nb_networks),
                                                     singularValues_EW)),
                      axis=0)/mean_norm_W,
              color=deep[7], s=2)
# axins.errorbar(x=indices, y=np.mean(np.abs(singularValues-singularValues_EW),
#                                     axis=0)/mean_norm_W,
#                yerr=np.std(np.abs(singularValues-singularValues_EW),
#                            axis=0)/mean_norm_W, fmt="o", color=deep[7],
#                markersize=1, capsize=0, elinewidth=1)
axins.set_ylabel("$\\langle\,\,|\,\\sigma_i(W)"
                 " - \\sigma_i(\\langle W\\rangle)\,|\,\,\\rangle"
                 "\,\,/\,\,\\langle\,||W||_2\,\\rangle$",
                 fontsize=8)
axins.set_xlabel("Index $i$", fontsize=8)
ticks = axins.get_xticks()
ticks[ticks.tolist().index(0)] = 1
axins.set_xticks(ticks[ticks > 0])
axins.set_xlim([-50, N+50])
for axis in ['top', 'bottom', 'left', 'right']:
    axins.spines[axis].set_linewidth(0.5)
axins.tick_params(axis='both', which='major', labelsize=8,
                  width=0.5, length=2)

# plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters,"
                       " the data, and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    path = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "singular_values/properties/singular_values_random_graphs/"
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    parameters_dictionary = {"graph_str": graph_str, "sizes": sizes,
                             "N": N,
                             "expected_nb_edges": expected_nb_edges.tolist(),
                             "kappa_in_min": kappa_in_min,
                             "kappa_in_max": kappa_in_max,
                             "gamma_in": gamma_in,
                             "kappa_out_min": kappa_out_min,
                             "kappa_out_max": kappa_out_max,
                             "gamma_out": gamma_out,
                             "kappa_in": kappa_in.tolist(),
                             "kappa_out": kappa_out.tolist(),
                             "nkappa_in": nkappa_in.tolist(),
                             "nkappa_out": nkappa_out.tolist(),
                             "directed": directed, "selfloops": selfloops,
                             "nb_samples (nb_networks)": nb_networks,
                             "norm_EW": norm_EW.tolist(),
                             "norm_R": norm_R.tolist(),
                             "norm_ratio": norm_ratio,
                             "norm_choice": norm_choice}

    fig.savefig(path + f'{timestr}_{file}_singular_values_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_singular_values_{graph_str}.png')

    with open(path + f'{timestr}_{file}_singular_values_W'
                     f'_{graph_str}.json', 'w') \
            as outfile:
        json.dump(singularValues.tolist(), outfile)
    with open(path + f'{timestr}_{file}_singular_values_EW'
                     f'_{graph_str}.json', 'w') \
            as outfile:
        json.dump(singularValues_EW.tolist(), outfile)
    with open(path + f'{timestr}_{file}_singular_values_R'
                     f'_{graph_str}.json', 'w') \
            as outfile:
        json.dump(singularValues_R.tolist(), outfile)
    with open(path + f'{timestr}_{file}_singular_values'
                     f'_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
