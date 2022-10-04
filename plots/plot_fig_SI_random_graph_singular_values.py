# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import json
import numpy as np
from plots.config_rcparams import *
from singular_values.compute_effective_ranks import computeEffectiveRanks

plot_histogram = True
path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/singular_values/properties/" \
           "singular_values_random_graphs/"

""" Warning: must be networks with N = 1000. """
N = 1000
nb_networks = 1000
nb_bins = 1000

""" SBM """
path_SBM = "2022_08_02_11h13min14sec_1000_samples_concatenated" \
           "_singular_values_stochastic_block_model.json"
with open(path_str + path_SBM) as json_data:
    singular_values_sbm = np.array(json.load(json_data))
print(computeEffectiveRanks(singular_values_sbm[:N], "SBM", N))

""" Barabasi-Albert """
path_BA = "2022_08_02_11h24min09sec_1000_samples_concatenated" \
          "_singular_values_barabasi_albert_graph.json"
with open(path_str + path_BA) as json_data:
    singular_values_ba = np.array(json.load(json_data))
print(computeEffectiveRanks(singular_values_ba[:N], "BA", N))

""" S1 model """
path_S1 = "2022_10_04_14h49min45sec_1000_samples_corrected_concatenated" \
          "_singular_values_s1_model.json"
with open(path_str + path_S1) as json_data:
    singular_values_s1 = np.array(json.load(json_data))
print(computeEffectiveRanks(singular_values_s1[:N], "s1", N))

xlabel = "Singular values $\\sigma$"
ylabel = "Spectral density $\\rho(\\sigma)$"
xlabel_inst = "Index $i$"
ylabel_inst = "Rescaled singular\n values $\\sigma_i/\\sigma_1$"
bar_color = "#064878"
letter_posx, letter_posy = -0.27, 1.08
fontsize_legend = 10

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                       figsize=(8, 5))
ax1.scatter(np.arange(1, N + 1, 1),
            singular_values_sbm[:N]/singular_values_sbm[0], s=10)
ax1.set_yscale('log')
plt.tick_params(axis='y', which='both', left=True,
                right=False, labelbottom=False)
plt.tick_params(axis='both', which='major')
ax1.set_ylim([3*10**(-5), 5])
ax1.set_xlabel(xlabel_inst)
ax1.set_ylabel(ylabel_inst, labelpad=10)
ax1.set_title("SBM", fontsize=fontsize_legend, pad=-12)
ax1.text(letter_posx, letter_posy, "a", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)
ax1.minorticks_off()
ticks = ax1.get_xticks()
ticks[ticks.tolist().index(0)] = 1
ax1.set_xticks(ticks[ticks > 0])
ax1.set_xlim([-50, 1050])

weights = np.ones_like(singular_values_sbm)/float(len(singular_values_sbm))
ax4.hist(singular_values_sbm, bins=nb_bins,
         color=bar_color, edgecolor=None,
         linewidth=1, weights=weights)
plt.tick_params(axis='both', which='major')
ax4.set_xlabel(xlabel)
ax4.set_ylabel(ylabel, labelpad=10)
# ax1.set_yscale('log')
ax4.set_title("SBM", fontsize=fontsize_legend, pad=-12)
ax4.text(letter_posx, letter_posy, "d", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax4.transAxes)
ax4.set_xticks([0, 100, 200, 300])

ax2.scatter(np.arange(1, N + 1, 1),
            singular_values_ba[:N]/singular_values_ba[0], s=10)
ax2.set_yscale('log')
plt.tick_params(axis='y', which='both', left=True,
                right=False, labelbottom=False)
plt.tick_params(axis='both', which='major')
ax2.set_ylim([3*10**(-3), 3])
ax2.set_xlabel(xlabel_inst)
# ax2.set_ylabel(ylabel_inst, labelpad=10)
ax2.set_title("BA", fontsize=fontsize_legend, pad=-12)
ax2.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)
ax2.minorticks_off()
ticks = ax2.get_xticks()
ticks[ticks.tolist().index(0)] = 1
ax2.set_xticks(ticks[ticks > 0])
ax2.set_xlim([-50, 1050])


weights = np.ones_like(singular_values_ba)/float(len(singular_values_ba))
ax5.hist(singular_values_ba, bins=nb_bins,
         color=bar_color, edgecolor=None,
         linewidth=1, weights=weights)
plt.tick_params(axis='both', which='major')
ax5.set_xlabel(xlabel)
# ax5.set_ylabel(ylabel, labelpad=10)
# ax1.set_yscale('log')
ax5.set_title("BA", fontsize=fontsize_legend, pad=-12)
ax5.text(letter_posx, letter_posy, "e", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax5.transAxes)
ax5.set_ylim([0, 0.009])
ax5.set_xticks([0, 5, 10, 15])

ax3.scatter(np.arange(1, N + 1, 1),
            singular_values_s1[:N]/singular_values_s1[0], s=10)
ax3.set_yscale('log')
plt.tick_params(axis='y', which='both', left=True,
                right=False, labelbottom=False)
plt.tick_params(axis='both', which='major')
ax3.set_ylim([3*10**(-5), 5])
ax3.set_xlabel(xlabel_inst)
# ax3.set_ylabel(ylabel_inst, labelpad=10)
ax3.set_title("S$^1$", fontsize=fontsize_legend, pad=-12)
ax3.text(letter_posx, letter_posy, "c", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax3.transAxes)
ax3.minorticks_off()
ticks = ax3.get_xticks()
ticks[ticks.tolist().index(0)] = 1
ax3.set_xticks(ticks[ticks > 0])
ax3.set_xlim([-50, 1050])

weights = np.ones_like(singular_values_s1)/float(len(singular_values_s1))
ax6.hist(singular_values_s1, bins=nb_bins,
         color=bar_color, edgecolor=None,
         linewidth=1, weights=weights)
ax6.tick_params(axis='both', which='major')
ax6.set_xlabel(xlabel)
# ax6.set_ylabel(ylabel, labelpad=10)
# ax1.set_yscale('log')
ax6.set_title("S$^1$", fontsize=fontsize_legend, pad=-12)
ax6.text(letter_posx, letter_posy, "f", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax6.transAxes)
ax6.set_ylim([0, 0.009])
ax6.set_xticks([0, 10, 20, 30])

plt.tight_layout()
plt.show()
