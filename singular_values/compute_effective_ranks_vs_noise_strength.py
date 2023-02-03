# -*- coding: utf-8 -*-\\
# @author: Vincent Thibeault

from singular_values.compute_effective_ranks import *
import json
import numpy as np
from tqdm import tqdm
from plots.config_rcparams import *
from graphs.random_matrix_generators import perturbed_gaussian


path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/singular_values/properties/" \
           "singular_values_random_graphs/"


N = 1000
nb_graphs = 10
rank = 5
L = np.random.uniform(0, 1/np.sqrt(N), (N, rank))
R = np.random.normal(0, 1, (rank, N))
g = 1    # Strength of the random part
var = g**2/N
args = (N, L, R, var)

min_strength = 0
max_strength = 2
nb_strength = 10
strength_array = np.linspace(min_strength, max_strength, nb_strength)

mean_effective_ranks = np.zeros((8, nb_strength))
std_effective_ranks = np.zeros((8, nb_strength))
for i, g in enumerate(strength_array):
    # graph_str = generator.__name__
    # singularValues = np.array([])
    effectiveRanks = np.zeros((8, nb_graphs))
    for j in tqdm(range(0, nb_graphs)):
        # if graph_str in ["tenpy_random_matrix", "perturbed_gaussian"]:
        #     W = generator(*args)
        # else:
        #     W = nx.to_numpy_array(generator(*args))
        var = g**2/N
        W = perturbed_gaussian(N, L, R, var)
        singularValues_instance = svdvals(W)
        effectiveRanks_instance = \
            np.array([computeRank(singularValues_instance),
                      computeOptimalThreshold(singularValues_instance),
                      computeOptimalShrinkage(singularValues_instance),
                      computeERank(singularValues_instance),
                      findEffectiveRankElbow(singularValues_instance),
                      computeEffectiveRankEnergyRatio(singularValues_instance,
                                                      threshold=0.9),
                      computeStableRank(singularValues_instance),
                      computeNuclearRank(singularValues_instance)])

        # singularValues = np.concatenate((singularValues,
        #                                  singularValues_instance))
        effectiveRanks[:, j] = effectiveRanks_instance
    mean_effective_ranks[:, i] = np.mean(effectiveRanks, axis=1)
    std_effective_ranks[:, i] = np.std(effectiveRanks, axis=1)

mean_rank = mean_effective_ranks[0, :]
mean_thrank = mean_effective_ranks[1, :]
mean_shrank = mean_effective_ranks[2, :]
mean_erank = mean_effective_ranks[3, :]
mean_elbow = mean_effective_ranks[4, :]
mean_energy = mean_effective_ranks[5, :]
mean_srank = mean_effective_ranks[6, :]
mean_nrank = mean_effective_ranks[7, :]

std_rank = std_effective_ranks[0, :]
std_thrank = std_effective_ranks[1, :]
std_shrank = std_effective_ranks[2, :]
std_erank = std_effective_ranks[3, :]
std_elbow = std_effective_ranks[4, :]
std_energy = std_effective_ranks[5, :]
std_srank = std_effective_ranks[6, :]
std_nrank = std_effective_ranks[7, :]


xlabel = "Noise strength"
color = dark_grey
alpha = 0.2
letter_posx, letter_posy = -0.27, 1.08
fontsize_legend = 10
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) =\
    plt.subplots(nrows=2, ncols=4, figsize=(10, 5))

ax1.plot(strength_array, mean_srank, color=color, linewidth=1)
ax1.fill_between(strength_array, mean_srank-std_srank,
                 mean_srank+std_srank, color=color, alpha=alpha)
ax1.tick_params(axis='both', which='major')
ax1.set_ylabel("srank")
ax1.set_xlabel(xlabel, labelpad=10)
ax1.text(letter_posx, letter_posy, "a", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)


ax2.plot(strength_array, mean_nrank, color=color, linewidth=1)
ax2.fill_between(strength_array, mean_nrank-std_nrank,
                 mean_nrank+std_nrank, color=color, alpha=alpha)
ax2.tick_params(axis='both', which='major')
ax2.set_xlabel("nrank")
ax2.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)

ax3.plot(strength_array, mean_elbow, color=color, linewidth=1)
ax3.fill_between(strength_array, mean_elbow-std_elbow,
                 mean_elbow+std_elbow, color=color, alpha=alpha)
ax3.tick_params(axis='both', which='major')
ax3.set_xlabel("elbow")
ax3.text(letter_posx, letter_posy, "c", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax3.transAxes)


ax4.plot(strength_array, mean_energy, color=color, linewidth=1)
ax4.fill_between(strength_array, mean_energy-std_energy,
                 mean_energy+std_energy, color=color, alpha=alpha)
ax4.tick_params(axis='both', which='major')
ax4.set_ylabel("energy")
ax4.text(letter_posx, letter_posy, "d", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax4.transAxes)

ax5.plot(strength_array, mean_thrank,  color=color, linewidth=1)
ax5.fill_between(strength_array, mean_thrank-std_thrank,
                 mean_thrank+std_thrank, color=color, alpha=alpha)
ax5.tick_params(axis='both', which='major')
ax5.set_ylabel("thrank")
ax5.set_xlabel(xlabel)
ax5.text(letter_posx, letter_posy, "e", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax5.transAxes)

ax6.plot(strength_array, mean_shrank, color=color, linewidth=1)
ax6.fill_between(strength_array, mean_shrank-std_shrank,
                 mean_shrank+std_shrank, color=color, alpha=alpha)
ax6.tick_params(axis='both', which='major')
ax6.set_ylabel("shrank")
ax6.text(letter_posx, letter_posy, "f", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax6.transAxes)


ax7.plot(strength_array, mean_erank, color=color, linewidth=1)
ax7.fill_between(strength_array, mean_erank-std_erank,
                 mean_erank+std_erank, color=color, alpha=alpha)
ax7.tick_params(axis='both', which='major')
ax7.set_ylabel("erank")
ax7.text(letter_posx, letter_posy, "g", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax7.transAxes)


ax8.plot(strength_array, mean_rank, color=color, linewidth=1)
ax8.fill_between(strength_array, mean_rank-std_rank,
                 mean_rank+std_rank, color=color, alpha=alpha)
ax8.tick_params(axis='both', which='major')
ax8.set_ylabel("rank")
ax8.text(letter_posx, letter_posy, "h", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax8.transAxes)

plt.show()
