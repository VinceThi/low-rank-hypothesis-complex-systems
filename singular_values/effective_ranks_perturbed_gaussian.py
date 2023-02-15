# -*- coding: utf-8 -*-\\
# @author: Vincent Thibeault

from singular_values.compute_effective_ranks import *
from numpy.linalg import norm
import json
import numpy as np
from tqdm.auto import tqdm
from plots.config_rcparams import *

plot_histogram = False
path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/singular_values/properties/" \
           "singular_values_random_graphs/"

""" Random graph parameters """
graph_str = "perturbed_gaussian"
N = 1000
nb_graphs = 1000    # 1000
rank = 5
prng = np.random.RandomState(1234567890)
# First try
# L = prng.uniform(0, 1/np.sqrt(N), (N, rank))  # max_strength = 4
# R = prng.normal(0, 1, (rank, N))    # prng.uniform(0, 1, (rank, N))   #

# Second try
# R = prng.normal(0, 0.2, (rank, N))   # max_strength = 2
# L = R.T/np.sqrt(N)

# Third try
L = prng.normal(0, 1/np.sqrt(N), (N, rank))
R = prng.normal(0, 0.5, (rank, N))
EW = L@R
norm_LR = np.linalg.norm(L@R)

""" Get effective ranks vs. noise strength"""
min_strength = 0.01
max_strength = 2.5
nb_strength = 50
strength_array = np.linspace(min_strength, max_strength, nb_strength)

norm_EW = np.zeros(nb_strength)
norm_R = np.zeros((nb_graphs, nb_strength))

rank = np.zeros((nb_graphs, nb_strength))
thrank = np.zeros((nb_graphs, nb_strength))
shrank = np.zeros((nb_graphs, nb_strength))
erank = np.zeros((nb_graphs, nb_strength))
elbow = np.zeros((nb_graphs, nb_strength))
energy = np.zeros((nb_graphs, nb_strength))
srank = np.zeros((nb_graphs, nb_strength))
nrank = np.zeros((nb_graphs, nb_strength))
frobenius_norm_noise = np.zeros((nb_graphs, nb_strength))
for j, g in enumerate(tqdm(strength_array, position=0, desc="Strength",
                           leave=True, ncols=80)):
    singularValues = np.array([])
    var = g**2/N
    for i in tqdm(range(0, nb_graphs), position=1, desc="Graph", leave=False,
                  ncols=80):
        gaussian = np.random.normal(0, np.sqrt(var), (N, N))
        W = EW + gaussian

        norm_R[i, j] = norm(W - EW)

        singularValues_instance = svdvals(W)
        singularValues = np.concatenate((singularValues,
                                         singularValues_instance))
        rank[i, j] = computeRank(singularValues_instance)
        thrank[i, j] = computeOptimalThreshold(singularValues_instance)
        shrank[i, j] = computeOptimalShrinkage(singularValues_instance)
        erank[i, j] = computeERank(singularValues_instance)
        elbow[i, j] = findEffectiveRankElbow(singularValues_instance)
        energy[i, j] = computeEffectiveRankEnergyRatio(singularValues_instance,
                                                       threshold=0.5)
        srank[i, j] = computeStableRank(singularValues_instance)
        nrank[i, j] = computeNuclearRank(singularValues_instance)
        frobenius_norm_noise[i, j] = np.linalg.norm(gaussian)

    if plot_histogram:
        nb_bins = 1000
        bar_color = "#064878"
        weights = np.ones_like(singularValues) / float(len(singularValues))
        plt.hist(singularValues,  bins=nb_bins,
                 color=bar_color, edgecolor=None,
                 linewidth=1, weights=weights)
        plt.tick_params(axis='both', which='major')
        plt.xlabel("Singular values $\\sigma$")
        plt.ylabel("Spectral density $\\rho(\\sigma)$", labelpad=20)
        plt.tight_layout()
        plt.show()

    norm_EW[j] = norm(EW)

mean_rank = np.mean(rank, axis=0)/N
mean_thrank = np.mean(thrank, axis=0)/N
mean_shrank = np.mean(shrank, axis=0)/N
mean_erank = np.mean(erank, axis=0)/N
mean_elbow = np.mean(elbow, axis=0)/N
mean_energy = np.mean(energy, axis=0)/N
mean_srank = np.mean(srank, axis=0)/N
mean_nrank = np.mean(nrank, axis=0)/N

std_rank = np.std(rank, axis=0)/N
std_thrank = np.std(thrank, axis=0)/N
std_shrank = np.std(shrank, axis=0)/N
std_erank = np.std(erank, axis=0)/N
std_elbow = np.std(elbow, axis=0)/N
std_energy = np.std(energy, axis=0)/N
std_srank = np.std(srank, axis=0)/N
std_nrank = np.std(nrank, axis=0)/N        

norm_ratio = np.mean(norm_R, axis=0)/norm_EW


xlabel = "$\\langle ||R||_F \\rangle\,/\,||\\langle W \\rangle||_F$"
# "Noise strength $g$"
color = dark_grey
alpha = 0.2
s = 3
letter_posx, letter_posy = -0.27, 1.08
fontsize_legend = 10
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) =\
    plt.subplots(nrows=2, ncols=4, figsize=(10, 5))

ax1.scatter(norm_ratio, mean_srank, color=color, s=s)
ax1.fill_between(norm_ratio, mean_srank-std_srank,
                 mean_srank+std_srank, color=color, alpha=alpha)
ax1.tick_params(axis='both', which='major')
ax1.set_ylabel("srank/N")
ax1.set_xlabel(xlabel)
ax1.text(letter_posx, letter_posy, "a", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax1.transAxes)


ax2.scatter(norm_ratio, mean_nrank, color=color, s=s)
ax2.fill_between(norm_ratio, mean_nrank-std_nrank,
                 mean_nrank+std_nrank, color=color, alpha=alpha)
ax2.tick_params(axis='both', which='major')
ax2.set_ylabel("nrank/N")
ax2.set_xlabel(xlabel)
ax2.text(letter_posx, letter_posy, "b", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax2.transAxes)

ax3.scatter(norm_ratio, mean_elbow, color=color, s=s)
ax3.fill_between(norm_ratio, mean_elbow-std_elbow,
                 mean_elbow+std_elbow, color=color, alpha=alpha)
ax3.tick_params(axis='both', which='major')
ax3.set_ylabel("elbow/N")
ax3.set_xlabel(xlabel)
ax3.text(letter_posx, letter_posy, "c", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax3.transAxes)


ax4.scatter(norm_ratio, mean_energy, color=color, s=s)
ax4.fill_between(norm_ratio, mean_energy-std_energy,
                 mean_energy+std_energy, color=color, alpha=alpha)
ax4.tick_params(axis='both', which='major')
ax4.set_ylabel("energy/N")
ax4.set_xlabel(xlabel)
ax4.text(letter_posx, letter_posy, "d", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax4.transAxes)

ax5.scatter(norm_ratio, mean_thrank,  color=color, s=s)
ax5.fill_between(norm_ratio, mean_thrank-std_thrank,
                 mean_thrank+std_thrank, color=color, alpha=alpha)
ax5.tick_params(axis='both', which='major')
ax5.set_ylabel("thrank/N")
ax5.set_xlabel(xlabel)
ax5.set_xlabel(xlabel)
ax5.text(letter_posx, letter_posy, "e", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax5.transAxes)

ax6.scatter(norm_ratio, mean_shrank, color=color, s=s)
ax6.fill_between(norm_ratio, mean_shrank-std_shrank,
                 mean_shrank+std_shrank, color=color, alpha=alpha)
ax6.tick_params(axis='both', which='major')
ax6.set_ylabel("shrank/N")
ax6.set_xlabel(xlabel)
ax6.text(letter_posx, letter_posy, "f", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax6.transAxes)


ax7.scatter(norm_ratio, mean_erank, color=color, s=s)
ax7.fill_between(norm_ratio, mean_erank-std_erank,
                 mean_erank+std_erank, color=color, alpha=alpha)
ax7.tick_params(axis='both', which='major')
ax7.set_ylabel("erank/N")
ax7.set_xlabel(xlabel)
ax7.text(letter_posx, letter_posy, "g", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax7.transAxes)

ax8.scatter(norm_ratio, mean_rank, color=color, s=s)
ax8.fill_between(norm_ratio, mean_rank-std_rank,
                 mean_rank+std_rank, color=color, alpha=alpha)
ax8.tick_params(axis='both', which='major')
ax8.set_ylabel("rank/N")
ax8.set_xlabel(xlabel)
ax8.text(letter_posx, letter_posy, "h", fontweight="bold",
         horizontalalignment="center", verticalalignment="top",
         transform=ax8.transAxes)
# ax8.scatter(strength_array, mean_fronorm_noise, color=color, s=s)
# ax8.plot(strength_array, strength_array*np.sqrt(N), color=color, linewidth=1,
#          label="$g\sqrt{N}$")
# ax8.fill_between(strength_array, mean_fronorm_noise-std_fronorm_noise,
#                  mean_fronorm_noise+std_fronorm_noise,
#                  color=color, alpha=alpha)
# ax8.plot(strength_array, norm_LR*np.ones(len(strength_array)),
#          linestyle="--")
# ax8.fill_between(strength_array, mean_rank-std_rank,
#                  mean_rank+std_rank, color=color, alpha=alpha)
# ax8.tick_params(axis='both', which='major')
# ax8.set_ylabel("$\\langle ||X||_F \\rangle$")
# ax8.set_xlabel(xlabel)
# ax8.text(letter_posx, letter_posy, "h", fontweight="bold",
#          horizontalalignment="center", verticalalignment="top",
#          transform=ax8.transAxes)
# ax8.legend(loc=2, fontsize=10)

plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters,"
                       " the data, and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    path = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "singular_values/properties/" + graph_str + "/"
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    parameters_dictionary = {"graph_str": graph_str,
                             "L": L.tolist(), "R": R.tolist(), "N": N,
                             "strength_array": strength_array.tolist(),
                             "nb_samples (nb_graphs)": nb_graphs,
                             "frobenius norm of L@R": norm_LR
                             }

    fig.savefig(path + f'{timestr}_{file}_effective_ranks_histogram'
                       f'_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_effective_ranks_histogram'
                       f'_{graph_str}.png')

    with open(path + f'{timestr}_{file}_norm_EW_{graph_str}.json', 'w') \
            as outfile:
        json.dump(norm_EW.tolist(), outfile)

    with open(path + f'{timestr}_{file}_norm_R_{graph_str}.json', 'w') \
            as outfile:
        json.dump(norm_R.tolist(), outfile)

    with open(path + f'{timestr}_{file}_rank_{graph_str}.json', 'w') \
            as outfile:
        json.dump(rank.tolist(), outfile)

    with open(path + f'{timestr}_{file}_thrank_{graph_str}.json', 'w') \
            as outfile:
        json.dump(thrank.tolist(), outfile)

    with open(path + f'{timestr}_{file}_shrank_{graph_str}.json', 'w') \
            as outfile:
        json.dump(shrank.tolist(), outfile)

    with open(path + f'{timestr}_{file}_elbow_{graph_str}.json', 'w') \
            as outfile:
        json.dump(elbow.tolist(), outfile)

    with open(path + f'{timestr}_{file}_erank_{graph_str}.json', 'w') \
            as outfile:
        json.dump(erank.tolist(), outfile)

    with open(path + f'{timestr}_{file}_energy_{graph_str}.json', 'w') \
            as outfile:
        json.dump(energy.tolist(), outfile)

    with open(path + f'{timestr}_{file}_srank_{graph_str}.json', 'w') \
            as outfile:
        json.dump(srank.tolist(), outfile)

    with open(path + f'{timestr}_{file}_nrank_{graph_str}.json', 'w') \
            as outfile:
        json.dump(nrank.tolist(), outfile)

    with open(path + f'{timestr}_{file}_{graph_str}'
                     f'_parameters_dictionary.json', 'w') as outfile:
        json.dump(parameters_dictionary, outfile)
