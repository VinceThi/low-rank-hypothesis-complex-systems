# -*- coding: utf-8 -*-\\
# @author: Vincent Thibeault

from singular_values.compute_effective_ranks import *
from singular_values.marchenko_pastur_pdf import marchenko_pastur_generator
from numpy.linalg import norm
import json
import numpy as np
from tqdm.auto import tqdm
from plots.config_rcparams import *

plot_singvals = False
plot_histogram = False
plot_norms = False
path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/singular_values/properties/" \
           "singular_values_random_graphs/"

""" Random graph parameters """
graph_str = "perturbed_gaussian"
N = 1000
nb_graphs = 1  # 100
# rank = 5
prng1 = np.random.RandomState(1234567890)
prng2 = np.random.RandomState(1334567890)
prng3 = np.random.RandomState(1434567890)
prng4 = np.random.RandomState(1534567890)
prng5 = np.random.RandomState(1634567890)
m1 = prng1.normal(0, 1/np.sqrt(N), (N, ))
n1 = prng1.normal(0, 0.1, (N, ))
m2 = prng2.normal(0, 1/np.sqrt(N), (N, ))
n2 = prng2.normal(0.1, 0.2, (N, ))
m3 = prng3.normal(0, 1/np.sqrt(N), (N, ))
n3 = prng3.normal(0.2, 0.3, (N, ))
m4 = prng4.normal(0, 1/np.sqrt(N), (N, ))
n4 = prng4.normal(0.3, 0.4, (N, ))
m5 = prng5.normal(0, 1/np.sqrt(N), (N, ))
n5 = prng5.normal(0.4, 0.5, (N, ))
P = np.array([m1, m2, m3, m4, m5]).T
Q = np.array([n1, n2, n3, n4, n5]).T
print(norm(np.outer(n1, m1)), norm(np.outer(n2, m2)), norm(np.outer(n3, m3)),
      norm(np.outer(n4, m4)), norm(np.outer(n5, m5)))
EW = P@Q.T


""" Get effective ranks vs. noise strength"""
min_strength = 0.01
max_strength = 4
nb_strength = 10   # 30
strength_array = np.linspace(min_strength, max_strength, nb_strength)

norm_choice = 2   # 'fro': frobenius norm, 2: spectral norm
norm_W = np.zeros((nb_graphs, nb_strength))
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
for j, g in enumerate(tqdm(strength_array, position=0, desc="Strength",
                           leave=True, ncols=80)):
    singularValues = np.array([])
    singularValues_R = np.array([])
    var = g**2/N
    for i in tqdm(range(0, nb_graphs), position=1, desc="Graph", leave=False,
                  ncols=80):
        R = np.random.normal(0, np.sqrt(var), (N, N))
        norm_R[i, j] = norm(R, ord=norm_choice)
        W = EW + R
        norm_W[i, j] = norm(W, ord=norm_choice)
        singularValues_instance = svdvals(W)
        singularValues = np.concatenate((singularValues,
                                         singularValues_instance))
        singularValues_R_instance = svdvals(R)
        singularValues_R = np.concatenate((singularValues_R,
                                           singularValues_R_instance))

        rank[i, j] = computeRank(singularValues_instance)
        thrank[i, j] = computeOptimalThreshold(singularValues_instance)
        shrank[i, j] = computeOptimalShrinkage(singularValues_instance,
                                               norm="operator")
        erank[i, j] = computeERank(singularValues_instance)
        elbow[i, j] = findEffectiveRankElbow(singularValues_instance)
        energy[i, j] = computeEffectiveRankEnergyRatio(singularValues_instance,
                                                       threshold=0.5)
        srank[i, j] = computeStableRank(singularValues_instance)
        nrank[i, j] = computeNuclearRank(singularValues_instance)

        if plot_singvals:
            plt.figure(figsize=(4, 4))
            indices = np.arange(1, N+1, 1)
            plt.scatter(indices, singularValues_instance, label="$W$", s=20)
            plt.scatter(indices, svdvals(EW), label="$\\langle W \\rangle$",
                        s=3)
            plt.scatter(indices, svdvals(W - EW), label="$R$", s=3)
            plt.legend(loc=1)
            plt.xlabel("Index $i$")
            plt.ylabel("Singular values $\\sigma_i$")
            plt.show()

    if plot_histogram:
        nb_bins = 1000
        plt.hist(singularValues**2/N, bins=nb_bins,
                 color=deep[0], edgecolor=None,
                 linewidth=1, density=True)
        plt.hist(singularValues_R**2/N, bins=nb_bins//20,
                 color=deep[1], edgecolor=None,
                 linewidth=1, density=True)
        plt.plot(marchenko_pastur_generator(var, 1, 1000),
                 linewidth=2, color=deep[9], label="Marchenko-Pastur pdf")
        plt.tick_params(axis='both', which='major')
        plt.xlabel("Singular values $\\sigma^2$")
        plt.ylabel("Spectral density $\\rho(\\sigma^2)$", labelpad=20)
        plt.ylim([0, 500])
        plt.tight_layout()
        plt.show()

    norm_EW[j] = norm(EW, ord=norm_choice)

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

# norm_ratio = np.mean(norm_R, axis=0)/norm_EW
norm_ratio = np.mean(norm_R, axis=0)/np.mean(norm_W, axis=0)

if plot_norms:
    plt.figure(figsize=(4, 4))
    plt.plot(strength_array, np.mean(norm_R, axis=0), label="Noise", s=3)
    plt.plot(strength_array, norm_EW, label="Expected", s=3)
    plt.xlabel("Density")
    plt.legend()
    plt.show()

if norm_choice == 'fro':
    # xlabel = "$\\langle ||R||_F \\rangle\,/\,||\\langle W \\rangle||_F$"
    xlabel = "$\\langle ||R||_F \\rangle\,/\,\\langle ||W||_F \\rangle$"
elif norm_choice == 2:
    # xlabel = "$\\langle ||R||_2 \\rangle\,/\,||\\langle W \\rangle||_2$"
    xlabel = "$\\langle ||R||_2 \\rangle\,/\,\\langle ||W||_2 \\rangle$"
else:
    xlabel = None
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
                             "P": P.tolist(), "Q": Q.tolist(), "N": N,
                             "strength_array": strength_array.tolist(),
                             "nb_samples (nb_graphs)": nb_graphs,
                             "norm_choice": norm_choice
                             }

    fig.savefig(path + f'{timestr}_{file}_effective_ranks_histogram'
                       f'_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_effective_ranks_histogram'
                       f'_{graph_str}.png')

    with open(path + f'{timestr}_{file}_norm_W_{graph_str}.json', 'w') \
            as outfile:
        json.dump(norm_W.tolist(), outfile)

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
