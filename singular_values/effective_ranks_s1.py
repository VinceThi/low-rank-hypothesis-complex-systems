# -*- coding: utf-8 -*-\\
# @author: Vincent Thibeault

from singular_values.compute_effective_ranks import *
# from singular_values.marchenko_pastur_pdf import marchenko_pastur_generator
from graphs.generate_s1_random_graph import *
from graphs.generate_truncated_pareto import truncated_pareto
import json
import numpy as np
from numpy.linalg import norm
from plots.plot_singular_values import plot_singular_values
from tqdm.auto import tqdm
from plots.config_rcparams import *

plot_histogram = False
plot_scree = False
plot_singvals_W_EW_R = False
plot_norms = False
path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/singular_values/properties/" \
           "singular_values_random_graphs/"

""" Random graph parameters """
graph_str = "s1"
directed = True
selfloops = True
expected = True
N = 1000
nb_graphs = 1  # 50    # 1000
kappa_in_min = 5
kappa_in_max = 100
gamma_in = 2.5
kappa_in = truncated_pareto(N, kappa_in_min, kappa_in_max, gamma_in)
kappa_out_min = 3
kappa_out_max = 50
gamma_out = 2
kappa_out = truncated_pareto(N, kappa_out_min,
                             kappa_out_max, gamma_out)
kappa_in, kappa_out = \
    generate_nonnegative_arrays_with_same_average(kappa_in, kappa_out)

theta = 2*np.pi*uniform.rvs(size=N)


""" Get effective ranks vs. norm ratio (through temperature variation) """
# ^ Inverse temperature, controls the clustering
#  (1, inf) -> lower to higher clustering
min_temperature = 0.01  # 1.1
max_temperature = 0.9  # 10
nb_temperature = 10
temperature_array = np.linspace(min_temperature, max_temperature,
                                nb_temperature)

norm_choice = 2   # 'fro': frobenius norm, 2: spectral norm
norm_W = np.zeros((nb_graphs, nb_temperature))
norm_EW = np.zeros(nb_temperature)
norm_R = np.zeros((nb_graphs, nb_temperature))

rank = np.zeros((nb_graphs, nb_temperature))
thrank = np.zeros((nb_graphs, nb_temperature))
shrank = np.zeros((nb_graphs, nb_temperature))
erank = np.zeros((nb_graphs, nb_temperature))
elbow = np.zeros((nb_graphs, nb_temperature))
energy = np.zeros((nb_graphs, nb_temperature))
srank = np.zeros((nb_graphs, nb_temperature))
nrank = np.zeros((nb_graphs, nb_temperature))
for j, temperature in enumerate(tqdm(temperature_array, position=0,
                                desc="Temperature",
                                leave=True, ncols=80)):
    singularValues = np.array([])
    singularValues_R = np.array([])
    # R_list = []

    for i in tqdm(range(0, nb_graphs), position=1, desc="Graph", leave=False,
                  ncols=80):
        W, EW = s1_model(1 / temperature, kappa_in, kappa_out, theta,
                         selfloops=selfloops, directed=directed, expected=True)
        norm_W[i, j] = norm(W, ord=norm_choice)
        R = W - EW
        # R_list.append(R)
        norm_R[i, j] = norm(R, ord=norm_choice)
        # plt.hist(np.sum(EW, axis=1), bins=100)
        # plt.hist(np.sum(EW, axis=0), bins=100)
        # plt.show()

        singularValues_instance = svdvals(W)
        singularValues = np.concatenate((singularValues,
                                         singularValues_instance))
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

        if plot_scree:
            plot_singular_values(singularValues_instance, effective_ranks=True)

        if plot_singvals_W_EW_R:
            plt.figure(figsize=(4, 4))
            indices = np.arange(1, N + 1, 1)
            plt.scatter(indices, singularValues_instance, label="$W$", s=20)
            plt.scatter(indices, svdvals(EW), label="$\\langle W \\rangle$",
                        s=3)
            plt.scatter(indices, svdvals(R), label="$R$", s=3)
            plt.legend(loc=1)
            plt.xlabel("Index $i$")
            plt.ylabel("Singular values $\\sigma_i$")
            plt.show()

    norm_EW[j] = norm(EW, ord=norm_choice)

    if plot_histogram:
        # var = np.var(R_list)  # np.sum(EW*(np.ones((N, N)) - EW))
        nb_bins = 500
        plt.hist(singularValues**2/N,  bins=nb_bins,
                 color=deep[0], edgecolor=None,
                 linewidth=1, density=True)
        plt.hist(singularValues_R**2/N, bins=nb_bins//10,
                 color=deep[1], edgecolor=None,
                 linewidth=1, density=True)
        plt.hist(svdvals(EW)**2/N, bins=nb_bins,
                 color=deep[2], edgecolor=None,
                 linewidth=1, density=True)
        # plt.plot(marchenko_pastur_generator(var, 1, 1000),
        #          linewidth=2, color=deep[9], label="Marchenko-Pastur pdf")
        plt.tick_params(axis='both', which='major')
        plt.xlabel("Singular values $\\sigma^2$")
        plt.ylabel("Spectral density $\\rho(\\sigma^2)$", labelpad=20)
        plt.tight_layout()
        plt.show()


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
    plt.scatter(temperature_array, np.mean(norm_R, axis=0), label="Noise", s=3)
    plt.scatter(temperature_array, norm_EW, label="Expected", s=3)
    plt.xlabel("Temperature")
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
# "Temperature"
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
                             "directed": directed, "selfloops": selfloops,
                             "N": N, "kappa_in_min": kappa_in_min,
                             "kappa_in_max": kappa_in_max,
                             "gamma_in": gamma_in,
                             "kappa_out_min": kappa_out_min,
                             "kappa_out_max": kappa_out_max,
                             "gamma_out": gamma_out,
                             "kappa_in": kappa_in,
                             "kappa_out": kappa_out,
                             "temperature_array": temperature_array.tolist(),
                             "norm_ratio": norm_ratio.tolist(),
                             "nb_samples (nb_graphs)": nb_graphs,
                             "norm_choice": norm_choice
                             }

    fig.savefig(path + f'{timestr}_{file}_effective_ranks_vs_norm_ratio'
                       f'_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_effective_ranks_vs_norm_ratio'
                       f'_{graph_str}.png')

    with open(path + f'{timestr}_{file}_norm_W_{graph_str}.json', 'w')\
            as outfile:
        json.dump(norm_W.tolist(), outfile)

    with open(path + f'{timestr}_{file}_norm_EW_{graph_str}.json', 'w') \
            as outfile:
        json.dump(norm_EW.tolist(), outfile)

    with open(path + f'{timestr}_{file}_norm_R_{graph_str}.json', 'w')\
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

    with open(path+f'{timestr}_{file}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
