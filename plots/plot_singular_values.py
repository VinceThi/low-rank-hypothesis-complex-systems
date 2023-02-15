# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault and Antoine Allard <antoineallard.info>
from itertools import cycle
import numpy as np
from scipy.linalg import svdvals
import os
import pandas as pd
import seaborn as sns
import tabulate
from plots.config_rcparams import *
import time
import json
import tkinter.simpledialog
from tkinter import messagebox
import networkx as nx
from tqdm import tqdm
from singular_values.compute_effective_ranks import computeERank, \
    computeStableRank, findEffectiveRankElbow, computeRank,\
    computeEffectiveRankEnergyRatio, \
    computeOptimalShrinkage, computeOptimalThreshold, computeNuclearRank


def plot_singular_values(singularValues,
                         effective_ranks=True,
                         cum_explained_var=False,
                         ysemilog=False):
    """ Scree plot of the singular values """

    singularValues = np.sort(singularValues)[::-1]

    if effective_ranks:
        rank = computeRank(singularValues)
        stableRank = computeStableRank(singularValues)
        nuclearRank = computeNuclearRank(singularValues)
        optimalThreshold = computeOptimalThreshold(singularValues)
        norm_str = 'frobenius'
        optimalShrinkage = computeOptimalShrinkage(singularValues)
        elbowRank = findEffectiveRankElbow(singularValues)
        erank = computeERank(singularValues)
        threshold = 0.9
        percentage_threshold = "%.0f" % (threshold*100)
        energyRatio = computeEffectiveRankEnergyRatio(singularValues,
                                                      threshold=threshold)
        header = ['Rank', 'Optimal threshold',
                  'Optimal shrinkage',
                  'Erank', 'Elbow', 'Energy ratio', 'Stable rank',
                  'Nuclear Rank']
        properties = [[rank, optimalThreshold, optimalShrinkage,
                       erank, elbowRank, energyRatio, stableRank, nuclearRank]]
        print("\n\n\n\n", tabulate.tabulate(properties, headers=header))

    if cum_explained_var:
        cumulative_explained_variance = []
        for r in range(1, len(singularValues) + 1):
            # explained_variance.append(S[r]**2/np.sum(S**2))
            cumulative_explained_variance.append(
                np.sum(singularValues[0:r]**2) / np.sum(singularValues**2))

    if cum_explained_var:
        plt.figure(figsize=(6, 4))
        ax1 = plt.subplot(121)
        if effective_ranks:
            plt.axvline(x=rank, linestyle="--",
                        color=reduced_grey, label="Rank")
            plt.axvline(x=stableRank, linestyle="--",
                        color=deep[0],
                        label="Stable rank")
            plt.axvline(x=nuclearRank, linestyle="--",
                        color=deep[6],
                        label="Nuclear rank")
            plt.axvline(x=elbowRank, linestyle="--",
                        color=deep[1],
                        label="Elbow position")
            plt.axvline(x=erank, linestyle="--",
                        color=deep[2],
                        label="erank")
            plt.axvline(x=energyRatio, linestyle="--",
                        color=deep[3],
                        label=f"Energy ratio ({percentage_threshold}%)")
            plt.axvline(x=optimalThreshold, linestyle="--",
                        color=deep[4],
                        label=f"Optimal threshold")
            plt.axvline(x=optimalShrinkage, linestyle="--",
                        color=deep[5],
                        label=f"Optimal shrinkage ({norm_str})")
        ax1.scatter(np.arange(1, len(singularValues) + 1, 1),
                    singularValues/singularValues[0], s=10)
        plt.ylabel("Rescaled singular\n values $\\sigma_i/\\sigma_1$")
        plt.xlabel("Index $i$")
        plt.legend(loc=1, fontsize=fontsize_legend)
        plt.tight_layout()
        ticks = ax1.get_xticks()
        ticks[ticks.tolist().index(0)] = 1
        plt.xticks(ticks[ticks > 0])
        if ysemilog:
            plt.ylim([0.01*np.min(singularValues), 1.5])
            ax1.set_yscale('log')
            plt.tick_params(axis='y', which='both', left=True,
                            right=False, labelbottom=False)

        ax2 = plt.subplot(122)
        plt.scatter(np.arange(1, len(singularValues) + 1, 1),
                    cumulative_explained_variance, zorder=1)
        plt.xlabel("Number of singular values $n$", fontsize=12)
        plt.ylabel("Cumulative explained variance "
                   "$\\sum_{j=1}^n\\sigma_j^2/\\sum_{j=1}^N \\sigma_j^2$")
        ticks = ax2.get_xticks()
        ticks[ticks.tolist().index(0)] = 1
        plt.xticks(ticks[ticks > 0])
        plt.ylim([-0.05, 1.05])
        plt.tight_layout()
        plt.show()

    else:
        # fig, ax = plt.subplots(1, figsize=(3, 2.8))
        fig, ax = plt.subplots(1, figsize=(6, 3.8))
        if effective_ranks:
            plt.axvline(x=rank, linestyle="--",
                        color=reduced_grey, label="Rank")
            plt.axvline(x=stableRank, linestyle="--",
                        color=deep[0],
                        label="Stable rank")
            plt.axvline(x=nuclearRank, linestyle="--",
                        color=deep[6],
                        label="Nuclear rank")
            plt.axvline(x=elbowRank, linestyle="--",
                        color=deep[1],
                        label="Elbow rank")
            plt.axvline(x=erank, linestyle="--",
                        color=deep[2],
                        label="erank")
            plt.axvline(x=energyRatio, linestyle="--",
                        color=deep[3],
                        label=f"Energy ratio ({percentage_threshold}%)")
            plt.axvline(x=optimalShrinkage, linestyle="--",
                        color=deep[4],
                        label=f"Optimal shrinkage ({norm_str})")
            plt.axvline(x=optimalThreshold, linestyle="--",
                        color=deep[5],
                        label=f"Optimal threshold")
        # ax.scatter(np.arange(1, len(singularValues) + 1, 1),
        #            singularValues / singularValues[0], s=10)
        # plt.ylabel("Rescaled singular\n values $\\sigma_i/\\sigma_1$")
        ax.scatter(np.arange(1, len(singularValues) + 1, 1),
                   singularValues, s=10)
        plt.ylabel("Singular\n values $\\sigma_i$")
        plt.xlabel("Index $i$")
        plt.legend(loc=1, fontsize=8)
        ticks = ax.get_xticks()
        ticks[ticks.tolist().index(0)] = 1
        ticks = [i for i in ticks
                 if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
        plt.xticks(ticks)
        if ysemilog:
            plt.ylim([0.01*np.min(singularValues), 1.5])
            ax.set_yscale('log')
            plt.tick_params(axis='y', which='both', left=True,
                            right=False, labelbottom=False)
        plt.show()


def plot_singular_values_histogram(singularValues,
                                   nb_bins=1000,
                                   bar_color="#064878",
                                   xlabel="Rescaled singular"
                                          " values $\\sigma$",
                                   ylabel="Spectral density $\\rho(\\sigma)$"):
    # axvline_color="#ef8a62"
    # for i, sigma in enumerate(singularValues):
    #     c = axvline_color
    #     if isinstance(axvline_color, list):
    #         c = axvline_color[i]
    #     ax.axvline(sigma, color=c, zorder=0)
    plt.figure(figsize=(6, 4))
    weights = np.ones_like(singularValues) / float(len(singularValues))
    plt.hist(singularValues/np.max(singularValues), bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    plt.tick_params(axis='both', which='major')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=20)
    plt.tight_layout()
    plt.show()


def plot_singular_values_histogram_random_networks(random_graph_generator,
                                                   random_graph_args,
                                                   nb_networks=1000,
                                                   nb_bins=1000,
                                                   bar_color="#064878",
                                                   xlabel="Singular"
                                                          " values $\\sigma$",
                                                   ylabel="Spectral density"
                                                          " $\\rho(\\sigma)$"):
    # from singular_values.marchenko_pastur_pdf import marchenko_pastur_pdf
    graph_str = random_graph_generator.__name__
    fig = plt.figure(figsize=(6, 4))
    singularValues = np.array([])
    for _ in tqdm(range(0, nb_networks)):
        if graph_str in ["tenpy_random_matrix", "perturbed_gaussian"]:
            W = random_graph_generator(*random_graph_args)
        else:
            W = nx.to_numpy_array(random_graph_generator(*random_graph_args))
        singularValues_instance = svdvals(W)
        singularValues = np.concatenate((singularValues,
                                         singularValues_instance))
    weights = np.ones_like(singularValues) / float(len(singularValues))
    # N = random_graph_args[0]
    # p = 0.1
    # var = N*p*(1-p)
    # x = np.linspace(0.01, 2*np.sqrt(var) - 0.0001, 1000)
    # plt.plot(x, marchenko_pastur_pdf(x, var, 1), "#111111")
    # plt.plot(x, np.sqrt((4*var - x)/x)/(2*np.pi*var))
    plt.hist(singularValues, bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    plt.tick_params(axis='both', which='major')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=20)
    plt.tight_layout()
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
        parameters_dictionary = {"graph_str": graph_str,
                                 "args graph gen": random_graph_args,
                                 "nb_samples (nb_networks)": nb_networks,
                                 "nb_bins histogram": nb_bins}

        fig.savefig(path + f'{timestr}_{file}_singular_values_histogram'
                           f'_{graph_str}.pdf')
        fig.savefig(path + f'{timestr}_{file}_singular_values_histogram'
                           f'_{graph_str}.png')

        with open(path + f'{timestr}_{file}_concatenated_singular_values'
                         f'_{graph_str}.json', 'w') \
                as outfile:
            json.dump(singularValues.tolist(), outfile)
        with open(path + f'{timestr}_{file}_weights_for_histogram'
                         f'_{graph_str}.json', 'w') \
                as outfile:
            json.dump(weights.tolist(), outfile)
        with open(path + f'{timestr}_{file}_singular_values_histogram'
                         f'_{graph_str}_parameters_dictionary.json',
                  'w') as outfile:
            json.dump(parameters_dictionary, outfile)


def plot_singular_values_scree_random_networks(random_graph_generator,
                                               random_graph_args,
                                               nb_vertices,
                                               nb_networks=1000,
                                               xlabel="Index $i$",
                                               ylabel="Average rescaled "
                                                      "singular\n values "
                                                      "$\\sigma_i/\\sigma_1$"):
    graph_str = random_graph_generator.__name__
    fig = plt.figure(figsize=(6, 4))
    singularValues = np.zeros((nb_networks, nb_vertices))
    for i in tqdm(range(0, nb_networks)):
        if graph_str in ["tenpy_random_matrix", "perturbed_gaussian"]:
            W = random_graph_generator(*random_graph_args)
        else:
            W = nx.to_numpy_array(random_graph_generator(*random_graph_args))
        singularValues_instance = svdvals(W)
        singularValues[i, :] = singularValues_instance
    mean_singularValues = np.mean(singularValues, axis=0)
    std_singularValues = np.std(singularValues, axis=0)
    plt.scatter(np.arange(1, len(singularValues_instance) + 1, 1),
                mean_singularValues, s=3, color=deep[0])
    plt.fill_between(np.arange(1, len(singularValues_instance) + 1, 1),
                     mean_singularValues - std_singularValues,
                     mean_singularValues + std_singularValues,
                     color=deep[0], alpha=0.2)
    plt.tick_params(axis='both', which='major')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=20)
    plt.tight_layout()
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
        parameters_dictionary = {"graph_str": graph_str,
                                 "args graph gen": random_graph_args,
                                 "nb_vertices": nb_vertices,
                                 "nb_samples (nb_networks)": nb_networks}

        fig.savefig(path + f'{timestr}_{file}_singular_values_histogram'
                           f'_{graph_str}.pdf')
        fig.savefig(path + f'{timestr}_{file}_singular_values_histogram'
                           f'_{graph_str}.png')

        with open(path + f'{timestr}_{file}_concatenated_singular_values'
                         f'_{graph_str}.json', 'w') \
                as outfile:
            json.dump(singularValues.tolist(), outfile)
        with open(path + f'{timestr}_{file}_singular_values_histogram'
                         f'_{graph_str}_parameters_dictionary.json',
                  'w') as outfile:
            json.dump(parameters_dictionary, outfile)


def plot_singular_values_given_effective_ranks(singularValues, effectiveRanks):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(2*11.7, 8.3), dpi=75)
    colors = cycle(sns.color_palette('deep', 8))

    line, = ax1.plot(range(1, np.int(effectiveRanks['rank']+1)),
                     singularValues, linestyle='None',
                     color=next(colors), marker='o')
    # header =['Name', 'Size', 'Rank', 'Optimal threshold','Optimal shrinkage',
    #           'Erank', 'Elbow', 'Energy ratio', 'Stable rank']
    ax1.axvline(x=effectiveRanks['rank'], linestyle="--",
                color=reduced_grey, label="Rank")
    ax1.axvline(x=effectiveRanks['Stable Rank'], linestyle="-.",
                color=deep[0],
                label="Stable rank")
    ax1.axvline(x=effectiveRanks['Elbow'], linestyle=":",
                color=deep[1],
                label="Elbow position")
    ax1.axvline(x=effectiveRanks['Erank'], linestyle="--",
                color=deep[2],
                label="erank")
    ax1.axvline(x=effectiveRanks['Energy ratio'], linestyle="-.",
                color=deep[3],
                label=f"Energy ratio")
    ax1.axvline(x=effectiveRanks['Optimal shrinkage'], linestyle=":",
                color=deep[4],
                label=f"Optimal shrinkage")
    ax1.axvline(x=effectiveRanks['Optimal threshold'], linestyle="--",
                color=deep[5],
                label=f"Optimal threshold")

    normalizedCumulSquaredSingularValues = np.cumsum(np.square(singularValues))
    normalizedCumulSquaredSingularValues /= \
        normalizedCumulSquaredSingularValues[-1]
    ax2.plot(range(1, np.int(effectiveRanks['rank']+1)),
             normalizedCumulSquaredSingularValues, linestyle='None',
             color=line.get_color(), marker='o')

    ax1.set_xlabel(r'index $i$')
    ax1.set_ylabel(r'singular value $\sigma_i$')
    ax2.set_xlabel(r'number of singular values $n$')
    ax2.set_ylabel(r'energy ratio $\sum_{i=1}^n '
                   r'\sigma_i^2 / \sum_{l=1}^r \sigma_l^2$')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax1.tick_params(length=8)
    ax2.tick_params(length=8)

    ax2.set_ylim(bottom=-0.05, top=1.05)

    ax1.legend(loc='upper right', frameon=False, fontsize='x-small')

    return fig


def main():

    effectiveRanksFilename = 'singular_values/properties/effective_ranks.txt'
    header = open(effectiveRanksFilename, 'r')\
        .readline().replace('#', ' ').split()
    effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                     comment="#", delimiter=r"\s+")
    effectiveRanksDF.set_index('name', inplace=True)

    for networkName in effectiveRanksDF.index:

        singularValuesFilename = 'properties/singular_values/' \
                                 + networkName + '_singular_values.txt'
        figureFilenamePDF = 'figures/pdf/singular_values/' \
                            + networkName + '.pdf'
        figureFilenamePNG = 'figures/png/singular_values/' \
                            + networkName + '.png'

        if not os.path.isfile(figureFilenamePNG):
            print(networkName)
            singularValues = np.loadtxt(singularValuesFilename)
            fig = plot_singular_values_given_effective_ranks(
                singularValues, effectiveRanksDF.loc[networkName])
            fig.savefig(figureFilenamePDF, bbox_inches='tight')
            fig.savefig(figureFilenamePNG, bbox_inches='tight')


if __name__ == "__main__":
    main()
