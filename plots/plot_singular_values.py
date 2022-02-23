# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info> and Vincent Thibeault
from itertools import cycle
import numpy as np
import os
import pandas as pd
import seaborn as sns
from plots.config_rcparams import *
from optht import optht
from singular_values.compute_effective_ranks import computeERank, \
    computeStableRank, findElbowPosition, computeRank


def plot_singular_values(singularValues,
                         plot_effective_ranks=True,
                         plot_cum_explained_var=False):

    if plot_effective_ranks:
        numberSingularValues = len(singularValues)
        rank = computeRank(singularValues)
        stableRank = computeStableRank(singularValues)
        gavishDonohoThreshold = optht(1, sv=singularValues, sigma=None)
        elbowPosition = findElbowPosition(singularValues)
        erank = computeERank(singularValues)

        print(f"Number of singular values = {numberSingularValues}")
        print(f"Stable rank = {stableRank}")
        print(f"Gavish-Donoho threshold = {gavishDonohoThreshold}")
        print(f"Elbow position = {elbowPosition}")
        print(f"Erank = {erank}")

    if plot_cum_explained_var:
        cumulative_explained_variance = []
        for r in range(1, len(singularValues) + 1):
            # explained_variance.append(S[r]**2/np.sum(S**2))
            cumulative_explained_variance.append(
                np.sum(singularValues[0:r] ** 2) / np.sum(singularValues**2))

    if plot_cum_explained_var:
        plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(121)
        if plot_effective_ranks:
            plt.axvline(x=rank, linestyle="--",
                        color=reduced_grey, label="Rank")
            plt.axvline(x=gavishDonohoThreshold, linestyle="--",
                        color=reduced_first_community_color,
                        label="Gavish-Donoho")
            plt.axvline(x=stableRank, linestyle="--",
                        color=reduced_second_community_color,
                        label="Stable rank")
            plt.axvline(x=elbowPosition, linestyle="--",
                        color=reduced_third_community_color,
                        label="Elbow position")
            plt.axvline(x=erank, linestyle="--",
                        color=reduced_fourth_community_color,
                        label="erank")
        ax1.scatter(np.arange(1, len(singularValues) + 1, 1),
                    singularValues/singularValues[0], s=10)
        plt.ylabel("Normalized singular\n values $\\sigma_i/\\sigma_1$")
        plt.xlabel("Index $i$")
        plt.legend(loc="best", fontsize=fontsize_legend)
        plt.tight_layout()
        ticks = ax1.get_xticks()
        ticks[ticks.tolist().index(0)] = 1
        plt.xticks(ticks[ticks > 0])

        ax2 = plt.subplot(122)
        plt.scatter(np.arange(1, len(singularValues) + 1, 1),
                    cumulative_explained_variance, zorder=1)
        plt.xlabel("Number of singular values $n$", fontsize=12)
        plt.ylabel("Cumulative explained variance "
                   "$\sum_{j=1}^n\\sigma_j^2/\\sum_{j=1}^N \\sigma_j^2$")
        ticks = ax2.get_xticks()
        ticks[ticks.tolist().index(0)] = 1
        plt.xticks(ticks[ticks > 0])
        plt.ylim([-0.05, 1.05])
        plt.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots(1, figsize=(3, 2.8))
        if plot_effective_ranks:
            plt.axvline(x=rank, linestyle="--",
                        color=reduced_grey, label="Rank")
            plt.axvline(x=gavishDonohoThreshold, linestyle="--",
                        color=reduced_first_community_color,
                        label="Gavish-Donoho")
            plt.axvline(x=stableRank, linestyle="--",
                        color=reduced_second_community_color,
                        label="Stable rank")
            plt.axvline(x=elbowPosition, linestyle="--",
                        color=reduced_third_community_color,
                        label="Elbow position")
            plt.axvline(x=erank, linestyle="--",
                        color=reduced_fourth_community_color,
                        label="erank")
        ax.scatter(np.arange(1, len(singularValues) + 1, 1),
                   singularValues/singularValues[0], s=10)
        plt.ylabel("Normalized singular\n values $\\sigma_i/\\sigma_1$")
        plt.xlabel("Index $i$")
        plt.legend(loc=4, fontsize=8)
        ticks = ax.get_xticks()
        ticks[ticks.tolist().index(0)] = 1
        ticks = [i for i in ticks
                 if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
        plt.xticks(ticks)
        plt.show()


def plot_singular_values_given_effective_ranks(singularValues, effectiveRanks):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(2*11.7, 8.3), dpi=75)
    colors = cycle(sns.color_palette('deep', 8))

    line, = ax1.plot(range(1, np.int(effectiveRanks['rank']+1)),
                     singularValues, linestyle='None',
                     color=next(colors), marker='o')

    ax1.axvline(effectiveRanks['rankGD'], linestyle='--',
                color=next(colors), label=r'G\&D')
    ax1.axvline(effectiveRanks['erank'], linestyle='-.',
                color=next(colors), label=r'erank')
    ax1.axvline(effectiveRanks['coude'], linestyle=':',
                color=next(colors), label=r'coude')
    ax1.axvline(effectiveRanks['energyRatio'], linestyle='--',
                color=next(colors), label=r'energy ratio')
    ax1.axvline(effectiveRanks['stableRank'], linestyle='-.',
                color=next(colors), label=r'stable rank')

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
