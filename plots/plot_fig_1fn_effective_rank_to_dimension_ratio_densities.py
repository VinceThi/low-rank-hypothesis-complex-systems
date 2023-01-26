# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info>
#          Vincent Thibeault
# from itertools import cycle

import pandas as pd
import matplotlib as plt
from plots.config_rcparams import *
import numpy as np


def plotDensities(weightedDF, unweightedDF):
    color = "lightsteelblue"
    letter_posx, letter_posy = 0.5, 1
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(3*2, 3*2))

    srankN = unweightedDF['StableRank'] / unweightedDF['Size']
    x, bins, p = axes[0, 0].hist(srankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    mean_srankN = np.mean(srankN)
    axes[0][0].vlines(x=mean_srankN, ymin=0, ymax=0.68,
                      linestyle="--", color=dark_grey, linewidth=1)
    plt.text(mean_srankN+0.08, 0.7, f"{np.round(mean_srankN*100, 1)}%",
             transform=axes[0, 0].transAxes, fontsize=10)
    for item in p:
        item.set_height(item.get_height()/sum(x))
    plt.text(letter_posx, letter_posy, "f", fontweight="bold",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 0].transAxes)
    axes[0][0].set_ylim([-0.02*0.85, 0.85])

    nrankN = unweightedDF['NuclearRank'] / unweightedDF['Size']
    x, bins, p = axes[0, 1].hist(nrankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    mean_nrankN = np.mean(nrankN)
    axes[0][1].vlines(x=mean_nrankN, ymin=0, ymax=0.4,
                      linestyle="--", color=dark_grey, linewidth=1)
    plt.text(mean_nrankN + 0.08, 0.7, f"{np.round(mean_nrankN*100, 1)}%",
             transform=axes[0, 1].transAxes, fontsize=10)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "g", fontweight="bold",      # erank",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 1].transAxes)
    axes[0][1].set_ylim([-0.02*0.5, 0.5])

    elbowN = unweightedDF['Elbow'] / unweightedDF['Size']
    x, bins, p = axes[0, 2].hist(elbowN, bins=30, range=(0, 1),
                                 density=True, color=color)
    mean_elbowN = np.mean(elbowN)
    axes[0][2].vlines(x=mean_elbowN, ymin=0, ymax=0.4,
                      linestyle="--", color=dark_grey, linewidth=1)
    plt.text(mean_elbowN + 0.08, 0.7, f"{np.round(mean_elbowN*100, 1)}%",
             transform=axes[0, 2].transAxes, fontsize=10)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "h", fontweight="bold",  # Energy ratio"
             horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 2].transAxes)
    axes[0][2].set_ylim([-0.02 * 0.5, 0.5])

    energyN = unweightedDF['EnergyRatio']/unweightedDF['Size']
    x, bins, p = axes[1, 0].hist(energyN, bins=30, range=(0, 1),
                                 density=True, color=color)
    mean_energyN = np.mean(energyN)
    axes[1][0].vlines(x=mean_energyN, ymin=0, ymax=0.4,
                      linestyle="--", color=dark_grey, linewidth=1)
    plt.text(mean_energyN + 0.08, 0.7, f"{np.round(mean_energyN*100, 1)}%",
             transform=axes[1, 0].transAxes, fontsize=10)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "i", fontweight="bold",      # Elbow",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[1, 0].transAxes)
    axes[1][0].set_ylim([-0.02*0.5, 0.5])

    thrankN = unweightedDF['OptimalThreshold'] / unweightedDF['Size']
    x, bins, p = axes[1, 1].hist(thrankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    mean_thrankN = np.mean(thrankN)
    axes[1][1].vlines(x=mean_thrankN, ymin=0, ymax=0.4,
                      linestyle="--", color=dark_grey, linewidth=1)
    plt.text(mean_thrankN + 0.08, 0.7, f"{np.round(mean_thrankN*100, 1)}%",
             transform=axes[1, 1].transAxes, fontsize=10)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "j", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[1, 1].transAxes)
    axes[1][1].set_ylim([-0.02*0.5, 0.5])

    shrankN = unweightedDF['OptimalShrinkage'] / unweightedDF['Size']
    x, bins, p = axes[1, 2].hist(shrankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    mean_shrankN = np.mean(shrankN)
    axes[1][2].vlines(x=mean_shrankN, ymin=0, ymax=0.4,
                      linestyle="--", color=dark_grey, linewidth=1)
    plt.text(mean_shrankN + 0.08, 0.7, f"{np.round(mean_shrankN*100, 1)}%",
             transform=axes[1, 2].transAxes, fontsize=10)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "k", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[1, 2].transAxes)
    axes[1][2].set_ylim([-0.02*0.5, 0.5])

    erankN = unweightedDF['Erank'] / unweightedDF['Size']
    x, bins, p = axes[2, 0].hist(erankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    mean_erankN = np.mean(erankN)
    axes[2][0].vlines(x=mean_erankN, ymin=0, ymax=0.4,
                      linestyle="--", color=dark_grey, linewidth=1)
    plt.text(mean_erankN + 0.08, 0.7, f"{np.round(mean_erankN*100, 1)}%",
             transform=axes[2, 0].transAxes, fontsize=10)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "l", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[2, 0].transAxes)
    axes[2][0].set_ylim([-0.02 * 0.5, 0.5])

    rankN = unweightedDF['Rank'] / unweightedDF['Size']
    x, bins, p = axes[2][1].hist(rankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    mean_rankN = np.mean(rankN)
    axes[2][1].vlines(x=mean_rankN, ymin=0, ymax=0.4,
                      linestyle="--", color=dark_grey, linewidth=1)
    plt.text(mean_rankN - 0.46, 0.7, f"{np.round(mean_rankN*100, 1)}%",
             transform=axes[2, 1].transAxes, fontsize=10)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "m", fontweight="bold",  # Rank",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[2, 1].transAxes)
    axes[2][1].set_ylim([-0.02 * 0.5, 0.5])

    nbVertices = unweightedDF['Size']
    nbVertices = nbVertices.values
    x, bins, p = axes[2][2].hist(nbVertices, bins=np.logspace(np.log10(0.1),
                                                              np.log10(21000),
                                                              40),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "n", fontweight="bold",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[2, 2].transAxes)
    axes[2][2].set_xscale('log')
    axes[2][2].set_ylim([-0.02 * 0.20, 0.20])
    axes[2][2].set_xlim([0.5, 9 * 10 ** 4])
    axes[2][2].set_xticks([10 ** 0, 10 ** 2, 10 ** 4])

    axes[0, 0].set_ylabel('Fraction\n of networks')
    axes[0, 1].set_ylabel('')
    axes[0, 2].set_ylabel('')
    axes[1, 0].set_ylabel('Fraction\n of networks')
    axes[1, 1].set_ylabel('')
    axes[1, 2].set_ylabel('')
    axes[2, 0].set_ylabel('Fraction\n of networks')
    axes[2, 1].set_ylabel('')
    axes[2, 2].set_ylabel('')
    axes[0, 0].set_xlabel('srank/N')
    axes[0, 1].set_xlabel('nrank/N')
    axes[0, 2].set_xlabel('elbow/N')
    axes[1, 0].set_xlabel('energy/N')
    axes[1, 1].set_xlabel('thrank/N')
    axes[1, 2].set_xlabel('shrank/N')
    axes[2, 0].set_xlabel('erank/N')
    axes[2, 1].set_xlabel('rank/N')
    axes[2, 2].set_xlabel('N')
    axes[2, 2].xaxis.set_label_coords(1.05, -0.025)

    # axes[0, 1].set_xlabel('Rank to dimension ratio')
    # axes[0, 2].set_xlabel('Stable rank to dimension ratio')
    # axes[1, 0].set_xlabel('Nuclear rank to dimension ratio')
    # axes[1, 1].set_xlabel('Energy ratio to dimension ratio')
    # axes[1, 2].set_xlabel('Elbow to dimension ratio')
    # axes[2, 0].set_xlabel('erank to dimension ratio')
    # axes[2, 1].set_xlabel('thrank to dimension ratio')
    # axes[2, 2].set_xlabel('shrank to dimension ratio')

    return fig


def main():

    # effectiveRanksFilename = "C:/Users/thivi/Documents/GitHub/" \
    #                          "low-rank-hypothesis-complex-systems/" \
    #                          "singular_values/properties/effective_ranks.txt"
    # # 'singular_values/properties/effective_ranks.txt'
    # header = \
    #     open(effectiveRanksFilename, 'r').readline().replace('#', ' ').split()
    # effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
    #                                  comment="#", delimiter=r"\s+")
    # effectiveRanksDF.set_index('Name', inplace=True)

    graphPropFilename = '../graphs/graph_data/graph_properties_augmented.txt'
    header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
    graphPropDF = pd.read_table(graphPropFilename, names=header, comment="#", delimiter=r"\s+")
    graphPropDF.set_index('name', inplace=True)

    effRanksFilename = '../singular_values/properties/effective_ranks.txt'
    header = open(effRanksFilename, 'r').readline().replace('#', ' ').split()
    effRanksDF = pd.read_table(effRanksFilename, names=header, comment="#", delimiter=r"\s+")
    effRanksDF.rename(columns={'Name': 'name'}, inplace=True)
    effRanksDF.set_index('name', inplace=True)

    fullDF = pd.merge(graphPropDF, effRanksDF, left_index=True, right_index=True)

    weightedDF = fullDF[fullDF['(un)weighted'] == 'weighted']
    unweightedDF = fullDF[fullDF['(un)weighted'] == 'unweighted']

    # figureFilenamePDF = 'figures/pdf/' \
    #                     'effective_rank_to_dimension_ratio_densities.pdf'
    # figureFilenamePNG = 'figures/png/' \
    #                     'effective_rank_to_dimension_ratio_densities.png'

    plotDensities(weightedDF, unweightedDF)
    plt.show()
    # fig.savefig(figureFilenamePDF, bbox_inches='tight')
    # fig.savefig(figureFilenamePNG, bbox_inches='tight')


if __name__ == "__main__":
    main()
