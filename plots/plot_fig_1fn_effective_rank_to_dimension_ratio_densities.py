# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info>
#          Vincent Thibeault
# from itertools import cycle

import pandas as pd
import matplotlib as plt
from plots.config_rcparams import *
import numpy as np


def plotDensities(effectiveRanksDF):
    color = "lightsteelblue"
    letter_posx, letter_posy = 0.5, 1
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(3*2, 3*2))

    srankN = effectiveRanksDF['StableRank'] / effectiveRanksDF['Size']
    x, bins, p = axes[0, 0].hist(srankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height()/sum(x))
    plt.text(letter_posx, letter_posy, "h", fontweight="bold",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 0].transAxes)
    axes[0][0].set_ylim([-0.02*0.85, 0.85])

    nrankN = effectiveRanksDF['NuclearRank'] / effectiveRanksDF['Size']
    x, bins, p = axes[0, 1].hist(nrankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "i", fontweight="bold",      # erank",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 1].transAxes)
    axes[0][1].set_ylim([-0.02*0.5, 0.5])

    elbowN = effectiveRanksDF['Elbow'] / effectiveRanksDF['Size']
    x, bins, p = axes[0, 2].hist(elbowN, bins=30, range=(0, 1),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "k", fontweight="bold",  # Energy ratio"
             horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 2].transAxes)
    axes[0][2].set_ylim([-0.02 * 0.5, 0.5])

    energyN = effectiveRanksDF['EnergyRatio']/effectiveRanksDF['Size']
    x, bins, p = axes[1, 0].hist(energyN, bins=30, range=(0, 1),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "j", fontweight="bold",      # Elbow",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[1, 0].transAxes)
    axes[1][0].set_ylim([-0.02*0.5, 0.5])

    thrankN = effectiveRanksDF['OptimalThreshold'] / effectiveRanksDF['Size']
    x, bins, p = axes[1, 1].hist(thrankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "m", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[1, 1].transAxes)
    axes[1][1].set_ylim([-0.02*0.5, 0.5])

    shrankN = effectiveRanksDF['OptimalShrinkage'] / effectiveRanksDF['Size']
    x, bins, p = axes[1, 2].hist(shrankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "n", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[1, 2].transAxes)
    axes[1][2].set_ylim([-0.02*0.5, 0.5])

    erankN = effectiveRanksDF['Erank'] / effectiveRanksDF['Size']
    x, bins, p = axes[2, 0].hist(erankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "l", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[2, 0].transAxes)
    axes[2][0].set_ylim([-0.02 * 0.5, 0.5])

    rankN = effectiveRanksDF['Rank'] / effectiveRanksDF['Size']
    x, bins, p = axes[2][1].hist(rankN, bins=30, range=(0, 1),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "g", fontweight="bold",  # Rank",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[2, 1].transAxes)
    axes[2][1].set_ylim([-0.02 * 0.5, 0.5])

    nbVertices = effectiveRanksDF['Size']
    nbVertices = nbVertices.values
    x, bins, p = axes[2][2].hist(nbVertices, bins=np.logspace(np.log10(0.1),
                                                              np.log10(21000),
                                                              40),
                                 density=True, color=color)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "f", fontweight="bold",
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

    effectiveRanksFilename = "C:/Users/thivi/Documents/GitHub/" \
                             "low-rank-hypothesis-complex-systems/" \
                             "singular_values/properties/effective_ranks.txt"
    # 'singular_values/properties/effective_ranks.txt'
    header = \
        open(effectiveRanksFilename, 'r').readline().replace('#', ' ').split()
    effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                     comment="#", delimiter=r"\s+")
    effectiveRanksDF.set_index('Name', inplace=True)

    # figureFilenamePDF = 'figures/pdf/' \
    #                     'effective_rank_to_dimension_ratio_densities.pdf'
    # figureFilenamePNG = 'figures/png/' \
    #                     'effective_rank_to_dimension_ratio_densities.png'

    plotDensities(effectiveRanksDF)
    plt.show()
    # fig.savefig(figureFilenamePDF, bbox_inches='tight')
    # fig.savefig(figureFilenamePNG, bbox_inches='tight')


if __name__ == "__main__":
    main()
