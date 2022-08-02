# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault

import pandas as pd
import matplotlib as plt
from plots.config_rcparams import *
import numpy as np


def plotEffectiveRanks_vs_N(effectiveRanksDF):
    color = "lightsteelblue"
    letter_posx, letter_posy = 0.5, 1
    ylim = [-500, 6500]
    ylim2 = [-1000, 10500]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(3*2.4, 3*2))

    axes[0][0].scatter(effectiveRanksDF['Size'], effectiveRanksDF['StableRank'], s=10)
    axes[0][0].text(letter_posx, letter_posy, "a", fontweight="bold",
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[0, 0].transAxes)
    axes[0][0].set_ylim(ylim)

    axes[0][1].scatter(effectiveRanksDF['Size'], effectiveRanksDF['NuclearRank'], s=10)
    axes[0][1].text(letter_posx, letter_posy, "b", fontweight="bold",      # erank",
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[0, 1].transAxes)
    axes[0][1].set_ylim(ylim)

    axes[0][2].scatter(effectiveRanksDF['Size'], effectiveRanksDF['Elbow'], s=10)
    axes[0][2].text(letter_posx, letter_posy, "c", fontweight="bold",  # Energy ratio"
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[0, 2].transAxes)
    axes[0][2].set_ylim(ylim)

    axes[1][0].scatter(effectiveRanksDF['Size'], effectiveRanksDF['EnergyRatio'], s=10)
    axes[1][0].text(letter_posx, letter_posy, "d", fontweight="bold",      # Elbow",
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[1, 0].transAxes)
    axes[1][0].set_ylim(ylim)

    axes[1][1].scatter(effectiveRanksDF['Size'], effectiveRanksDF['OptimalThreshold'], s=10)
    axes[1][1].text(letter_posx, letter_posy, "e", fontweight="bold",
                    horizontalalignment="center", verticalalignment="top",
                    transform=axes[1, 1].transAxes)
    axes[1][1].set_ylim(ylim2)

    axes[1][2].scatter(effectiveRanksDF['Size'], effectiveRanksDF['OptimalShrinkage'], s=10)
    axes[1][2].text(letter_posx, letter_posy, "f", fontweight="bold",
                    horizontalalignment="center", verticalalignment="top",
                    transform=axes[1, 2].transAxes)
    axes[1][2].set_ylim(ylim2)

    axes[2][0].scatter(effectiveRanksDF['Size'], effectiveRanksDF['Erank'], s=10)
    axes[2][0].text(letter_posx, letter_posy, "g", fontweight="bold",
                    horizontalalignment="center", verticalalignment="top",
                    transform=axes[2, 0].transAxes)
    axes[2][0].set_ylim(ylim2)

    axes[2][1].scatter(effectiveRanksDF['Size'], effectiveRanksDF['Rank'], s=10)
    axes[2][1].text(letter_posx, letter_posy, "h", fontweight="bold",  # Rank",
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[2, 1].transAxes)
    axes[2][1].set_ylim([-2000, np.max(effectiveRanksDF['Size'])+3000])

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

    axes[0, 0].set_xlabel('N')
    axes[0, 1].set_xlabel('N')
    axes[0, 2].set_xlabel('N')
    axes[1, 0].set_xlabel('N')
    axes[1, 1].set_xlabel('N')
    axes[1, 2].set_xlabel('N')
    axes[2, 0].set_xlabel('N')
    axes[2, 1].set_xlabel('N')
    axes[2, 2].set_xlabel('N')
    axes[0, 0].set_ylabel('srank')
    axes[0, 1].set_ylabel('nrank')
    axes[0, 2].set_ylabel('elbow')
    axes[1, 0].set_ylabel('energy')
    axes[1, 1].set_ylabel('thrank')
    axes[1, 2].set_ylabel('shrank')
    axes[2, 0].set_ylabel('erank')
    axes[2, 1].set_ylabel('rank')
    axes[2, 2].set_ylabel('Fraction\nof graphs')
    axes[2, 2].xaxis.set_label_coords(1.05, -0.025)

    # axes[0, 0].set_xticks([10**0, 10**2, 10**4])
    # axes[0, 1].set_xticks([10**0, 10**2, 10**4])
    # axes[0, 2].set_xticks([10**0, 10**2, 10**4])
    # axes[1, 0].set_xticks([10**0, 10**2, 10**4])
    # axes[1, 1].set_xticks([10**0, 10**2, 10**4])
    # axes[1, 2].set_xticks([10**0, 10**2, 10**4])
    # axes[2, 0].set_xticks([10**0, 10**2, 10**4])
    # axes[2, 1].set_xticks([10**0, 10**2, 10**4])
    # axes[2, 2].set_xticks([10**0, 10**2, 10**4])
    # axes[0, 0].set_yticks([10**0, 10**2, 10**4])
    # axes[0, 1].set_yticks([10**0, 10**2, 10**4])
    # axes[0, 2].set_yticks([10**0, 10**2, 10**4])
    # axes[1, 0].set_yticks([10**0, 10**2, 10**4])
    # axes[1, 1].set_yticks([10**0, 10**2, 10**4])
    # axes[1, 2].set_yticks([10**0, 10**2, 10**4])
    # axes[2, 0].set_yticks([10**0, 10**2, 10**4])
    # axes[2, 1].set_yticks([10**0, 10**2, 10**4])
    # axes[2, 2].set_yticks([10**0, 10**2, 10**4])
    # 
    # axes[0, 0].set_xscale("log")
    # axes[0, 1].set_xscale("log")
    # axes[0, 2].set_xscale("log")
    # axes[1, 0].set_xscale("log")
    # axes[1, 1].set_xscale("log")
    # axes[1, 2].set_xscale("log")
    # axes[2, 0].set_xscale("log")
    # axes[2, 1].set_xscale("log")
    # axes[2, 2].set_xscale("log")
    # 
    # axes[0, 0].set_yscale("log")
    # axes[0, 1].set_yscale("log")
    # axes[0, 2].set_yscale("log")
    # axes[1, 0].set_yscale("log")
    # axes[1, 1].set_yscale("log")
    # axes[1, 2].set_yscale("log")
    # axes[2, 0].set_yscale("log")
    # axes[2, 1].set_yscale("log")
    # axes[2, 2].set_yscale("log")




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

    plotEffectiveRanks_vs_N(effectiveRanksDF)
    plt.show()
    # fig.savefig(figureFilenamePDF, bbox_inches='tight')
    # fig.savefig(figureFilenamePNG, bbox_inches='tight')


if __name__ == "__main__":
    main()
