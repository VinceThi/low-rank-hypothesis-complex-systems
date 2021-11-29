# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info>
#          Vincent Thibeault
# from itertools import cycle

import pandas as pd
import seaborn as sns
from plots.config_rcparams import *


def plotDensities(effectiveRanksDF):

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(3*3, 2*3))

    sns.histplot(effectiveRanksDF['rank'] / effectiveRanksDF['nbVertices'],
                 ax=axes[0, 0], stat='probability', bins=30, binrange=(0, 1),
                 color="lightsteelblue")
    plt.text(0.500, 1.000, "(a) Rank", horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 0].transAxes)
    axes[0][0].set_ylim([0, 0.5])

    sns.histplot(effectiveRanksDF['rankGD'] / effectiveRanksDF['nbVertices'],
                 ax=axes[0, 1], stat='probability', bins=30, binrange=(0, 1),
                 color="lightsteelblue")
    plt.text(0.500, 1.000, "(b) Gavish-Donoho", horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 1].transAxes)
    axes[0][1].set_ylim([0, 0.5])

    sns.histplot(effectiveRanksDF['erank'] / effectiveRanksDF['nbVertices'],
                 ax=axes[0, 2], stat='probability', bins=30, binrange=(0, 1),
                 color="lightsteelblue")
    plt.text(0.500, 1.000, "(c) erank", horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 2].transAxes)
    axes[0][2].set_ylim([0, 0.5])

    sns.histplot(effectiveRanksDF['coude'] / effectiveRanksDF['nbVertices'],
                 ax=axes[1, 0], stat='probability', bins=30, binrange=(0, 1),
                 color="lightsteelblue")
    plt.text(0.500, 1.000, "(d) Elbow", horizontalalignment="center",
             verticalalignment="top", transform=axes[1, 0].transAxes)
    axes[1][0].set_ylim([0, 0.5])

    sns.histplot(effectiveRanksDF['energyRatio'] /
                 effectiveRanksDF['nbVertices'],
                 ax=axes[1, 1], stat='probability', bins=30, binrange=(0, 1),
                 color="lightsteelblue")
    plt.text(0.500, 1.000, "(e) Energy ratio", horizontalalignment="center",
             verticalalignment="top", transform=axes[1, 1].transAxes)
    axes[1][1].set_ylim([0, 0.5])

    sns.histplot(effectiveRanksDF['stableRank']/effectiveRanksDF['nbVertices'],
                 ax=axes[1, 2], stat='probability', bins=30, binrange=(0, 1),
                 color="lightsteelblue")
    plt.text(0.500, 1.000, "(f) Stable rank",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[1, 2].transAxes)
    axes[1][2].set_ylim([0, 0.85])

    axes[0, 0].set_ylabel('Fraction of graphs')
    axes[0, 1].set_ylabel('')
    axes[0, 2].set_ylabel('')
    axes[1, 0].set_ylabel('Fraction of graphs')
    axes[1, 1].set_ylabel('')
    axes[1, 2].set_ylabel('')
    axes[0, 0].set_xlabel('')
    axes[0, 1].set_xlabel('')
    axes[0, 2].set_xlabel('')
    axes[1, 0].set_xlabel('Effective rank to dimension ratio')
    axes[1, 1].set_xlabel('Effective rank to dimension ratio')
    axes[1, 2].set_xlabel('Effective rank to dimension ratio')
    
    return fig


def main():

    effectiveRanksFilename = 'singular_values/properties/effective_ranks.txt'
    header = \
        open(effectiveRanksFilename, 'r').readline().replace('#', ' ').split()
    effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                     comment="#", delimiter=r"\s+")
    effectiveRanksDF.set_index('name', inplace=True)

    figureFilenamePDF = 'figures/pdf/' \
                        'effective_rank_to_dimension_ratio_densities.pdf'
    figureFilenamePNG = 'figures/png/' \
                        'effective_rank_to_dimension_ratio_densities.png'

    fig = plotDensities(effectiveRanksDF)
    fig.savefig(figureFilenamePDF, bbox_inches='tight')
    fig.savefig(figureFilenamePNG, bbox_inches='tight')


if __name__ == "__main__":
    main()
