# -*- coding: utf-8 -*-
# @author: Vincent Thibeault-

import pandas as pd
import matplotlib as plt
from plots.config_rcparams import *
from scipy.stats import pearsonr
import numpy as np


def plotEffectiveRank_vs_density(ax, letter_str, effrank_name, networkDF,
                                 ylim, labelpad):
    print(f"\n{effrank_name} vs. density\n--------------")

    """ ----------------- Get the data -------------------------- """
    density = networkDF['matDensity']
    effrank = networkDF[effrank_name]
    print(f"{len(density)} networks")

    """ ----------------- Plot effective ranks vs density ----------------- """
    s = 2
    letter_posx, letter_posy = -0.27, 1.05
    ax.scatter(density, effrank, s=s)
    ax.text(letter_posx, letter_posy, letter_str, fontweight="bold",
            horizontalalignment="center",
            verticalalignment="top", transform=ax.transAxes)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Density')
    ax.set_ylim([0.3, 30000])
    ax.set_xlim([10**(-4), 2])
    ax.set_yticks([10**0, 10**2, 10**4])
    ax.set_ylabel(effrank_name, labelpad=labelpad)
    ax.tick_params(axis="y", which="minor", top=False, right=False, left=False)

    """ ----------------- Print correlation measures  --------------------- """
    corr = pearsonr(np.log10(density), np.log10(effrank))[0]
    print(f"Pearson correlation coefficient = {corr}")
    ax.text(0.02, 18000, f"$r = {np.round(corr, 2)}$", fontsize=10)


def plot_effective_ranks_vs_density(networkDF):

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
    plotEffectiveRank_vs_density(axes[0][0], "a", "srank",
                                 networkDF, [-10, 250], 0)

    plotEffectiveRank_vs_density(axes[0][1], "b", "nrank",
                                 networkDF, [-50, 1000], 0)

    plotEffectiveRank_vs_density(axes[0][2], "c", "elbow",
                                 networkDF, [-50, 1000], 0)

    plotEffectiveRank_vs_density(axes[0][3], "d",  "energy",
                                 networkDF, [-50, 2000], 0)

    plotEffectiveRank_vs_density(axes[1][0], "e",  "thrank",
                                 networkDF, [-100, 2000], 0)

    plotEffectiveRank_vs_density(axes[1][1], "f", "shrank",
                                 networkDF, [-100, 2000], 0)

    plotEffectiveRank_vs_density(axes[1][2], "g", "erank",
                                 networkDF, [-100, 2000], 0)

    plotEffectiveRank_vs_density(axes[1][3], "h", "rank",
                                 networkDF, [-250, 5000], 0)

    plt.subplots_adjust(right=0.5, left=0)


def main():

    effectiveRanksFilename = "C:/Users/thivi/Documents/GitHub/" \
                             "low-rank-hypothesis-complex-systems/" \
                             "graphs/graph_data/datasets_table.txt"
    header = \
        open(effectiveRanksFilename, 'r').readline().replace('#', ' ').split()
    effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                     comment="#", delimiter=r"\s+")
    effectiveRanksDF.set_index('Name', inplace=True)

    # figureFilenamePDF = 'figures/pdf/' \
    #                     'effective_rank_to_dimension_ratio_densities.pdf'
    # figureFilenamePNG = 'figures/png/' \
    #                     'effective_rank_to_dimension_ratio_densities.png'

    plot_effective_ranks_vs_density(effectiveRanksDF)
    plt.show()
    # fig.savefig(figureFilenamePDF, bbox_inches='tight')
    # fig.savefig(figureFilenamePNG, bbox_inches='tight')


if __name__ == "__main__":
    main()
