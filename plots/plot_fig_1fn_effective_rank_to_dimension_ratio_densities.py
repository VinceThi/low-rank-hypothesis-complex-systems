# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info>
#          Vincent Thibeault
# from itertools import cycle

import pandas as pd
import matplotlib as plt
from plots.config_rcparams import *
import numpy as np
import re

# from https://stackoverflow.com/questions/44111169/
# remove-trailing-zeros-after-the-decimal-point-in-python
tail_dot_rgx = re.compile(r'(?:(\.)|(\.\d*?[1-9]\d*?))0+(?=\b|[^0-9])')


def remove_dot_zeros(a):
    return tail_dot_rgx.sub(r'\2', a)


def plotDensities(weightedDF, unweightedDF):
    color_u = dark_grey
    color_w = "lightsteelblue"
    fontsize_mean = 8
    letter_posx, letter_posy = 0.5, 1
    alpha_u = 0.9
    alpha_w = 0.75
    r = 0
    linestyle_u = ":"
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(3*2, 3*2))

    """ srank """
    srankN_u = unweightedDF['StableRank'] / unweightedDF['Size']
    x, bins, p = axes[0, 0].hist(srankN_u, bins=30, range=(0, 1),
                                 density=True, color=color_u, alpha=alpha_u)
    mean_srankN_u = np.mean(srankN_u)
    axes[0][0].vlines(x=mean_srankN_u, ymin=0, ymax=0.68,
                      linestyle="-", color=color_u, linewidth=1)
    plt.text(mean_srankN_u + 0.08, 0.7,
             remove_dot_zeros(f"{np.round(mean_srankN_u*100, r)}%"),
             transform=axes[0, 0].transAxes, fontsize=fontsize_mean,
             color=color_u)
    for item in p:
        item.set_height(item.get_height()/sum(x))

    srankN_w = weightedDF['StableRank'] / weightedDF['Size']
    x, bins, p = axes[0, 0].hist(srankN_w, bins=30, range=(0, 1),
                                 density=True, color=color_w, alpha=alpha_w)
    mean_srankN_w = np.mean(srankN_w)
    axes[0][0].vlines(x=mean_srankN_w, ymin=0, ymax=0.68,
                      linestyle="--", color=color_w, linewidth=1)
    plt.text(mean_srankN_w + 0.074, 0.55,
             remove_dot_zeros(f"{np.round(mean_srankN_w*100, r)}%"),
             transform=axes[0, 0].transAxes, fontsize=fontsize_mean,
             color=color_w)
    for item in p:
        item.set_height(item.get_height()/sum(x))
    plt.text(letter_posx, letter_posy, "f", fontweight="bold",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 0].transAxes)
    axes[0][0].set_ylim([-0.02 * 0.85, 0.85])
    axes[0][0].set_yticks([0, 0.4, 0.8])

    """ nrank """
    nrankN = unweightedDF['NuclearRank'] / unweightedDF['Size']
    x, bins, p = axes[0, 1].hist(nrankN, bins=30, range=(0, 1),
                                 density=True, color=color_u, alpha=alpha_u)
    mean_nrankN = np.mean(nrankN)
    axes[0][1].vlines(x=mean_nrankN, ymin=0, ymax=0.4,
                      linestyle=linestyle_u, color=color_u, linewidth=1)
    plt.text(mean_nrankN + 0.08, 0.7,
             remove_dot_zeros(f"{np.round(mean_nrankN*100, r)}%"),
             transform=axes[0, 1].transAxes, fontsize=fontsize_mean,
             color=color_u)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    nrankN = weightedDF['NuclearRank'] / weightedDF['Size']
    x, bins, p = axes[0, 1].hist(nrankN, bins=30, range=(0, 1),
                                 density=True, color=color_w, alpha=alpha_w)
    mean_nrankN = np.mean(nrankN)
    axes[0][1].vlines(x=mean_nrankN, ymin=0, ymax=0.4,
                      linestyle="--", color=color_w, linewidth=1)
    plt.text(mean_nrankN + 0.09, 0.55,
             remove_dot_zeros(f"{np.round(mean_nrankN*100, r)}%"),
             transform=axes[0, 1].transAxes, fontsize=fontsize_mean,
             color=color_w)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    plt.text(letter_posx, letter_posy, "g", fontweight="bold",      # erank",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 1].transAxes)
    axes[0][1].set_ylim([-0.02*0.5, 0.55])

    """ elbow """
    elbowN = unweightedDF['Elbow'] / unweightedDF['Size']
    x, bins, p = axes[0, 2].hist(elbowN, bins=30, range=(0, 1),
                                 density=True, color=color_u, alpha=alpha_u)
    mean_elbowN = np.mean(elbowN)
    axes[0][2].vlines(x=mean_elbowN, ymin=0, ymax=0.4,
                      linestyle="-", color=color_u, linewidth=1)
    plt.text(mean_elbowN + 0.08, 0.7,
             remove_dot_zeros(f"{np.round(mean_elbowN*100, r)}%"),
             transform=axes[0, 2].transAxes, fontsize=fontsize_mean,
             color=color_u)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    elbowN = weightedDF['Elbow'] / weightedDF['Size']
    x, bins, p = axes[0, 2].hist(elbowN, bins=30, range=(0, 1),
                                 density=True, color=color_w, alpha=alpha_w)
    mean_elbowN = np.mean(elbowN)
    axes[0][2].vlines(x=mean_elbowN, ymin=0, ymax=0.4,
                      linestyle="--", color=color_w, linewidth=1)
    plt.text(mean_elbowN + 0.08, 0.55,
             remove_dot_zeros(f"{np.round(mean_elbowN*100, r)}%"),
             transform=axes[0, 2].transAxes, fontsize=fontsize_mean,
             color=color_w)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "h", fontweight="bold",  # Energy ratio"
             horizontalalignment="center",
             verticalalignment="top", transform=axes[0, 2].transAxes)
    axes[0][2].set_ylim([-0.02 * 0.5, 0.5])

    """ energy """
    energyN = unweightedDF['EnergyRatio'] / unweightedDF['Size']
    x, bins, p = axes[1, 0].hist(energyN, bins=30, range=(0, 1),
                                 density=True, color=color_u, alpha=alpha_u)
    mean_energyN = np.mean(energyN)
    axes[1][0].vlines(x=mean_energyN, ymin=0, ymax=0.4,
                      linestyle=linestyle_u, color=color_u, linewidth=1)
    plt.text(mean_energyN + 0.07, 0.72,
             remove_dot_zeros(f"{np.round(mean_energyN*100, r)}%"),
             transform=axes[1, 0].transAxes, fontsize=fontsize_mean,
             color=color_u)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    energyN = weightedDF['EnergyRatio'] / weightedDF['Size']
    x, bins, p = axes[1, 0].hist(energyN, bins=30, range=(0, 1),
                                 density=True, color=color_w, alpha=alpha_w)
    mean_energyN = np.mean(energyN)
    axes[1][0].vlines(x=mean_energyN, ymin=0, ymax=0.4,
                      linestyle="--", color=color_w, linewidth=1)
    plt.text(mean_energyN-0.05, 0.82,
             remove_dot_zeros(f"{np.round(mean_energyN*100, r)}%"),
             transform=axes[1, 0].transAxes, fontsize=fontsize_mean,
             color=color_w)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    plt.text(letter_posx, letter_posy, "i", fontweight="bold",      # Elbow",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[1, 0].transAxes)
    axes[1][0].set_ylim([-0.02*0.5, 0.5])

    """ thrank """
    thrankN = unweightedDF['OptimalThreshold'] / unweightedDF['Size']
    x, bins, p = axes[1, 1].hist(thrankN, bins=30, range=(0, 1),
                                 density=True, color=color_u, alpha=alpha_u)
    mean_thrankN = np.mean(thrankN)
    axes[1][1].vlines(x=mean_thrankN, ymin=0, ymax=0.4,
                      linestyle=linestyle_u, color=color_u, linewidth=1)
    plt.text(mean_thrankN - 0.03, 0.82,
             remove_dot_zeros(f"{np.round(mean_thrankN*100, r)}%"),
             transform=axes[1, 1].transAxes, fontsize=fontsize_mean,
             color=color_u)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    thrankN = weightedDF['OptimalThreshold'] / weightedDF['Size']
    x, bins, p = axes[1, 1].hist(thrankN, bins=30, range=(0, 1),
                                 density=True, color=color_w, alpha=alpha_w)
    mean_thrankN = np.mean(thrankN)
    axes[1][1].vlines(x=mean_thrankN, ymin=0, ymax=0.4,
                      linestyle="--", color=color_w, linewidth=1)
    plt.text(mean_thrankN + 0.07, 0.7,
             remove_dot_zeros(f"{np.round(mean_thrankN*100, r)}%"),
             transform=axes[1, 1].transAxes, fontsize=fontsize_mean,
             color=color_w)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    plt.text(letter_posx, letter_posy, "j", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[1, 1].transAxes)
    axes[1][1].set_ylim([-0.02*0.5, 0.5])

    """ shrank """
    shrankN = unweightedDF['OptimalShrinkage'] / unweightedDF['Size']
    x, bins, p = axes[1, 2].hist(shrankN, bins=30, range=(0, 1),
                                 density=True, color=color_u, alpha=alpha_u)
    mean_shrankN = np.mean(shrankN)
    axes[1][2].vlines(x=mean_shrankN, ymin=0, ymax=0.4,
                      linestyle=linestyle_u, color=color_u, linewidth=1)
    plt.text(mean_shrankN - 0.05, 0.82,
             remove_dot_zeros(f"{np.round(mean_shrankN*100, r)}%"),
             transform=axes[1, 2].transAxes, fontsize=fontsize_mean,
             color=color_u)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    shrankN = weightedDF['OptimalShrinkage'] / weightedDF['Size']
    x, bins, p = axes[1, 2].hist(shrankN, bins=30, range=(0, 1),
                                 density=True, color=color_w, alpha=alpha_w)
    mean_shrankN = np.mean(shrankN)
    axes[1][2].vlines(x=mean_shrankN, ymin=0, ymax=0.4,
                      linestyle="--", color=color_w, linewidth=1)
    plt.text(mean_shrankN + 0.07, 0.72,
             remove_dot_zeros(f"{np.round(mean_shrankN*100, r)}%"),
             transform=axes[1, 2].transAxes, fontsize=fontsize_mean,
             color=color_w)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    plt.text(letter_posx, letter_posy, "k", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[1, 2].transAxes)
    axes[1][2].set_ylim([-0.02*0.5, 0.5])

    """ erank """
    erankN = unweightedDF['Erank'] / unweightedDF['Size']
    x, bins, p = axes[2, 0].hist(erankN, bins=30, range=(0, 1),
                                 density=True, color=color_u, alpha=alpha_u)
    mean_erankN = np.mean(erankN)
    axes[2][0].vlines(x=mean_erankN, ymin=0, ymax=0.4,
                      linestyle=linestyle_u, color=color_u, linewidth=1)
    plt.text(mean_erankN + 0.05, 0.72,
             remove_dot_zeros(f"{np.round(mean_erankN*100, r)}%"),
             transform=axes[2, 0].transAxes, fontsize=fontsize_mean,
             color=color_u)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    erankN = weightedDF['Erank'] / weightedDF['Size']
    x, bins, p = axes[2, 0].hist(erankN, bins=30, range=(0, 1),
                                 density=True, color=color_w, alpha=alpha_w)
    mean_erankN = np.mean(erankN)
    axes[2][0].vlines(x=mean_erankN, ymin=0, ymax=0.4,
                      linestyle="--", color=color_w, linewidth=1)
    plt.text(mean_erankN - 0.22, 0.72,
             remove_dot_zeros(f"{np.round(mean_erankN*100, r)}%"),
             transform=axes[2, 0].transAxes, fontsize=fontsize_mean,
             color=color_w)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    plt.text(letter_posx, letter_posy, "l", fontweight="bold",
             horizontalalignment="center", verticalalignment="top",
             transform=axes[2, 0].transAxes)
    axes[2][0].set_ylim([-0.02 * 0.5, 0.5])

    """ rank """
    rankN = unweightedDF['Rank'] / unweightedDF['Size']
    x, bins, p = axes[2][1].hist(rankN, bins=30, range=(0, 1),
                                 density=True, color=color_u, alpha=alpha_u)
    mean_rankN = np.mean(rankN)
    axes[2][1].vlines(x=mean_rankN, ymin=0, ymax=0.4,
                      linestyle=linestyle_u, color=color_u, linewidth=1)
    plt.text(mean_rankN + 0.02, 0.72,
             remove_dot_zeros(f"{np.round(mean_rankN*100, r)}%"),
             transform=axes[2, 1].transAxes, fontsize=fontsize_mean,
             color=color_u)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    rankN = weightedDF['Rank'] / weightedDF['Size']
    x, bins, p = axes[2][1].hist(rankN, bins=30, range=(0, 1),
                                 density=True, color=color_w, alpha=alpha_w)
    mean_rankN = np.mean(rankN)
    axes[2][1].vlines(x=mean_rankN, ymin=0, ymax=0.4,
                      linestyle="--", color=color_w, linewidth=1)
    plt.text(mean_rankN - 0.26, 0.72,
             remove_dot_zeros(f"{np.round(mean_rankN*100, r)}%"),
             transform=axes[2, 1].transAxes, fontsize=fontsize_mean,
             color=color_w)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    plt.text(letter_posx, letter_posy, "m", fontweight="bold",  # Rank",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[2, 1].transAxes)
    axes[2][1].set_ylim([-0.02 * 0.5, 0.5])

    """ Number of vertices """
    nbVertices = unweightedDF['Size']
    nbVertices = nbVertices.values
    x, bins, p = axes[2][2].hist(nbVertices, bins=np.logspace(np.log10(0.1),
                                                              np.log10(21000),
                                                              40),
                                 density=True, color=color_u, alpha=alpha_u)
    for item in p:
        item.set_height(item.get_height() / sum(x))

    nbVertices = weightedDF['Size']
    nbVertices = nbVertices.values
    x, bins, p = axes[2][2].hist(nbVertices, bins=np.logspace(np.log10(0.1),
                                                              np.log10(21000),
                                                              40),
                                 density=True, color=color_w, alpha=alpha_w)
    for item in p:
        item.set_height(item.get_height() / sum(x))
    plt.text(letter_posx, letter_posy, "n", fontweight="bold",
             horizontalalignment="center",
             verticalalignment="top", transform=axes[2, 2].transAxes)
    axes[2][2].set_xscale('log')
    axes[2][2].set_ylim([-0.04 * 0.20, 0.5])
    axes[2][2].set_yticks([0, 0.2, 0.4])
    axes[2][2].set_xlim([0.5, 9 * 10 ** 4])
    axes[2][2].set_xticks([10**0, 10**2, 10**4])

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
    #    open(effectiveRanksFilename, 'r').readline().replace('#', ' ').split()
    # effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
    #                                  comment="#", delimiter=r"\s+")
    # effectiveRanksDF.set_index('Name', inplace=True)

    graphPropFilename = '../graphs/graph_data/graph_properties_augmented.txt'
    header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
    graphPropDF = pd.read_table(graphPropFilename, names=header,
                                comment="#", delimiter=r"\s+")
    graphPropDF.set_index('name', inplace=True)

    effRanksFilename = '../singular_values/properties/effective_ranks.txt'
    header = open(effRanksFilename, 'r').readline().replace('#', ' ').split()
    effRanksDF = pd.read_table(effRanksFilename, names=header,
                               comment="#", delimiter=r"\s+")
    effRanksDF.rename(columns={'Name': 'name'}, inplace=True)
    effRanksDF.set_index('name', inplace=True)

    fullDF = pd.merge(graphPropDF, effRanksDF,
                      left_index=True, right_index=True)

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
