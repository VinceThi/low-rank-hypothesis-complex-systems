# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info>
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def plotSingularValues(singularValues, effectiveRanks):

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
            fig = plotSingularValues(singularValues,
                                     effectiveRanksDF.loc[networkName])
            fig.savefig(figureFilenamePDF, bbox_inches='tight')
            fig.savefig(figureFilenamePNG, bbox_inches='tight')


if __name__ == "__main__":
    main()
