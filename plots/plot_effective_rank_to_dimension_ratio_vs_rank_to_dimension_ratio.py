# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info> and
#          Vincent Thibeault
from itertools import cycle
import pandas as pd
import seaborn as sns
from plots.config_rcparams import *


def main():
    """ Plots the effective ranks to ratios
     vs the ranks to ratios of various graphs. """

    figureFilenamePDF = 'figures/pdf/effective_rank_to_dimension_ratio' \
                        '_vs_rank_to_dimension_ratio.pdf'
    figureFilenamePNG = 'figures/png/effective_rank_to_dimension_ratio' \
                        '_vs_rank_to_dimension_ratio.png'

    graphPropFilename = 'properties/graph_properties.txt'
    header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
    graphPropDF = pd.read_table(graphPropFilename, names=header,
                                comment="#", delimiter=r"\s+")
    graphPropDF.set_index('name', inplace=True)

    effectiveRanksFilename = 'properties/effective_ranks.txt'
    header = open(effectiveRanksFilename, 'r')\
        .readline().replace('#', ' ').split()
    effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                     comment="#", delimiter=r"\s+")
    effectiveRanksDF.set_index('name', inplace=True)

    effectiveRanksDF['x'] = effectiveRanksDF['rank'] / \
        effectiveRanksDF['nbVertices']
    effectiveRanksDF['y'] = effectiveRanksDF['stableRank'] / \
        effectiveRanksDF['nbVertices']

    # Valid categories, in order of priority.
    validCategories = ['Connectome', 'Animal', 'Communication', 'Foodweb',
                       'Economic', 'Transportation', 'Technological',
                       'Informational', 'Biological', 'Social']

    markers = cycle(('^', 'X', 'D', 'H', 'v', 'p', 'h', 'P', 'o', 's'))
    colorMap = dict(zip(validCategories,
                        sns.color_palette('deep', int(len(validCategories)))))
    markerMap = dict(zip(validCategories, markers))

    rank = {cat: [] for cat in validCategories}
    effectiveRank = {cat: [] for cat in validCategories}

    for networkName in effectiveRanksDF.index:

        cat = [tag for tag in graphPropDF.loc[networkName]['tags'].split(',')
               if tag in validCategories]
        if len(cat) == 0:
            cat = 'other'
            print(networkName + ' does not have a category')
        else:
            cat = cat[0]

        rank[cat].append(effectiveRanksDF.loc[networkName]['x'])
        effectiveRank[cat].append(effectiveRanksDF.loc[networkName]['y'])

    g = sns.JointGrid(height=4.5, ratio=4, space=-3)  # marginal_ticks=True,

    # plt.plot([0, 1], [0, 1], linestyle='--', color="#CFCFCF")

    for i, cat in enumerate(sorted(validCategories)):
        sns.scatterplot(x=rank[cat], y=effectiveRank[cat], facecolor='None',
                        edgecolor=colorMap[cat], alpha=0.7,
                        marker=markerMap[cat], ax=g.ax_joint)
        cat = sorted(validCategories)[i]
        sns.scatterplot(x=[0], y=[1.5], facecolor='None',
                        edgecolor=colorMap[cat],
                        marker=markerMap[cat], label=cat.lower(),
                        ax=g.ax_joint)

    g.ax_joint.legend(loc='upper left', frameon=False, fontsize='x-small')

    sns.histplot(x=effectiveRanksDF['rank'] / effectiveRanksDF['nbVertices'],
                 stat='probability', bins=30, binrange=(0, 1),
                 color="lightsteelblue", linewidth=0.3, ax=g.ax_marg_x)
    sns.histplot(y=effectiveRanksDF['stableRank'] /
                 effectiveRanksDF['nbVertices'],
                 stat='probability', bins=4*30, binrange=(0, 1),
                 color="lightsteelblue", linewidth=0.3, ax=g.ax_marg_y)

    plt.xlim(left=-0.01, right=1.05)
    plt.ylim(bottom=-0.01, top=0.27)
    
    # ax.set_yscale('log')
    # ax.set_xscale('log')

    g.ax_joint.set_xlabel('Rank to dimension ratio')
    g.ax_joint.set_ylabel('Effective rank to dimension ratio')

    # plt.show()

    plt.savefig(figureFilenamePDF)  # , bbox_inches='tight')
    plt.savefig(figureFilenamePNG)  # , bbox_inches='tight')


if __name__ == "__main__":
    main()
