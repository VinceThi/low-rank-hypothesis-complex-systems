# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info> and
#          Vincent Thibeault
import pandas as pd
import seaborn as sns
import numpy as np
from plots.config_rcparams import *


def main():
    """ Plots the effective ranks to ratios
     vs the ranks to ratios of various graphs. """

    figureFilenamePDF = 'figures/pdf/effective_rank_to_dimension_ratio' \
                        '_vs_rank_to_dimension_ratio.pdf'
    figureFilenamePNG = 'figures/png/effective_rank_to_dimension_ratio' \
                        '_vs_rank_to_dimension_ratio.png'

    graphPropFilename = 'C:/Users/thivi/Documents/GitHub/' \
                        'low-rank-hypothesis-complex-systems/graphs/' \
                        'graph_data/graph_properties.txt'
    header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
    graphPropDF = pd.read_table(graphPropFilename, names=header,
                                comment="#", delimiter=r"\s+")

    graphPropArray = graphPropDF.to_numpy()
    # print(graphPropDF)
    graphPropDF.set_index('name', inplace=True)

    # Connectome tag
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,Connectome", 'tags'] = "Connectome"

    # Ecological tag
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Social,Animal", 'tags'] = "Ecological"
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,Foodweb", 'tags'] = "Ecological"
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,FoodWeb,Uncertain", 'tags'] = "Ecological"
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,Foodweb,Multilayer", 'tags'] = "Ecological"
    
    # Interactome tag
    # See Interactome Networks and Human Disease by Vidal et al. for more info
    # on interactomes. We include the drug-drug interactions in the
    # interactome category.
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,Proteininteractions", 'tags'] = "Interactome"
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,Generegulation", 'tags'] = "Interactome"
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,Generegulation,Proteininteractions,Multilayer",
                    'tags'] = "Interactome"
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,Genetic,Projection,Multilayer",
                    'tags'] = "Interactome"
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,Metabolic", 'tags'] = "Interactome"
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Biological,Druginteractions", 'tags'] = "Interactome"

    # Economic tag
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Economic,Trade,Multilayer", 'tags'] = "Economic"

    # Communication tag
    graphPropDF.loc[graphPropDF['tags'] ==
                    "Social,Communication", 'tags'] = "Communication"
    graphPropDF.loc[graphPropDF['tags'] == "Social,Communication,Timestamps",
                    "tags"] = "Communication"

    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None):
    #     print(graphPropDF['tags'])

    effectiveRanksFilename = 'C:/Users/thivi/Documents/GitHub/' \
                             'low-rank-hypothesis-complex-systems/' \
                             'singular_values/properties/effective_ranks.txt'
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
    validCategories = ['Connectome', 'Interactome', 'Ecological',
                       'Communication', 'Economic',
                       'Transportation', 'Technological',
                       'Informational', 'Social']
    markers = ['*', 'h', '^', 'D', 'H', 'v', 'd', 'p', '.', 's',
               'X', 'P', 'o', 's']
    colors = ["#C44E52", "#DD8452", "#55A868", "#DA8BC3", "#8172B3",
              "#937860", "#64B5CD", "#8C8C8C",  "#4C72B0"]
    facecolors = ["#C44E52", None, None, None, None,
                  None, None, None, None]
    # "#CCB974",
    colorMap = dict(zip(validCategories, colors))
    facecolorMap = dict(zip(validCategories, facecolors))
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

    # g = sns.JointGrid(x=[0], y=[0],
    #                   height=4.5, ratio=4, space=-3)  # marginal_ticks=True,

    # plt.plot([0, 1], [0, 1], linestyle='--', color="#CFCFCF")

    plt.figure(figsize=(4.5, 4.5))

    for i, cat in enumerate(reversed(validCategories)):

        if cat == "Connectome":
            sns.scatterplot(x=rank[cat], y=effectiveRank[cat],
                            facecolor='None', s=100,
                            edgecolor=colorMap[cat],  # alpha=0.7,
                            marker=markerMap[cat])  # , ax=g.ax_joint)
            other_connectomes = ["mouse_meso", "mouse_voxel", "zebrafish_meso",
                                 "celegans_signed", "drosophila"]
            N_connectomes = np.array([213, 15314, 71, 297, 21733])
            rank_connectome = np.array([185, 15313, 71, 292, 21687])
            srank_connectome = np.array([3.61975, 8.04222, 1.80914, 8.67117, 11.5811])
            optshrink_connectome = np.array([ , 2741, , , ,])

            print(srank_connectome/N_connectomes)
            sns.scatterplot(x=rank_connectome/N_connectomes,
                            y=srank_connectome/N_connectomes, s=100,
                            facecolor='None',
                            edgecolor=colorMap[cat], marker=markerMap[cat])
            # cat = "Connectome"
            # sns.scatterplot(x=[0], y=[1.5], facecolor='None', s=100,
            #                 edgecolor=colorMap[cat],
            #                 marker=markerMap[cat], label=cat.lower())
            # # ax=g.ax_joint)

        else:
            sns.scatterplot(x=rank[cat], y=effectiveRank[cat],
                            facecolor='None',
                            edgecolor=colorMap[cat], alpha=0.7,
                            marker=markerMap[cat])  # , ax=g.ax_joint)
            cat = sorted(validCategories)[i]
            sns.scatterplot(x=[0], y=[1.5], facecolor='None',
                            edgecolor=colorMap[cat],
                            marker=markerMap[cat], label=cat.lower(),
                            zorder=len(validCategories)-i)
            # ax=g.ax_joint)

    # g.ax_joint.legend(loc='upper left', frameon=False, fontsize='x-small')

    # sns.histplot(x=effectiveRanksDF['rank'] / effectiveRanksDF['nbVertices'],
    #              stat='probability', bins=30, binrange=(0, 1),
    #              color="lightsteelblue", linewidth=0.3, ax=g.ax_marg_x)
    # sns.histplot(y=effectiveRanksDF['stableRank'] /
    #              effectiveRanksDF['nbVertices'],
    #              stat='probability', bins=4*30, binrange=(0, 1),
    #              color="lightsteelblue", linewidth=0.3, ax=g.ax_marg_y)

    plt.xlim(left=-0.01, right=1.05)
    plt.ylim(bottom=-0.01, top=0.27)
    
    # ax.set_yscale('log')
    # ax.set_xscale('log')

    # g.ax_joint.set_xlabel('Rank to dimension ratio')
    # g.ax_joint.set_ylabel('Effective rank to dimension ratio')

    plt.xlabel('Rank to dimension ratio')
    plt.ylabel('Effective rank to dimension ratio')
    plt.show()

    # plt.savefig(figureFilenamePDF)  # , bbox_inches='tight')
    # plt.savefig(figureFilenamePNG)  # , bbox_inches='tight')


if __name__ == "__main__":
    main()
