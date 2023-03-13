# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault and Antoine Allard <antoineallard.info>
import pandas as pd
import seaborn as sns
import numpy as np
from plots.config_rcparams import *

""" Plots the effective ranks to ratios
 vs the ranks to ratios of various real networks. """

figureFilenamePDF = 'figures/pdf/fig1_effective_rank_vs_rank_real_networks.pdf'
figureFilenamePNG = 'figures/png/fig1_effective_rank_vs_rank_real_networks.png'

graphPropFilename = 'C:/Users/thivi/Documents/GitHub/' \
                    'low-rank-hypothesis-complex-systems/graphs/' \
                    'graph_data/graph_properties_augmented.txt'
header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
graphPropDF = pd.read_table(graphPropFilename, names=header,
                            comment="#", delimiter=r"\s+")
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

effectiveRanksFilename = 'C:/Users/thivi/Documents/GitHub/' \
                         'low-rank-hypothesis-complex-systems/' \
                         'singular_values/properties/effective_ranks.txt'
header = open(effectiveRanksFilename, 'r')\
    .readline().replace('#', ' ').split()
effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                 comment="#", delimiter=r"\s+")
effectiveRanksDF.set_index('Name', inplace=True)

effectiveRanksDF['x'] = effectiveRanksDF['Rank'] / \
    effectiveRanksDF['Size']
effectiveRanksDF['y'] = effectiveRanksDF['StableRank'] / \
    effectiveRanksDF['Size']


# --- Valid categories, in increasing order of number of networks in the data
validCategories = ['Transportation', 'Communication', 'Economic', 'Learned',
                   'Technological', 'Connectome',  'Ecological',
                   'Informational', 'Interactome', 'Social']
markers = ['v', 'D', 'p', 'H', 'd', '*', '^', 'X', 'h', '.', 's',
           'P', 'o']
colors = ["#937860", "#DA8BC3", "#8172B3", "#CCB974", "#64B5CD",
          "#C44E52", "#55A868", "#8C8C8C", "#DD8452", "#4C72B0"]
facecolors = ["#C44E52", None, None, None, None,
              None, None, None, None, None]
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


plt.figure(figsize=(5, 5))

total_number_networks = 0
for i, cat in enumerate(reversed(validCategories)):

    sns.scatterplot(x=rank[cat], y=effectiveRank[cat],
                    facecolor='None',
                    edgecolor=colorMap[cat], alpha=0.7,
                    marker=markerMap[cat])  # , ax=g.ax_joint)
    nb_network_cat = len(effectiveRank[cat])
    print(f"{cat}: {nb_network_cat} networks")
    # cat = sorted(validCategories)[i]
    sns.scatterplot(x=[0], y=[1.5], facecolor='None',
                    edgecolor=colorMap[cat],
                    marker=markerMap[cat],  # .lower(),to have lower case label
                    label=cat + f" ({np.around(nb_network_cat/679*100, 1)}%)",
                    zorder=len(validCategories)-i)

    total_number_networks += len(effectiveRank[cat])

print(f"Total number of networks = {total_number_networks}")

if total_number_networks != 679:
    raise ValueError("One much change the number of networks. 679 networks is "
                     "hard coded in other places in the script.")

plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=0.22)

plt.xlabel('rank/N')
plt.ylabel('srank/N')
plt.show()

plt.savefig(figureFilenamePDF)
plt.savefig(figureFilenamePNG)
