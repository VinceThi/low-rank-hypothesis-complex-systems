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
                    'graph_data/datasets_table.txt'
header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
graphPropDF = pd.read_table(graphPropFilename, names=header,
                            comment="#", delimiter=r"\s+")
graphPropDF.set_index('Name', inplace=True)

# Connectome tag
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,Connectome", 'Tags'] = "Connectome"

# Ecological tag
graphPropDF.loc[graphPropDF['Tags'] ==
                "Social,Animal", 'Tags'] = "Ecological"
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,Foodweb", 'Tags'] = "Ecological"
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,FoodWeb,Uncertain", 'Tags'] = "Ecological"
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,Foodweb,Multilayer", 'Tags'] = "Ecological"

# Interactome tag
# See Interactome Networks and Human Disease by Vidal et al. for more info
# on interactomes. We include the drug-drug interactions in the
# interactome category.
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,Proteininteractions", 'Tags'] = "Interactome"
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,Generegulation", 'Tags'] = "Interactome"
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,Generegulation,Proteininteractions,Multilayer",
                'Tags'] = "Interactome"
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,Genetic,Projection,Multilayer",
                'Tags'] = "Interactome"
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,Metabolic", 'Tags'] = "Interactome"
graphPropDF.loc[graphPropDF['Tags'] ==
                "Biological,Druginteractions", 'Tags'] = "Interactome"

# Economic tag
graphPropDF.loc[graphPropDF['Tags'] ==
                "Economic,Trade,Multilayer", 'Tags'] = "Economic"

# Communication tag
graphPropDF.loc[graphPropDF['Tags'] ==
                "Social,Communication", 'Tags'] = "Communication"
graphPropDF.loc[graphPropDF['Tags'] == "Social,Communication,Timestamps",
                "Tags"] = "Communication"

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

    cat = [tag for tag in graphPropDF.loc[networkName]['Tags'].split(',')
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

theoretical_bound = 0.0915
theoretical_bound_avg = 0.01445
# 95th      96th     97th     Average    50th
# 0.0915   0.102    0.114     0.01445    0.00911
# ^ Computed with Wolfram alpha:
# https://www.wolframalpha.com/input?i=2F1%281%2C+2*0.53985635%2C+2*%282.34169587+-+1%29%2C+-+25.45627526%29%2F%282*2.34169587+-+3%29+++
# https://www.wolframalpha.com/input?i=integrate+%28%281+-+y%29%5E%282.34+-+2%29%2F%281+%2B+25.456*y%29%5E0.54%29%5E2+from+0+to+1+
nb_network_below_bound =\
    np.count_nonzero(effectiveRanksDF['y'] < theoretical_bound)
percentage_below_bound = 100*nb_network_below_bound/total_number_networks

nb_network_below_bound_avg =\
    np.count_nonzero(effectiveRanksDF['y'] < theoretical_bound_avg)
percentage_below_bound_avg =\
    100*nb_network_below_bound_avg/total_number_networks

x = np.linspace(0.55, 1.02, 100)
xavg = np.linspace(0.05, 1.02, 100)
plt.plot(x, theoretical_bound*np.ones(len(x)), color="#38ccf9", linewidth=1)
# plt.plot(xavg, theoretical_bound_avg*np.ones(len(x)),
#  color="#38ccf9", linewidth=1)
plt.text(x[5], theoretical_bound+0.003, f"Theoretical bound", fontsize=8)
plt.text(x[5], theoretical_bound-0.013,
         f"{int(np.round(percentage_below_bound))}% of the\nnetworks",
         fontsize=8)

print(f"Total number of networks = {total_number_networks}\n"
      f"Number of networks below the bound = {nb_network_below_bound}"
      f"\nthat is, {np.round(percentage_below_bound, 4)}% of the networks.")

print(f"{np.round(percentage_below_bound_avg, 4)}% of networks below the bound"
      f"when the singular value enveloping curve is for the avg ")

if total_number_networks != 679:
    raise ValueError("One much change the number of networks. 679 networks is "
                     "hard coded in other places in the script.")

plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=0.22)
plt.legend(loc=2)

plt.xlabel('rank/N')
plt.ylabel('srank/N')
plt.show()

plt.savefig(figureFilenamePDF)
plt.savefig(figureFilenamePNG)
