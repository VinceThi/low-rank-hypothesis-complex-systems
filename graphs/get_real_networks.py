# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import networkx as nx
import numpy as np
import pandas as pd
import scipy.io
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


def get_connectome_weight_matrix(graph_name):
    """
    Return the weight matrix for a given graph.
    graph_name (str): "mouse_meso", "zebrafish_meso", "celegans",
                      "celegans_signed", "drosophila", "ciona"
    """
    path_str = "C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
               "graphs/graph_data/connectomes/"

    if graph_name == "celegans":
        # Data obtained from Mohamed Bahdine, extracted as described in the
        # supplementary material of the article : Network control principles
        # predict neuron function in the C. elegans connectome - Yan et al.
        # The data come from Wormatlas.
        A = np.array(1 * np.load(path_str + "C_Elegans.npy"))
        # N = 279
        # rank_celegans = 273

    elif graph_name == "celegans_signed":
        # Data: https://elegansign.linkgroup.hu/#!NT+R%20method%20prediction
        # Paper: https://doi.org/10.1371/journal.pcbi.1007974
        df = pd.read_excel(
            path_str + 'celegans_weighted_directed_signed.xls',
            usecols="A,D,E,P")
        df = df.replace(to_replace=['+', '-', 'no pred', 'complex'],
                        value=[1, -1, 0, 0])
        df_dale = df[["Source"]]
        df_dale["Strength x Sign"] = df["Edge Weight"] * df["Sign"]
        df_dale = df_dale.groupby(['Source']).sum()
        # We complete the missing data using Dale's principle, i.e., if most
        # of the synapses of a neuron are excitatory (inhibitory), then we
        #  consider that the unknown ones are excitatory (inhibitory). When no
        # information allows to apply Dale's principle, we consider the neuron
        # as an excitator given the fact that there are more excitators than
        # inhibitors in connectomes.
        for i, neuron in enumerate(df["Source"]):
            if df["Sign"].values[i] == 0 \
                    and df_dale.loc[neuron].values[0] >= 0:
                df.loc[i, ['Sign']] = 1
            elif df["Sign"].values[i] == 0 \
                    and df_dale.loc[neuron].values[0] < 0:
                df.loc[i, ['Sign']] = -1
        df["Weight x Sign"] = df["Edge Weight"] * df["Sign"]
        # with pd.option_context('display.max_rows', None,
        #                        'display.max_columns', None):
        #     print(df)
        # print(df["Weight x Sign"].sum())
        G_celegans = nx.from_pandas_edgelist(df,
                                             source='Source',
                                             target='Target',
                                             edge_attr='Weight x Sign',
                                             create_using=nx.DiGraph())
        A = nx.to_numpy_array(G_celegans, weight='Weight x Sign')
        # N = 297

    elif graph_name == "drosophila":
        df = pd.read_csv(
            path_str + 'drosophila_exported-traced-adjacencies-v1.1/'
                       'traced-total-connections.csv')
        Graphtype = nx.DiGraph()
        G_drosophila = nx.from_pandas_edgelist(df,
                                               source='bodyId_pre',
                                               target='bodyId_post',
                                               edge_attr='weight',
                                               create_using=Graphtype)
        A = nx.to_numpy_array(G_drosophila, weight='weight')
        # N = 21733

    elif graph_name == "ciona":
        A_from_xlsx = pd.read_excel(path_str +
                                    'ciona_intestinalis_lavaire_elife-16962'
                                    '-fig16-data1-v1_modified.xlsx').values
        A_ciona_nan = np.array(A_from_xlsx[0:, 1:])
        A_ciona = np.array(A_ciona_nan, dtype=float)
        where_are_NaNs = np.isnan(A_ciona)
        A_ciona[where_are_NaNs] = 0
        A = A_ciona
        # A = (A_ciona > 0).astype(float)
        # N = 213
        # rank = 203

    elif graph_name == "mouse_meso":
        # Oh, S., Harris, J., Ng, L. et al.
        # A mesoscale connectome of the mouse brain.
        # Nature 508, 207–214 (2014) doi:10.1038/nature13186
        # To binary matrix  (with "> 0")
        # A = (np.loadtxt(path_str + "ABA_weight_mouse.txt") > 0).astype(float)
        A = (np.loadtxt(path_str + "ABA_weight_mouse.txt")).astype(float)
        # N = 213
        # rank = 185

    elif graph_name == "zebrafish_meso":
        # Kunst et al.
        # "A Cellular-Resolution Atlas of the Larval Zebrafish Brain",
        # (2019) avec le traitement de Antoine Légaré
        # On a pas exactement les mêmes régions que l'article non plus,
        #  où la matrice est faite avec 36 régions. Ici, on en a 71 qui sont
        #  mutually exclusive et collectively exhaustive (je reprends les
        # termes du dude dans le courriel) donc ça couvre tout le volume au
        #  complet sans overlap

        df = pd.read_csv(path_str +
                         'Connectivity_matrix_zebra_fish_mesoscopic.csv')
        dictio = {'X': 0}  # We put zeros temporarily on the diagonal
        df = df.replace(dictio)

        volumes = np.array(
            1 * np.load(path_str + "volumes_zebrafish_meso.npy"))
        relativeVolumes = volumes / sum(volumes)
        adjacency = df.to_numpy()[:, 1:-1].astype(float)
        # """ To get an undirected graph """
        # for i in range(adjacency.shape[0]):
        #     for j in range(i+1, adjacency.shape[0]):
        #         adjacency[i, j] = (adjacency[i, j] + adjacency[j, i]) /
        #  (relativeVolumes[i] + relativeVolumes[j])
        #         adjacency[j, i] = adjacency[i, j]
        """ To get a directed graph """
        for i in range(adjacency.shape[0]):
            for j in range(adjacency.shape[0]):
                adjacency[i, j] = adjacency[i, j] / (
                        relativeVolumes[i] + relativeVolumes[j])
        adjacency = adjacency / np.amax(adjacency)
        adjacency = np.log(adjacency + 0.00001)
        adjacency -= np.amin(adjacency)
        adjacency = adjacency / np.amax(adjacency)
        # We add a diagonal because there are interactions within modules
        A = adjacency + np.eye(len(adjacency[0]))
        # N = 71
        # rank_zebrafish_meso = 71

    else:
        raise ValueError("This graph_str connectome is not an option. "
                         "See the documentation of "
                         "get_connectome_weight_matrix")

    return A


def get_microbiome_weight_matrix(graph_str):

    path_str = f"C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
               f"graphs/graph_data/microbiomes/{graph_str}/"

    if graph_str == "gut":
        # See p.27-28 of the supplementary information of
        # Reviving a failed network through microscopic interventions
        # R. Lim, J.J.T. Cabatbat, T.L.P. Martin, H. Kim, S. Kim, J. Sung,
        # C.-M. Ghim and P.-J. Kim. Large- scale metabolic interaction network
        # of the mouse and human gut microbiota. Scientific Data, 7, 204, 2020.
        dictionary = scipy.io.loadmat(path_str+'MicrobiomeNetworks.mat')
        P = dictionary['complementarity']
        Q = dictionary['competition']

        # These are the parameters mentionned in p.28 of the SI
        omegaP = 30
        omegaQ = 1

        A = omegaP*P - omegaQ*Q

    else:
        raise ValueError("This graph_str microbiome is not an option. See the"
                         " documentation of get_microbiome_weight_matrix")

    return A


def get_foodweb_weight_matrix(graph_str):

    path_str = f"C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
               f"graphs/graph_data/foodwebs/{graph_str}/"

    if graph_str == "little_rock":
        # Taken from Netzschleuder : https://networks.skewed.de/
        G = nx.read_edgelist(path_str + "edges.csv", delimiter=',',
                             create_using=nx.DiGraph)
        A = nx.to_numpy_array(G).T
        # A_{ij} = 1, if edge j -> i (other convention in the data)

    elif graph_str == "caribbean":
        # Taken from Web of life : https://www.web-of-life.es/
        A = np.genfromtxt(path_str + 'foodweb_caribbean_matrix.csv',
                          delimiter=",")
        A[np.isnan(A)] = 0

        # Note: Canibalism has been found in 11 species or groups.
        # Crabs, Shrimps, Polychaetes, Gastropods, Squids, Octopuses,
        # Asteroids, Echinoids, Mycteroperca venenosa,
        # Scomberomorus cavalla, Tylosurus acus
        # We should add self-loops, because they are not included right now

    else:
        raise ValueError("This graph_str foodweb is not an option. "       
                         "See the documentation of get_foodweb_weight_matrix")

    return A


def get_epidemiological_weight_matrix(graph_str):
    path_str = f"C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
               f"graphs/graph_data/epidemiological/{graph_str}/"

    if graph_str == "high_school_proximity":
        # Taken from Netzschleuder : https://networks.skewed.de/
        G = nx.read_edgelist(path_str + "edges_no_time.csv", delimiter=',',
                             create_using=nx.Graph)
        A = nx.to_numpy_array(G)

    else:
        raise ValueError("This graph_str epidemiological is not an option. "       
                         "See the documentation of"
                         "get_epidemiological_weight_matrix")

    return A


def get_learned_weight_matrix(graph_str):
    path_str = f"C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
               f"graphs/graph_data/learned/{graph_str}/"

    if graph_str == "zebrafish_rnn":
        W = np.load(path_str + "zebrafish1-presz-model.npz")["J"]

    elif graph_str == "mouse_rnn":
        W = np.load(path_str + "mouse-tle1-model.npz")["J"]

    elif graph_str == "mouse_control_rnn":
        W = np.load(path_str + "mouse-control1-model.npz")["J"]

    else:
        raise ValueError("This graph_str learned is not an option. "       
                         "See the documentation of"
                         "get_learned_weight_matrix")

    return W
