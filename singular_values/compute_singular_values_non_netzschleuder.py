# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import scipy.linalg as la
from plots.plot_singular_values import plot_singular_values
from graphs.get_real_networks import *
from singular_values.compute_effective_ranks import *

graph_str = "connectome"
save_data = True
compute_effective_ranks = True
plot_singular_vals = True


if graph_str == "learned":
    networkName = "fully_connected_layer_cnn_01000"
    #  "mouse_control_rnn", "zebrafish_rnn", "mouse_rnn",
    # "fully_connected_layer_cnn_00100", "fully_connected_layer_cnn_00200",...,
    #  "fully_connected_layer_cnn_01000"
    singularValuesFilename = 'properties/' + networkName \
                             + '_singular_values.txt'

    # if networkName == "zebrafish_rnn":
    #     N = 21733
    #     singularValues = np.loadtxt(singularValuesFilename)
    # elif networkName == "mouse_voxel":
    #     N = 15314
    #     singularValues = np.loadtxt(singularValuesFilename)

    W = get_learned_weight_matrix(networkName)
    N = len(W[0])
    singularValues = la.svdvals(W)

elif graph_str == "economic":
    networkName = "non_financial_institution04-Jan-2001"
    # "AT_2008", "CY_2015", "EE_2010", "PT_2009", "SI_2016",
    #  "financial_institution07-Apr-1999", "households_04-Sep-1998",
    # "households_09-Jan-2002", "non_financial_institution04-Jan-2001"
    singularValuesFilename = 'properties/' + networkName \
                             + '_singular_values.txt'

    W = get_economic_weight_matrix(networkName)
    N = len(W[0])
    singularValues = la.svdvals(W)

elif graph_str == "connectome":
    networkName = "mouse_meso"   # "celegans_signed"
    # "mouse_meso", "zebrafish_meso", "celegans",
    # "celegans_signed", "drosophila", "ciona",
    #   "platynereis_dumerilii_neuronal", "platynereis_dumerilii_desmosomal"
    singularValuesFilename = 'properties/' + networkName \
                             + '_singular_values.txt'

    W = get_connectome_weight_matrix(networkName)
    N = len(W[0])
    singularValues = la.svdvals(W)

elif graph_str == "microbiome":
    networkName = "gut"
    singularValuesFilename = 'properties/' + networkName \
                             + '_singular_values.txt'

    W = get_microbiome_weight_matrix(networkName)
    N = len(W[0])
    singularValues = la.svdvals(W)

if save_data:
    with open(singularValuesFilename, 'wb') as singularValuesFile:
        singularValuesFile.write('# Singular values\n'.encode('ascii'))
        np.savetxt(singularValuesFile, singularValues)

if compute_effective_ranks:
    print(computeEffectiveRanks(singularValues, networkName, N))

if plot_singular_vals:
    plot_singular_values(singularValues, effective_ranks=0)
