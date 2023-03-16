# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import scipy.linalg as la
import matplotlib.pyplot as plt
from plots.plot_singular_values import plot_singular_values
from plots.plot_weight_matrix import plot_weight_matrix
from graphs.get_real_networks import *
from singular_values.compute_effective_ranks import *

graph_str = "epidemiological"
plot_degrees = False
plot_weight_mat = False
save_data = False
compute_effective_ranks = True
plot_singular_vals = False


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

elif graph_str == "epidemiological":
    networkName = "high_school_proximity"
    singularValuesFilename = 'properties/' + networkName \
                             + '_singular_values.txt'
    W = get_epidemiological_weight_matrix(networkName)
    singularValues = la.svdvals(W)
    N = len(W[0])

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
    networkName = "cintestinalis"   # "celegans_signed"
    # "mouse_meso", "zebrafish_meso", "celegans",
    # "celegans_signed", "drosophila", "cintestinalis",
    #   "pdumerilii_neuronal", "pdumerilii_desmosomal"
    singularValuesFilename = 'properties/' + networkName \
                             + '_singular_values.txt'

    W = get_connectome_weight_matrix(networkName)
    N = len(W[0])
    # print(np.sum(W), np.sum(W > 0)/2, np.all(W == W.T))
    singularValues = la.svdvals(W)

elif graph_str == "microbiome":
    networkName = "gut"
    singularValuesFilename = 'properties/' + networkName \
                             + '_singular_values.txt'

    W = get_microbiome_weight_matrix(networkName)
    N = len(W[0])
    singularValues = la.svdvals(W)

if plot_degrees:
    plt.figure(figsize=(6, 4))
    plt.hist(np.sum(W, axis=1), bins=100, label="$\\kappa_{in}$",
             density=True)
    plt.hist(np.sum(W, axis=0), bins=100, label="$\\kappa_{out}$",
             density=True)
    plt.ylabel("Density")
    plt.legend(loc=1)
    plt.show()

if plot_weight_mat:
    plot_weight_matrix((np.abs(W) > 0).astype(float))

if save_data:
    with open(singularValuesFilename, 'wb') as singularValuesFile:
        singularValuesFile.write('# Singular values\n'.encode('ascii'))
        np.savetxt(singularValuesFile, singularValues)

if compute_effective_ranks:
    print(computeEffectiveRanks(singularValues, networkName, N))

if plot_singular_vals:
    plot_singular_values(singularValues, effective_ranks=0)
