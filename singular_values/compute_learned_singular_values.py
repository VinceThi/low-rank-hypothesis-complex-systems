# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import scipy.linalg as la
from plots.plot_singular_values import plot_singular_values
from graphs.get_real_networks import get_learned_weight_matrix,\
    get_microbiome_weight_matrix
from singular_values.compute_effective_ranks import *

networkName = "fully_connected_layer_cnn_01000"
#  "mouse_control_rnn", "zebrafish_rnn", "mouse_rnn",
#  "cnn_nws_main_XXXXX_020" where XXXXX is
#  fully_connected_layer_cnn_00100,..., fully_connected_layer_cnn_01000
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

with open(singularValuesFilename, 'wb') as singularValuesFile:
    singularValuesFile.write('# Singular values\n'.encode('ascii'))
    np.savetxt(singularValuesFile, singularValues)


print(computeEffectiveRanks(singularValues, networkName, N))

plot_singular_values(singularValues, effective_ranks=0)

