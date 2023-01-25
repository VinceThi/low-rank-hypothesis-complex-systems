# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import scipy.linalg as la
from plots.plot_singular_values import plot_singular_values
from graphs.get_real_networks import get_learned_weight_matrix,\
    get_microbiome_weight_matrix
from singular_values.compute_effective_ranks import *

# networkName = "00600"
networkName = "gut"
#  "mouse_control_rnn", "zebrafish_rnn", "mouse_rnn",
#  "cnn_nws_main_XXXXX_020" where XXXXX is 00001, 00100, 00200, ..., 00900
singularValuesFilename = 'properties/' + networkName \
                         + '_singular_values.txt'

# if networkName == "zebrafish_rnn":
#     N = 21733
#     singularValues = np.loadtxt(singularValuesFilename)
# elif networkName == "mouse_voxel":
#     N = 15314
#     singularValues = np.loadtxt(singularValuesFilename)

W = get_microbiome_weight_matrix(networkName)
# get_learned_weight_matrix(networkName)
N = len(W[0])
singularValues = la.svdvals(W)

with open(singularValuesFilename, 'wb') as singularValuesFile:
    singularValuesFile.write('# Singular values\n'.encode('ascii'))
    np.savetxt(singularValuesFile, singularValues)


print(computeEffectiveRanks(singularValues, networkName, N))

plot_singular_values(singularValues, effective_ranks=0)

