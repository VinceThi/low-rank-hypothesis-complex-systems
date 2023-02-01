# -*- coding: utf-8 -*-\\
# @author: Vincent Thibeault
import numpy as np
import tenpy as tp
from plots.plot_singular_values import \
    plot_singular_values_histogram_random_matrices, plot_singular_values
from scipy.linalg import svdvals
from plots.plot_weight_matrix import plot_weight_matrix

plot_histogram = False
N = 500

if plot_histogram:
    generator = tp.linalg.random_matrix.COE  # GOE, GUE, CUE
    args = [(N, N)]

    plot_singular_values_histogram_random_matrices(generator, args,
                                                   nb_networks=500,
                                                   nb_bins=500)

else:  # show scree plot of one instance
    A = tp.linalg.random_matrix.CUE((N, N))
    plot_weight_matrix(np.real(A))
    plot_weight_matrix(np.imag(A))
    singularValues = svdvals(A)
    print(singularValues)
    plot_singular_values(singularValues)
