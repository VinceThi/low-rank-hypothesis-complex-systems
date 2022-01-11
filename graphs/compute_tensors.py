# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numpy.linalg import pinv

"""
See https://stackoverflow.com/questions/35838037/efficient-reduction-of-multiple-tensors-in-python  
it's faster to vectorize with tensordot before taking the einsum.    
"""


def compute_tensor_order_3(M, W):
    """
    A tensor in T_{3,n} (order 3 and dimension (n, n, n)) where n is the
    dimension of the reduced system. This tensor appears in the dimension
    reduction of the Lotka-Volterra dynamics, the quenched-mean-field SIS
    dynamics, ... for which there is a nonlinear term sum_j W_{ij} x_i x_j.

    :param M: Reduction matrix. Array of shape (n,N) where N is the dimension
              of the complete system
    :param W: Weight matrix of the network. Array with shape (N,N)
    :return: Tensor T_{3,N}. Array with shape (n,n,n).
    """
    Mp = pinv(M)
    calT = np.tensordot(np.tensordot(M.T, Mp, axes=0), W@Mp, axes=0)
    return np.einsum("iuiviw->uvw", calT)


def compute_tensor_order_4(M, W):
    """
    A tensor in T_{4,n} (order 4 and dimension (n, n, n, n)) where n is the
    dimension of the reduced system. This tensor appears in the dimension
    reduction of the Kuramoto dynamics for which there is a nonlinear term
    sum_j W_{ij} z_i^2 bar{z}_j.

    :param M: Reduction matrix. Array of shape (n,N) where N is the dimension
              of the complete system
    :param W: Weight matrix of the network. Array with shape (N,N)
    :return: Tensor T_{4,N}. Array with shape (n,n,n,n).

    """
    Mp = pinv(M)
    calT = np.tensordot(
        np.tensordot(M.T, np.tensordot(Mp, Mp, axes=0), axes=0), W@Mp, axes=0)
    return np.einsum("iuiviwix->uvwx", calT)
