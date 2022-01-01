# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numpy.linalg import pinv


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

    See https://stackoverflow.com/questions/35838037/efficient-reduction-of-multiple-tensors-in-python
    it's better to vectorize with tensordot before taking the einsum.
    """
    Mp = pinv(M)
    calT = np.tensordot(np.tensordot(M.T, Mp, axes=0), W@Mp, axes=0)
    return np.einsum("iuiviw->uvw", calT)
