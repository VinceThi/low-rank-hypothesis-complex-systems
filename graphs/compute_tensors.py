# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numpy.linalg import pinv
from numba import njit, prange


@njit(parallel=True)
def compute_tensor_order_3(M, W):
    """
    Compute a tensor in T_{3,n} (order 3 and dimension (n,n,n)) where n is the
    dimension of the reduced system. For instance, this tensor appears in the
    dimension reduction of the Lotka-Volterra dynamics,
    the quenched-mean-field SIS dynamics, the microbial dynamics...
    for which there are nonlinear terms of the form sum_j W_{ij} x_i x_j.
    If we set W = I, the tensor appears when there is a quadratic term x_i^2

    :param M: Reduction matrix. Array of shape (n,N) where N is the dimension
              of the complete system
    :param W: Weight matrix of the network. Array with shape (N,N)

    :return: Tensor T_{3,n}. Array with shape (n,n,n).
    """
    n = len(M[:, 0])
    Mp = pinv(M)
    WMp = W@Mp
    calW = np.zeros((n, n, n))
    for mu in prange(n):
        for nu in prange(n):
            for tau in prange(n):
                calW[mu, nu, tau] += np.sum(M[mu, :]*Mp[:, nu]*WMp[:, tau])
    return calW


def compute_tensor_order_3_nojit(M, W):
    """
    This is a copy of compute_tensor_order_3, but not decorated by jit.
    Use for speed comparison in tests/test_graphs/test_compute_tensors.
    """
    n = len(M[:, 0])
    Mp = pinv(M)
    WMp = W@Mp
    calW = np.zeros((n, n, n))
    for mu in range(n):
        for nu in range(n):
            for tau in range(n):
                calW[mu, nu, tau] += np.sum(M[mu, :]*Mp[:, nu]*WMp[:, tau])
    return calW


def compute_tensor_order_3_tensordot(M, W):
    """
    This does the same as compute_tensor_order_3, but with nested tensordot
    and einsum.
    Use for speed comparison in tests/test_graphs/test_compute_tensors.
    """
    Mp = pinv(M)
    calT = np.tensordot(np.tensordot(M.T, Mp, axes=0), W@Mp, axes=0)
    return np.einsum("iuiviw->uvw", calT)


@njit(parallel=True)
def compute_tensor_order_4(M, W):
    """
    A tensor in T_{4,n} (order 4 and dimension (n, n, n, n)) where n is the
    dimension of the reduced system. This tensor appears in the dimension
    reduction of the Kuramoto dynamics for which there is a nonlinear term
    sum_j W_{ij} z_i^2 bar{z}_j.
    If we set W = I, the tensor appears when there is a cubic term x_i^2

    :param M: Reduction matrix. Array of shape (n,N) where N is the dimension
              of the complete system
    :param W: Weight matrix of the network. Array with shape (N,N)
    :return: Tensor T_{4,N}. Array with shape (n,n,n,n).

    """
    n = len(M[:, 0])
    Mp = np.linalg.pinv(M)
    WMp = W @ Mp
    calW = np.zeros((n, n, n, n))
    for mu in prange(n):
        for nu in prange(n):
            for tau in prange(n):
                for eta in prange(n):
                    calW[mu, nu, tau, eta] \
                        += np.sum(M[mu, :]*Mp[:, nu]*Mp[:, tau]*WMp[:, eta])
    return calW
