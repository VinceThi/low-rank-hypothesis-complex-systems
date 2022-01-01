# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def reduced_wilson_cowan(t, X, L, M, coupling, calD, a, b, c):
    """
    Reduced Wilson-Cowan dynamics
    :param t: (float) time
    :param X: (n-dim array) Reduced firing-rate (activity) trajectory
    :param L: (Nxn array) L where W = LM is the weight matrix
    :param M: (nxN array) M is the reduction (lumping) matrix
               if W = LM = USV^T, L = US et M = V^T
    :param coupling: (float) coupling constant
    :param calD: (nxn array) reduced matrix (typically diagonal) of
                             inverse time constant
    :param a: (float) related to the refractory fraction of neuron to fire
    :param b: (float) steepness of the logistic curve
    :param c: (float) midpoint of the logistic curve
    :return: (n-dim array)
     Vector field of the reduced Wilson-Cowan dynamics 
    """
    N = len(M[0, :])
    return -calD@X + M @ ((np.ones(N) - a*np.linalg.pinv(M)@X) /
                          (1 + np.exp(-b*(coupling/N*L@X - c))))


def reduced_lotka_volterra(t, X, calW_tensor3, coupling, calD, N):
    """
    Reduced Lotka-Volterra dynamics
    :param t: (float) time
    :param X: (n-dim array) Population size trajectory
    :param calW_tensor3: (n x n x n array) third-order interactions
    :param coupling: (float) coupling constant
    :param calD: (n x n array) reduced matrix (typically diagonal) of
                               intrinsic growth rates
    :return: (n-dim array)
     Vector field of the reduced Lotka-Volterra dynamics
    """
    return -calD@X + coupling/N*np.einsum("uvw,v,w->u", calW_tensor3, X, X)


def reduced_qmf_sis(t, X, calW, calW_tensor3, coupling, calD, N):
    """
    Reduced Quenched Mean-Field Susceptible-Infected-Susceptible dynamics
    :param t: (float) time
    :param X: (n-dim array) Trajectory of the probability of being infected
    :param calW: (n x n array) Reduced weight matrix
    :param calW_tensor3: (n x n x n array)
    :param coupling: (float) coupling constant
    :param calD: (n x n array) reduced matrix (typically diagonal) of
                               recovery rates
    :return: (n-dim array)
     Vector field of the reduced QMF SIS dynamics
    """
    return (coupling/N*calW-calD)@X \
        + coupling/N*np.einsum("uvw,v,w->u", calW_tensor3, X, X)
