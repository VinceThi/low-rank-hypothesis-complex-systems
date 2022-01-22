# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics import *
import numpy as np


def reduced_lotka_volterra(t, X, calWD_tensor3, coupling, calD):
    """
    Reduced Lotka-Volterra dynamics
    :param t: (float) time
    :param X: (n-dim array) Population size trajectory
    :param calWD_tensor3: (n x n x n array) Third-order interaction and
                                            parameters tensor, related to
                                            W - D/coupling
    :param coupling: (float) Coupling constant
    :param calD: (n x n array) Reduced matrix (typically diagonal) of
                               intrinsic growth rates
    :return: (n-dim array)
     Vector field of the reduced Lotka-Volterra dynamics
    """
    return calD@X + coupling*np.einsum("uvw,v,w->u", calWD_tensor3, X, X)


def reduced_qmf_sis(t, X, calW, calW_tensor3, coupling, calD):
    """
    Reduced Quenched Mean-Field Susceptible-Infected-Susceptible dynamics
    :param t: (float) time
    :param X: (n-dim array) Trajectory of the probability of being infected
    :param calW: (n x n array) Reduced weight matrix
    :param calW_tensor3: (n x n x n array) Third-order interactions
    :param coupling: (float) Coupling constant
    :param calD: (n x n array) Reduced matrix (typically diagonal) of
                               recovery rates

    :return: (n-dim array)
     Vector field of the reduced QMF SIS dynamics
    """
    return (coupling*calW - calD)@X - coupling*np.einsum("uvw,v,w->u",
                                                         calW_tensor3, X, X)


def reduced_kuramoto_sakaguchi(t, Z, calW, calW_tensor4,
                               coupling, calD, alpha):
    """
    Reduced Wilson-Cowan dynamics
    :param t: (float) time
    :param Z: (n-dim array) Complex observable trajectory
    :param calW: (n x n array) Reduced weight matrix
    :param calW_tensor4: (n x n x n array) Fourth-order interactions
    :param coupling: (float) Coupling constant
    :param calD: (n x n array) Reduced matrix (typically diagonal) of
                               natural frequencies
    :param alpha: Phase lag between the oscillators

    :return: (n-dim array)
     Vector field of the reduced kuramoto-sakaguchi dynamics
    """
    return (1j*calD + coupling/2*np.exp(-1j*alpha)*calW)@Z \
        - coupling/2*np.exp(1j*alpha)*np.einsum("uvwx,v,w,x->u", calW_tensor4,
                                                Z, Z, np.conj(Z))


def reduced_wilson_cowan(t, X, L, M, coupling, calD, a, b, c):
    """
    Reduced Wilson-Cowan dynamics defined directly from the vector field.
    :param t: (float) time
    :param X: (n-dim array) Reduced firing-rate (activity) trajectory
    :param L: (Nxn array) L where W = LM is the weight matrix
    :param M: (nxN array) M is the reduction (lumping) matrix
               if W = LM = USV^T, L = US et M = V^T
    :param coupling: (float) Coupling constant
    :param calD: (nxn array) reduced matrix (typically diagonal) of
                             inverse time constant
    :param a: (float) related to the refractory fraction of neuron to fire
    :param b: (float) steepness of the logistic curve
    :param c: (float) midpoint of the logistic curve

    :return: (n-dim array)
     Vector field of the reduced Wilson-Cowan dynamics
    """
    return -calD@X + M @ ((np.ones(len(M[0, :])) - a*np.linalg.pinv(M)@X) /
                          (1 + np.exp(-b*(coupling*L@X - c))))


""" --------- Alternative definitions for the reduced dynamics ------------ """


def reduced_lotka_volterra_vector_field(t, X, W, coupling, M, Mp, D):
    """
    Reduced Lotka-Volterra dynamics defined directly from the vector field.
    :param t: (float) time
    :param X: (n-dim array) Population size trajectory
    :param W: (N x N array) Weight matrix
    :param coupling: (float) Coupling constant
    :param M: (n x N array) Reduction (lumping) matrix
    :param Mp: (n x N array) Moore-Penrose pseudoinv. of the reduction matrix
    :param D: (N x N array) parameter matrix (typically diagonal) of
                            intrinsic growth rates

    :return: (n-dim array)
     Vector field of the reduced Lotka-Volterra dynamics
    """
    return M@lotka_volterra(t, Mp@X, W, coupling, D)


def reduced_qmf_sis_vector_field(t, X, W, coupling, M, Mp, D):
    """
    Reduced Quenched Mean-Field Susceptible-Infected-Susceptible dynamics
    defined directly from the vector field.
    :param t: (float) time
    :param X: (n-dim array) Trajectory of the probability of being infected
    :param W: (N x N array) Weight matrix
    :param coupling: (float) Coupling constant representing the infectionrate
    :param M: (n x N array) Reduction (lumping) matrix
    :param Mp: (n x N array) Moore-Penrose pseudoinv. of the reduction matrix
    :param D: (N x N array) parameter matrix (typically diagonal) of
                            recovery rates

    :return: (n-dim array)
     Vector field of the reduced QMF SIS dynamics
    """
    return M@qmf_sis(t, Mp@X, W, coupling, D)


def reduced_kuramoto_sakaguchi_vector_field(t, Z, W, coupling,
                                            M, Mp, D, alpha):
    """
    Reduced Wilson-Cowan dynamics defined directly from the vector field.
    :param t: (float) time
    :param Z: (n-dim array) Complex observable trajectory
    :param W: (N x N array) Weight matrix
    :param coupling: (float) Coupling constant
    :param M: (n x N array) Reduction (lumping) matrix
    :param Mp: (n x N array) Moore-Penrose pseudoinv. of the reduction matrix
    :param D: (N x N array) parameter matrix (typically diagonal) of
                            natural frequencies
    :param alpha: Phase lag between the oscillators

    :return: (n-dim array)
     Vector field of the reduced kuramoto-sakaguchi dynamics
    """
    return M@complex_kuramoto_sakaguchi(t, Mp@Z, W, coupling, D, alpha)


def reduced_wilson_cowan_vector_field(t, X, W, coupling, M, Mp, D, a, b, c):
    """
    Reduced Wilson-Cowan dynamics
    :param t: (float) time
    :param X: (n-dim array) Reduced firing-rate (activity) trajectory
    :param W: (N x N array) Weight matrix
    :param coupling: (float) Coupling constant
    :param M: (n x N array) Reduction (lumping) matrix
    :param Mp: (n x N array) Moore-Penrose pseudoinv. of the reduction matrix
    :param D: (N x N array) parameter matrix (typically diagonal) of
                            inverse time constant
    :param a: (float) related to the refractory fraction of neuron to fire
    :param b: (float) steepness of the logistic curve
    :param c: (float) midpoint of the logistic curve

    :return: (n-dim array)
     Vector field of the reduced Wilson-Cowan dynamics
    """
    return M @ wilson_cowan(t, Mp @ X, W, coupling, D, a, b, c)
