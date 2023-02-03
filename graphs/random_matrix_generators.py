# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def tenpy_random_matrix(N, generator_random_matrix):
    """

    :param N: Dimension of the matrix
    :param generator_random_matrix: random matrix string from tenpy
           "GOE", "GUE", "COE", or "CUE". The argument must be the shape of
           the matrix, i.e., (N, N).
    :return: An instance from a rank perturbed random matrix.
    """
    return generator_random_matrix((N, N))


def perturbed_gaussian(N, L, R, var):
    """
    Perturbed Gaussian random matrix
    :param N: Dimension of the matrix
    :param L: array of shape (N, r), left matrix of the rank perturbation
    :param R: array of shape (r, N), right matrix of the rank perturbation
    :param var: variance of the Gaussian random matrix
    :return: An instance from a rank perturbed gaussian random matrix.
    """
    perturbation = L@R
    gaussian = np.random.normal(0, np.sqrt(var), (N, N))
    # print(np.linalg.norm(L@R), np.linalg.norm(gaussian))
    return perturbation + gaussian

