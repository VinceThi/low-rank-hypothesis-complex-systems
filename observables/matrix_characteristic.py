# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from numpy.linalg import det
import numpy as np


def matrix_is_singular(M):
    boolean = 0
    if np.abs(det(M)) < 1e-8:
        boolean = 1
    return boolean


def matrix_is_negative(M):
    return np.any(M < -1e-8)


def matrix_has_rank_n(M):
    n = len(M[:, 0])
    return np.linalg.matrix_rank(M) == n


def matrix_is_normalized(M):
    if len(np.shape(M)) == 1:
        bool_value = np.absolute(np.sum(M) - 1) < 0.000001
    else:
        n = len(M[:, 0])
        bool_value = \
            np.all(np.absolute(np.sum(M, axis=1) - np.ones(n)) < 0.000001)
    return bool_value


def matrix_is_orthogonal(M):
    boolean = 0
    X = np.abs(M@M.T - np.identity(np.shape(M)[0]))
    Y = np.abs(X - np.diag(np.diag(X)))
    # Because we don't want to check normalization, we substract the diagonal
    # of X to X to get Y. If it is a zero matrix, then M is orthogonal.
    if np.all(Y < 1e-8):
        boolean = 1
    return boolean


def matrix_is_orthonormalized_MM_T(M):
    n = len(M[:, 0])
    return np.all(np.absolute(M@M.T - np.identity(n)) < 1e-8)


def matrix_is_positive(M):
    return np.all(M >= -1e-8)
