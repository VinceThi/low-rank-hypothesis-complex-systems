# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numpy.linalg import pinv, solve, norm


def mse(a, b):
    return np.mean((a - b)**2)


def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))


def rmse_compatibility_equation(M, X):
    return rmse(M@X, M@X@pinv(M)@M)


def relative_error_vector_fields(M, vfield_true, vfield_approximate,
                                 args_vfield_true, args_vfield_approximate):
    """

    :param M: (nxN array) Reduction matrix
    :param vfield_true: (N-dim array) Complete vector field
    :param vfield_approximate: (n-dim array) Reduced vector field
    :param args_vfield_true: (tuple) Arguments of the complete vector field
    :param args_vfield_approximate: (tuple) Arguments of the reduced v. field

    :return: Relative error between the reduced and the complete vector fields
    """
    n, N = np.shape(M)
    diff = mse(M@vfield_true(*args_vfield_true),
               vfield_approximate(*args_vfield_approximate))
    normalization = mse(M@vfield_true(*args_vfield_true), np.zeros(n))
    numerical_zero = 1e-13
    if normalization < numerical_zero:
        if diff < numerical_zero:
            return 0
        else:
            raise ValueError("The relative error is undefined. The denominator"
                             "is 0 but not the numerator.")
    else:
        return diff/normalization


# ------------------------- Errors Wilson-Cowan -------------------------------
def sigmoid(x):
    return 1/(1+np.exp(-x))


def jacobian_x_wilson_cowan(x, W, coupling, D, a, b, c):
    return -D - a*np.diag(sigmoid(b*(coupling*W@x-c)))


def jacobian_y_wilson_cowan(x, W, coupling, a, b, c):
    return b*coupling*(np.eye(len(x)) - a*np.diag(x))\
           * (sigmoid(b*(coupling*W@x-c))*(1 - sigmoid(b*(coupling*W@x-c))))


# ------------------------------ Errors RNN -----------------------------------
def jacobian_y_rnn(y, coupling):
    return 4*coupling*(np.eye(len(y)))\
           * (sigmoid(2*coupling*y)*(1 - sigmoid(2*coupling*y)))


def y_prime_SIS(x, W, M, coupling, sign):
    P = pinv(M)@M
    chi = (np.eye(len(x)) - P)@x
    d = np.diag(W@chi)**(-1)\
        * (sigmoid(2*coupling*W@x) - sigmoid(2*coupling*W@P@x))
    return np.log((1+sign*np.sqrt(1 - 4*d))/(1-sign*np.sqrt(1 - 4*d))) / \
        (2*coupling)


def error_upper_bound_rnn(x, y_prime, coupling, D, n, s, M):
    """
    :param x: (N-dim array) position in
    :param y_prime: (N dim array) point between Wx and WPx
    :param coupling: (float) infection rate
    :param D: (NxN array) recovery rate diagonal matrix
    :param n: (int) Dimension of the reduced system
    :param s: (N-dim array) Singular values of W
    :param M: (nxN array )Reduction matrix = V_n^T = n-truncated, transposed,
                          right singular vector of the weight matrix W
    :return: (float)
    Upper bound on the error between the vector field of the N-dimensional
    qmf sis dynamics and its least-square optimal reduction

    """
    P = pinv(M)@M
    chi = (np.eye(len(x)) - P)@x
    Jx = -D
    Jy = jacobian_y_rnn(y_prime, coupling)
    return (norm(M@Jx@chi) + s[n]*norm(M@Jy, ord=2)*norm(x))/np.sqrt(n)


# -------------------------------- Errors SIS ---------------------------------
def jacobian_x_SIS(x, W, coupling, D):
    return -D - coupling*np.diag(W@x)


def jacobian_y_SIS(x, coupling):
    return coupling*(np.eye(len(x)) - np.diag(x))


def x_prime_SIS(x, W, M):
    P = pinv(M)@M
    chi = (np.eye(len(x)) - P)@x
    A = np.diag(chi)@W + np.diag(W@chi)
    b = x*(W@x) - (P@x)*(W@P@x)
    return solve(A, b)


def error_upper_bound_SIS(x, x_prime, W, coupling, D, n, s, M):
    """
    :param x: (N-dim array) position in
    :param x_prime: (N dim array) point between x and Px
    :param W: (NxN array) Weight matrix
    :param coupling: (float) infection rate
    :param D: (NxN array) recovery rate diagonal matrix
    :param n: (int) Dimension of the reduced system
    :param s: (N-dim array) Singular values of W
    :param M: (nxN array )Reduction matrix = V_n^T = n-truncated, transposed,
                          right singular vector of the weight matrix W
    :return: (float)
    Upper bound on the error between the vector field of the N-dimensional
    qmf sis dynamics and its least-square optimal reduction

    """
    P = pinv(M)@M
    chi = (np.eye(len(x)) - P)@x
    Jx = jacobian_x_SIS(x_prime, W, coupling, D)
    Jy = jacobian_y_SIS(x_prime, coupling)
    return (norm(M@Jx@chi) + s[n]*norm(M@Jy, ord=2)*norm(x))/np.sqrt(n)


def error_upper_bound_SIS_no_triangle(x, x_prime, W, coupling, D, n, M):
    P = pinv(M)@M
    chi = (np.eye(len(x)) - P)@x
    Jx = jacobian_x_SIS(x_prime, W, coupling, D)
    Jy = jacobian_y_SIS(x_prime, coupling)
    # This is exact (verified numerically)
    return (norm(M@Jx@chi + M@Jy@W@(np.eye(len(x)) - P)@x))/np.sqrt(n)


def error_upper_bound_SIS_no_induced_norm(x, x_prime, W, coupling, D, n, M):
    P = pinv(M)@M
    chi = (np.eye(len(x)) - P)@x
    Jx = jacobian_x_SIS(x_prime, W, coupling, D)
    Jy = jacobian_y_SIS(x_prime, coupling)
    return (norm(M@Jx@chi) + norm(M@Jy@W@(np.eye(len(x)) - P)@x))/np.sqrt(n)


def error_upper_bound_SIS_no_submul(x, x_prime, W, coupling, D, n, M):
    P = pinv(M)@M
    chi = (np.eye(len(x)) - P)@x
    Jx = jacobian_x_SIS(x_prime, W, coupling, D)
    Jy = jacobian_y_SIS(x_prime, coupling)
    return (norm(M@Jx@chi)
            + norm(M@Jy@W@(np.eye(len(x)) - P), ord=2)*norm(x))/np.sqrt(n)


def error_upper_bound_SIS_no_eckart(x, x_prime, W, coupling, D, n, M):
    P = pinv(M)@M
    chi = (np.eye(len(x)) - P)@x
    Jx = jacobian_x_SIS(x_prime, W, coupling, D)
    Jy = jacobian_y_SIS(x_prime, coupling)
    return (norm(M@Jx@chi)
            + norm(M@Jy, ord=2) *
            norm(W@(np.eye(len(x)) - P), ord=2)*norm(x))/np.sqrt(n)



