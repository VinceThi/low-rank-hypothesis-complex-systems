# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def lotka_volterra(t, x, W, coupling, a):
    N = len(W[:, 0])
    return -a*x + coupling/N*x*(W@x)


def qmf_sis(t, x, W, coupling, a):
    N = len(W[:, 0])
    return -a*x + (np.ones(len(x)) - x)*(coupling/N*W@x)


def wilson_cowan(t, x, W, coupling, a, b, c, d):
    N = len(W[:, 0])
    return -a*x + (np.ones(len(x)) - b*x)/(1+np.exp(-c*(coupling/N*W@x-d)))


def kuramoto_sakaguchi(t, theta, W, coupling, omega, alpha):
    N = len(W[:, 0])
    return omega \
        + coupling/N*(np.cos(theta+alpha)*np.dot(W, np.sin(theta))
                      - np.sin(theta+alpha)*np.dot(W, np.cos(theta)))


def theta(t, theta, W, coupling, Iext):
    N = len(W[:, 0])
    return 1 - np.cos(theta) + (1 + np.cos(theta)) * \
        (Iext + coupling/N*(W@(np.ones(N)-np.cos(theta))))


def winfree(t, theta, W, coupling, omega):
    N = len(W[:, 0])
    return omega - coupling/N*np.sin(theta)*(W@(np.ones(N) + np.cos(theta)))


def lorenz(t, X, W, coupling, a, b, c):
    N = len(W[0])
    x, y, z = X[0:N], X[N: 2*N], X[2*N:3*N]
    dxdt = a*(y - x) - coupling/N*(x*np.sum(W, axis=1) - W@x)
    dydt = (b*x - y - x*z)
    dzdt = (x*y - c*z)
    return np.concatenate((dxdt, dydt, dzdt))


def rossler(t, X, W, coupling, a, b, c):
    N = len(W[0])
    x, y, z = X[0:N], X[N: 2*N], X[2*N:3*N]
    dxdt = (-y - z) - coupling/N*(x*np.sum(W, axis=1) - W@x)
    dydt = (x + a*y)
    dzdt = (b - c*z + z*x)
    return np.concatenate((dxdt, dydt, dzdt))
