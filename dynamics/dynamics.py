# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def linear(t, x, W, coupling, D):
    return (D - coupling*W)@x


def lotka_volterra(t, x, W, coupling, D):
    return D@x + coupling*x*(W@x)


def qmf_sis(t, x, W, coupling, D):
    return -D@x + (1 - x)*(coupling*W@x)


def microbial(t, x, W, coupling, D, a, b, c):
    return a - D@x + b*x**2 - c*x**3 + coupling*x*(W@x)


def wilson_cowan(t, x, W, coupling, D, a, b, c):
    return -D@x + (1 - a*x)/(1+np.exp(-b*(coupling*W@x-c)))


def kuramoto_sakaguchi(t, theta, W, coupling, D, alpha):
    return np.diag(D) + coupling*(np.cos(theta+alpha)*(W@np.sin(theta))
                                  - np.sin(theta+alpha)*(W@np.cos(theta)))


def complex_kuramoto_sakaguchi(t, z, W, coupling, D, alpha):
    return 1j*D@z + coupling/2*(W@z*np.exp(-1j*alpha)
                                - (z**2)*(W@np.conj(z))*np.exp(1j*alpha))


def theta(t, theta, W, coupling, Iext):
    return 1 - np.cos(theta) + (1 + np.cos(theta)) * \
        (Iext + coupling*(W@(np.ones(len(theta))-np.cos(theta))))


def winfree(t, theta, W, coupling, omega):
    return omega-coupling*np.sin(theta)*(W@(np.ones(len(theta))+np.cos(theta)))


def lorenz(t, X, W, coupling, a, b, c):
    N = len(X)
    x, y, z = X[0:N], X[N: 2*N], X[2*N:3*N]
    dxdt = a*(y - x) - coupling/N*(x*np.sum(W, axis=1) - W@x)
    dydt = (b*x - y - x*z)
    dzdt = (x*y - c*z)
    return np.concatenate((dxdt, dydt, dzdt))


def rossler(t, X, W, coupling, a, b, c):
    N = len(X)
    x, y, z = X[0:N], X[N: 2*N], X[2*N:3*N]
    dxdt = (-y - z) - coupling/N*(x*np.sum(W, axis=1) - W@x)
    dydt = (x + a*y)
    dzdt = (b - c*z + z*x)
    return np.concatenate((dxdt, dydt, dzdt))
