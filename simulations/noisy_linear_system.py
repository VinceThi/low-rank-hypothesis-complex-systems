# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
from dynamics.dynamics import linear
from plots.plot_weight_matrix import plot_weight_matrix

""" Time parameters """
t0, t1, dt = 0, 100, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))


""" Graph parameters """
graph_str = "gaussian"
N = 100
noise_level = 1/np.sqrt(N)
gaussian_random_matrix = np.random.normal(0, 0.2, (N, N))
cut_value = 60
X = np.block([[np.ones(N - cut_value), np.zeros(cut_value)],
              [np.zeros(N - cut_value), np.ones(cut_value)]])
unknown_weight_matrix = X.T@X
W = unknown_weight_matrix + noise_level*gaussian_random_matrix
plot_weight_matrix(W)

""" Dynamical parameters """
dynamics_str = "linear"
coupling_constants = np.linspace(0.01, 5, 50)
D = 0*np.eye(N)




