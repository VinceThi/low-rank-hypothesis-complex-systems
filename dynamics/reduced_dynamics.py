# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from scipy.linalg import pinv


def reduced_wilson_cowan(t, X, L, M, coupling, a, b, c, d):
    N = len(M[0, :])
    return -a*X \
        + M@((np.ones(N) - b*pinv(M)@X) /
             (1+np.exp(-c*(coupling/N*L@X - d))))
