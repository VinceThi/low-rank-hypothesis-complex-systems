# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numba import jit


# Wilson-Cowan

def reduced_wilson_cowan_factorization(t, x, L, M, a, b):
    return -x + M@(1/(1+np.exp(-a*(L@x - b))))


def reduced_wilson_cowan_suboptimal(t, x, redW, a, b, alpha):
    return -x + alpha*(1/(1+np.exp(-a*((redW@x/alpha)-b))))
