# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


import numpy as np


def wilson_cowan(t, x, W, a, b, c):
    return -x + (np.ones(len(x)) - c*x)/(1+np.exp(-a*(W@x-b)))
