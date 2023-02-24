# -*- coding: utf-8 -*-
# @author: Vincent Thibeault
"""
We modify the code of Ben Erichson at https://github.com/erichson/optht,
which is a Python traduction of the Matlab script optimal_SVHT_coef.m from
Gavish and Donoho 2017.
"""
import numpy as np
from scipy.integrate import quad
import pandas as pd


def marchenko_pastur_pdf(x, var, beta):
    """
    Marchenko-Pastur probability density function
    :param x: (float)
    :param var: (float) variance
    :param beta: (float) aspect ratio of the matrix beta = m/n,
                         where m is the number of rows and n is the
                         number of columns

    :return: The Marchenko-Pastur pdf evaluated at x
    """
    botSpec = var*(1 - np.sqrt(beta))**2
    topSpec = var*(1 + np.sqrt(beta))**2
    if np.all((topSpec - x) * (x - botSpec) > 0):
        return np.sqrt((topSpec - x)*(x - botSpec))/(2*np.pi*var*beta*x)
    else:
        return 0


def marchenko_pastur_generator(var, beta, nb_pts):
    botSpec = var*(1 - np.sqrt(beta))**2
    topSpec = var*(1 + np.sqrt(beta))**2
    x_array = np.linspace(botSpec, topSpec, nb_pts)
    pdf = np.nan_to_num(np.sqrt((topSpec - x_array)*(x_array - botSpec))
                        / (2*np.pi*var*beta*x_array))
    return pd.Series(pdf, index=x_array)


def median_marcenko_pastur(beta):
    """ Compute median of Marcenko-Pastur density with variance 1
     (the standard Marcenko-Pastur law [Bai&Silverstein, 2010, Sec.3.1]).
    Ref: https://github.com/erichson/optht """
    var = 1
    lobnd = var*(1 - np.sqrt(beta))**2   # = botSpec
    topSpec = hibnd = var*(1 + np.sqrt(beta))**2
    change = 1

    while change & ((hibnd - lobnd) > .001):
        change = 0
        x = np.linspace(lobnd, hibnd, 10)
        y = np.zeros_like(x)
        for i in range(len(x)):
            yi = quad(marchenko_pastur_pdf, x[i], topSpec,
                      args=(var, beta))[0]
            y[i] = 1.0 - yi

        if np.any(y < 0.5):
            lobnd = np.max(x[y < 0.5])
            change = 1

        if np.any(y > 0.5):
            hibnd = np.min(x[y > 0.5])
            change = 1

    return (hibnd + lobnd) / 2.
