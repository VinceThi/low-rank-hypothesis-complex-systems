# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from singular_values.marchenko_pastur_pdf import marchenko_pastur_generator
from plots.plot_singular_values import plot_singular_values
from scipy.linalg import svdvals
from plots.config_rcparams import *
import numpy as np


def test_marchenko_pastur_generator_gaussian_matrix():
    """ For this example, we assume that nbrows >= nbcolumns """
    plot_scree = False
    nbrows = 200
    nbcolumns = 100
    beta = nbrows/nbcolumns
    var = 0.5
    nbinstances = 1000
    singularValues = np.array([])
    for i in range(nbinstances):
        W = np.random.normal(0, np.sqrt(var), (nbrows, nbcolumns))

        """ First way """
        # Y = W@W.T/nbcolumns
        # eigenvals, _ = np.linalg.eig(Y)
        # singularValues_instance2 = np.nan_to_num(np.sqrt(np.real(eigenvals)))

        """ Second way """
        singularValues_instance = np.zeros(nbrows)
        # ^Important to have nbrows singular values, since Marchenko-Pastur pdf
        # is for squared singular values (eigenvalues) of Y = W@W.T/nbcolumns
        singularValues_instance[:nbcolumns] = svdvals(W)/np.sqrt(nbcolumns)

        singularValues = np.concatenate((singularValues,
                                         singularValues_instance))

        if plot_scree:
            plot_singular_values(singularValues_instance[:nbcolumns])
    nb_bins = 100
    bar_color = "#064878"
    plt.figure(figsize=(4, 4))
    # plt.title("Marchenko-Pastur theorem", fontsize=12)
    plt.hist(singularValues**2, density=True,  bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, label=f"Gaussian ($\\beta ="
                                f" {np.round(beta, 1)}$,"
                                f" var = ${np.round(var, 1)}$)")
    plt.plot(marchenko_pastur_generator(var, beta, nbinstances), linewidth=2,
             color=deep[9], label="Marchenko-Pastur pdf")
    plt.tick_params(axis='both', which='major')
    plt.xlabel("Squared singular values $\\sigma^2$")
    plt.ylabel("Spectral density $\\rho(\\sigma^2)$")
    plt.legend(loc=1, fontsize=10)
    plt.ylim([0, 0.5])
    plt.tight_layout()
    plt.show()

    assert 1


def test_marchenko_pastur_generator_uniform_matrix():
    """ For this example, we assume that nbrows >= nbcolumns """
    nbrows = 500
    nbcolumns = 250
    beta = nbrows/nbcolumns
    a = 0
    b = 1
    var = (b - a)**2/12
    nbinstances = 1000
    singularValues = np.array([])
    for i in range(nbinstances):
        W = np.random.uniform(a, b, (nbrows, nbcolumns))

        singularValues_instance = np.zeros(nbrows)
        # ^ Important to have nbrows singular values, since Marchenko-Pastu pdf
        # is for squared singular values (eigenvalues) of Y = W@W.T/nbcolumns
        singularValues_instance[:nbcolumns] = svdvals(W/np.sqrt(nbcolumns))

        singularValues = np.concatenate((singularValues,
                                         singularValues_instance))

    nb_bins = 10000
    bar_color = "#064878"
    plt.figure(figsize=(4, 4))
    # plt.title("Marchenko-Pastur theorem", fontsize=12)
    plt.hist(singularValues**2, density=True,  bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, label=f"Uniform ($\\beta ="
                                f" {np.round(beta, 1)}$,"
                                f" var = ${np.round(var, 1)}$)")
    plt.plot(marchenko_pastur_generator(var, beta, nbinstances), linewidth=2,
             color=deep[9], label="Marchenko-Pastur pdf")
    plt.tick_params(axis='both', which='major')
    plt.xlabel("Squared singular values $\\sigma^2$")
    plt.ylabel("Spectral density $\\rho(\\sigma^2)$")
    plt.legend(loc=1, fontsize=10)
    plt.ylim([0, 3])
    plt.xlim([0, 0.55])
    """ Note : in this case there are singular values away from the bulk """
    plt.tight_layout()
    plt.show()

    assert 1


if __name__ == "__main__":
    pytest.main()
