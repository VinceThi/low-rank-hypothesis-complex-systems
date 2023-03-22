# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from scipy.linalg import svdvals
from plots.config_rcparams import *
from graphs.generate_soft_configuration_model import *
from dynamics.error_vector_fields import rmse
from graphs.generate_random_graphs import truncated_pareto
from singular_values.compute_singular_values_dscm_upper_bounds import\
    upper_bound_singvals_infinite_sum_sparser, \
    upper_bound_singvals_infinite_sum_denser,\
    upper_bound_singvals_infinite_sum_weighted,\
    upper_bound_all_singvals
from tqdm import tqdm
from math import log10, floor


def round_sig(x, sig=1):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def plot_singular_values_expected_matrix(ax, indices, a, b,
                                         upper_bound_function, tol, color,
                                         label, letter, ylabel=False,
                                         weighted=True):
    if weighted:
        EW = np.outer(a, b)/(1 - np.outer(a, b))
    else:
        EW = np.outer(a, b)/(1 + np.outer(a, b))
    print(np.min(EW), np.max(EW))
    singular_values = svdvals(EW)
    sig1 = np.max(singular_values)
    upper_bound = \
        upper_bound_all_singvals(indices, a, b, upper_bound_function, tol)
    RMSE = rmse(upper_bound / sig1, singular_values / sig1)

    letter_posx, letter_posy = -0.27, 1.05
    ax.text(letter_posx, letter_posy, letter, fontweight="bold",
            horizontalalignment="center", verticalalignment="top",
            transform=ax.transAxes)
    ax.plot(indices, upper_bound/sig1,
            color=color, linestyle="-", linewidth=1.5,
            label=label + f" ($e \\approx {round_sig(RMSE)}$)")
    ax.plot(indices, singular_values / sig1, linestyle="--",
            linewidth=0.5, color=color)
    ax.scatter(indices, singular_values / sig1, s=8, color=color)
    if ylabel:
        ax.set_ylabel("Rescaled singular\n values $\\sigma_i/\\sigma_1$")
    ax.set_xlabel("Index i")
    # ax.set_xscale("log")
    ax.set_xticks([1, 5, 10])
    ax.set_xlim([0.9, 10.1])
    # ax.set_xticks([1, 20])
    # ax.set_xlim([0, 25])
    ax.set_yscale("log")
    ax.set_yticks([10**(-10), 10**(-8), 10**(-6), 10**(-4), 10**(-2), 1])
    ax.set_ylim([10**(-10), 20])
    plt.tick_params(axis="x", which="both", top=False)
    ax.legend(loc=1, fontsize=8, bbox_to_anchor=(1, 1.06))


def plot_rmse_upper_bound_dscm(ax, x_array, a_array, b_array,
                               upper_bound_function, tol,
                               weighted=True):
    nb_pts = a_array[:, 0]
    rmse_array = np.zeros(nb_pts)
    for i in tqdm(range(len(nb_pts))):
        a, b = a_array[i, :], b_array[i, :]
        if weighted:
            EW = np.outer(a, b)/(1 - np.outer(a, b))
        else:
            EW = g*np.outer(a, b)/(1 + np.outer(a, b))

        singular_values = svdvals(EW)
        sig1 = np.max(singular_values)
        upper_bound = upper_bound_all_singvals(indices, a, b,
                                               upper_bound_function, tol)

        rmse_array[i] = rmse(upper_bound/sig1, singular_values/sig1)

    ax.plot(x_array, rmse_array)


N = 1000
tol = 10**(-12)
indices = np.arange(1, N + 1, 1)

gamma_in, gamma_out = 2, 2.5

fig = plt.figure(figsize=(10, 3.6))

ax1 = plt.subplot(131)
for i, g in enumerate([1, 2, 3]):
    alpha_min, beta_min, alpha_max, beta_max = 2, 1, g*10, g*5
    alpha = truncated_pareto(N, alpha_min, alpha_max, gamma_in) / np.sqrt(N)
    beta = truncated_pareto(N, beta_min, beta_max, gamma_out) / np.sqrt(N)
    maxEA = np.max(np.outer(alpha, beta)/(1 + np.outer(alpha, beta)))
    plot_singular_values_expected_matrix(
        ax1, indices, alpha, beta, upper_bound_singvals_infinite_sum_sparser,
        tol, deep[i],
        f"max$(\\langle A \\rangle) \\approx {round_sig(maxEA)}$",
        "a", ylabel=True, weighted=False)

ax2 = plt.subplot(132)
for i, g in enumerate([3, 1.5, 1]):
    alpha_min, beta_min, alpha_max, beta_max = 50*g, 40*g, 200, 150
    alpha = truncated_pareto(N, alpha_min, alpha_max, gamma_in) / np.sqrt(N)
    beta = truncated_pareto(N, beta_min, beta_max, gamma_out) / np.sqrt(N)
    minEA = np.min(np.outer(alpha, beta)/(1 + np.outer(alpha, beta)))
    plot_singular_values_expected_matrix(
        ax2, indices, alpha, beta, upper_bound_singvals_infinite_sum_denser,
        tol, deep[i],
        f"min$(\\langle A \\rangle) \\approx {round_sig(minEA)}$",
        "b", weighted=False)

ax3 = plt.subplot(133)
for i, g in enumerate([0.3, 0.6, 0.9]):
    ymin, zmin, ymax, zmax = 0.05, 0.05, g, g-0.1
    y = truncated_pareto(N, ymin, ymax, gamma_in)
    z = truncated_pareto(N, zmin, zmax, gamma_out)
    maxEW = np.max(np.outer(y, z)/(1 - np.outer(y, z)))
    plot_singular_values_expected_matrix(
        ax3, indices, y, z, upper_bound_singvals_infinite_sum_weighted,
        tol, deep[i],
        f"max$(\\langle W \\rangle) \\approx {round_sig(maxEW)}$",
        "c", weighted=True)


# ax4 = plt.subplot(234)
#
# ax5 = plt.subplot(235)
#
# ax6 = plt.subplot(236)

plt.show()
