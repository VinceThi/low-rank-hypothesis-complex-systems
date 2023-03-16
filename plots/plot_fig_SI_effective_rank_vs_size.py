# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault

import pandas as pd
import matplotlib as plt
from numpy.linalg import norm
from scipy.optimize import minimize
from plots.config_rcparams import *
import numpy as np
from scipy.stats import pearsonr, binned_statistic


def power_function(x, params):
    # return params[1]*x**(params[0])
    return (x/10)**params


def linear_function(x, params):
    # return params[0]*x + params[1]
    return params[0]*(x - 1)


def objective_function(params, x, y, w, norm_choice):
    return norm(w*(y - linear_function(x, params)), norm_choice)


def plotEffectiveRank_vs_N(ax, letter_str, effRank_str, effrank_name,
                           effectiveRanksDF, ylim, labelpad):
    print(f"\n{effrank_name} vs. N\n--------------")

    """ ----------------- Get and treat the data -------------------------- """
    df = effectiveRanksDF.sort_values(by=['Size'])
    min_size = np.min(df['Size'])
    max_size = 5000   # np.max(df['Size'])
    df = df[df['Size'] <= max_size]
    size = df['Size'].array
    effrank = df[effRank_str].array

    print(f"{len(size)} networks with less than {max_size} vertices")

    """ ----------------- Plot effective ranks vs N ----------------------- """
    s = 2
    letter_posx, letter_posy = 0.1, 1
    # np.ones(len(size))/len(size)
    ax.scatter(size, effrank, s=s)
    ax.text(letter_posx, letter_posy, letter_str, fontweight="bold",
            horizontalalignment="center",
            verticalalignment="top", transform=ax.transAxes)

    """ ----------------- Regression with weights ------------------------- """
    norm_choice = 1

    # ----- Get weights
    weights_choice = "uniform"
    # uniform or inverse-variance
    if weights_choice == "inverse-variance":

        # ----- Choose bins
        nb_bins = 5
        bin_choice = "log-equal-frequency"
        # ^ "log-equal-width" or "log-equal-frequency"
        var_extension_choice = "interp"
        # ^ "equal" or "interp"

        if bin_choice == "log-equal-width":
            bins = np.logspace(np.log10(min_size), np.log10(max_size), nb_bins)

        elif bin_choice == "log-equal-frequency":
            q, bins = pd.qcut(np.log10(df['Size']), q=nb_bins, retbins=True)
            bins = 10 ** bins

        else:
            ValueError("The bin choice is either log-equal-frequency "
                       "or log-equal-width")

        variance, bin_edges, _ = \
            binned_statistic(size, effrank, statistic=np.var, bins=bins)
        count, _, _ = \
            binned_statistic(size, effrank, statistic='count', bins=bins)
        if var_extension_choice == "equal":
            variance_extended = []
            for i, var in enumerate(variance):
                variance_extended += [var] * int(count[i])
        elif var_extension_choice == "interp":
            x = np.linspace(min_size, max_size, len(size))
            variance_extended = np.interp(x, np.arange(len(variance)),
                                          variance)
        else:
            raise ValueError("The bin choice is either log-equal-frequency "
                             "or log-equal-width")
        variance_extended = np.array(variance_extended)
        inverse_variance = 1 / variance_extended
        weights = inverse_variance / np.sum(inverse_variance)
        ax.vlines(bin_edges, ylim[0], ylim[-1] // 20, zorder=-100,
                  color="#ababab",
                  linewidth=0.5, linestyles="--")

    elif weights_choice == "uniform":
        weights = np.ones(len(size))/len(size)

    else:
        raise ValueError("The weights choice is uniform or inverse-variance")

    # ----- Regression
    args = (np.log10(size), np.log10(effrank), weights, norm_choice)
    reg = minimize(objective_function, np.array([1]), args)
    error = \
        objective_function(reg.x, args[0], args[1], args[2], norm_choice)
    print(f"Optimal exponent = {reg.x[0]}")
    print(f"L{norm_choice} regression error = {error}")
    N_array = np.linspace(min_size, max_size, 10000)
    eval_reg = power_function(N_array, reg.x[0])
    ax.plot(N_array, eval_reg, color=dark_grey,
            label="Fit $\\left(\dfrac{N}{10}\\right)"
                  + "^{{{}}}$".format(np.round(reg.x[0], 2)))

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel('N')
    ax.set_xticks([10, 5000])
    ax.set_xlim([-100, 5100])
    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.set_ylim(ylim)
    ax.set_ylabel(effrank_name, labelpad=labelpad)
    ax.set_yticks([1, ylim[1]])
    ax.legend(loc=1)

    """ ----------------- Print correlation measures  --------------------- """
    corr = pearsonr(size, effrank)[0]
    print(f"Pearson correlation coefficient = {corr}")
    # from tests.zold.npeet import mi, entropy
    # mutinf = mi(size, effrank)
    # print(f"Estimated mutual information = {mutinf}")
    # Nlist = [[x] for x in size]
    # effranklist = [[x] for x in effrank]
    # hN = entropy(Nlist)
    # heffrank = entropy(effranklist)
    # nmutinf = mutinf/np.max([hN, heffrank])   # (hN + heffrank - mutinf)
    # print(f"Estimated uncertainty coefficient = {nmutinf}\n")


def plot_effective_ranks_vs_N(effectiveRanksDF):

    plot_d_h = False

    if plot_d_h:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 4))
        plotEffectiveRank_vs_N(axes[0][0], "a", "StableRank", "srank",
                               effectiveRanksDF, [-10, 200], -10)

        plotEffectiveRank_vs_N(axes[0][1], "b", "NuclearRank", "nrank",
                               effectiveRanksDF, [-50, 400], -20)

        plotEffectiveRank_vs_N(axes[0][2], "c", "Elbow", "elbow",
                               effectiveRanksDF, [-50, 400], -20)

        plotEffectiveRank_vs_N(axes[0][3], "d", "EnergyRatio", "energy",
                               effectiveRanksDF, [-50, 1000], -20)

        plotEffectiveRank_vs_N(axes[1][0], "e", "OptimalThreshold", "thrank",
                               effectiveRanksDF, [-100, 2000], -20)

        plotEffectiveRank_vs_N(axes[1][1], "f", "OptimalShrinkage", "shrank",
                               effectiveRanksDF, [-100, 2000], -20)

        plotEffectiveRank_vs_N(axes[1][2], "g", "Erank", "erank",
                               effectiveRanksDF, [-100, 2000], -20)

        plotEffectiveRank_vs_N(axes[1][3], "h", "Rank", "rank",
                               effectiveRanksDF,
                               [-250, 5000], -30)

        plt.subplots_adjust(right=0.5, left=0)

    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

        plotEffectiveRank_vs_N(axes[0], "a", "StableRank", "srank",
                               effectiveRanksDF, [-1, 100], -10)

        plotEffectiveRank_vs_N(axes[1], "b", "NuclearRank", "nrank",
                               effectiveRanksDF, [-5, 400], -10)

        plotEffectiveRank_vs_N(axes[2], "c", "Elbow", "elbow",
                               effectiveRanksDF, [-5, 400], -10)


def main():

    effectiveRanksFilename = "C:/Users/thivi/Documents/GitHub/" \
                             "low-rank-hypothesis-complex-systems/" \
                             "singular_values/properties/effective_ranks.txt"
    header = \
        open(effectiveRanksFilename, 'r').readline().replace('#', ' ').split()
    effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                     comment="#", delimiter=r"\s+")
    effectiveRanksDF.set_index('Name', inplace=True)
    # plotEffectiveRanks_vs_N(effectiveRanksDF)
    # plt.subplots_adjust(wspace=20, left=0, right=1)
    plot_effective_ranks_vs_N(effectiveRanksDF)
    plt.show()


if __name__ == "__main__":
    main()


# nb_iterations = 10
# regression_error = 10**10
# best_regression = None
# for _ in range(nb_iterations):  #  tqdm(range(nb_iterations)):
#     exponent = np.random.uniform(0.05, 1)
#     # offset = np.random.uniform(-2, 2)
#     # r = minimize(objective_function, np.array([exponent,
# offset]), args)
#     r = minimize(objective_function, np.array([exponent]), args)
#     L1_error = \
#         objective_function(r.x, args[0], args[1], args[2], norm_choice)
#
#     if L1_error < regression_error:
#         best_regression = r
#         best_guess_exponent = exponent
#         regression_error = L1_error
#         # print(f"Regression parameters: {r.x}")
#         # print(
#         #     f"Regression error: "
#         #     f"{objective_function(r.x, args[0], args[1],
#         #  args[2], norm_choice)}")
#
# reg = best_regression
# print(f"\nBest guess exponent = {best_guess_exponent}")
# print(f"Regression parameters: {reg.x}")
# print(f"Regression error: "
#       f"{objective_function(reg.x, args[0], args[1], args[2], norm_choice)}")


# def coefficient_of_determination(y, haty, cv_choice="L1"):
#     if cv_choice == "L1":
#         cv = 1 - np.sum(np.abs(y - haty))/np.sum(np.abs(y - np.mean(y)))
#     else:  # cv_choice = "L2"
#         cv = 1 - np.sum((y - haty)**2) / np.sum((y - np.mean(y))**2)
#     return cv
