# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault

import pandas as pd
import matplotlib as plt
from numpy.linalg import norm
from scipy.optimize import minimize
from plots.config_rcparams import *
import numpy as np
from scipy.stats import pearsonr, binned_statistic
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def power_function(x, params):
    return params[0]*x**params[1] + params[2]


def objective_function(params, x, y, w, norm_choice):
    return norm(w * (y - power_function(x, params)), norm_choice)


def inset_setup(ax):
    axins = inset_axes(ax, width="45%", height="35%",
                       bbox_to_anchor=(-0.25, 0.55, 1, 1),
                       bbox_transform=ax.transAxes, loc=4)
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        axins.spines[axis].set_linewidth(0.5)
    axins.tick_params(axis='both', which='major', labelsize=8,
                      width=0.5, length=2)
    return axins


def plotEffectiveRank_vs_N(ax, letter_str, effRank_str, effrank_name,
                           effectiveRanksDF, ylim, labelpad):
    print(f"\n{effrank_name} vs. N\n--------------")

    """ ----------------- Get and treat the data -------------------------- """
    df = effectiveRanksDF.sort_values(by=['Size'])
    min_size = np.min(df['Size'])
    max_size = np.max(df['Size'])
    df = df[df['Size'] <= max_size]
    size = df['Size'].array
    effrank = df[effRank_str].array

    print(f"{len(size)} networks with less than {max_size} vertices")

    """ ----------------- Plot effective ranks vs N ----------------------- """
    s = 2
    letter_posx, letter_posy = 0.05, 1.1
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
    bound = [(0, 10), (0, 1), (-100, 30)]
    # args = (np.log10(size), np.log10(effrank), weights, norm_choice)
    args = (size, effrank, weights, norm_choice)
    reg = minimize(objective_function, np.array([1, 0.5, -1]),
                   args, bounds=bound)
    error = \
        objective_function(reg.x, args[0], args[1], args[2], norm_choice)
    print(f"Optimal exponent = {reg.x[1]}")
    print(f"Params = {reg.x}")
    print(f"Normalized L{norm_choice} regression error ="
          f" {error/(np.mean(effrank))}")
    N_array = np.linspace(min_size, max_size, 10000)
    eval_reg = power_function(N_array, reg.x)  # [0])
    ax.plot(N_array, eval_reg, color=dark_grey)
    # label="Fit $\\propto N"
    #       + "^{{{}}}$".format(np.round(reg.x[1], 2)))
    ax.text(N_array[-1]-2000, eval_reg[-1] - 5,
            "$\\propto N" + "^{{{}}}$".format("%.2f" % np.round(reg.x[1], 2)),
            fontsize=9, clip_on=False)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel('$N$')
    ax.set_xticks([10, 20000])
    ax.set_xlim([-500, 20000])
    ax.xaxis.set_label_coords(0.5, -0.06)
    ax.set_ylim(ylim)
    ax.set_ylabel(effrank_name, labelpad=labelpad)
    ax.set_yticks([1, ylim[1]])
    # ax.legend(loc=1)

    axins = inset_setup(ax)
    axins.scatter(size, effrank, s=s)
    axins.plot(N_array, eval_reg, color=dark_grey)
    ymax = ylim[1]/8
    ylim_axins = [-ymax/20, ymax]
    axins.set_ylim(ylim_axins)
    axins.set_yticks([1, np.round(ymax)])
    axins.set_xlim([-200, 3050])
    axins.set_xticks([10, 3000])

    """ ----------------- Print correlation measures  --------------------- """
    corr = pearsonr(size, effrank)[0]
    print(f"Pearson correlation coefficient = {corr}")


def plot_effective_ranks_vs_N(effectiveRanksDF):

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(11, 5))
    plotEffectiveRank_vs_N(axes[0][0], "a", "StableRank", "srank",
                           effectiveRanksDF, [-10, 200], -10)

    plotEffectiveRank_vs_N(axes[0][1], "b", "NuclearRank", "nrank",
                           effectiveRanksDF, [-50, 1000], -20)

    plotEffectiveRank_vs_N(axes[0][2], "c", "Elbow", "elbow",
                           effectiveRanksDF, [-50, 1000], -20)

    plotEffectiveRank_vs_N(axes[0][3], "d", "EnergyRatio", "energy",
                           effectiveRanksDF, [-250, 5000], -20)

    plotEffectiveRank_vs_N(axes[1][0], "e", "OptimalThreshold", "thrank",
                           effectiveRanksDF, [-250, 5000], -20)

    plotEffectiveRank_vs_N(axes[1][1], "f", "OptimalShrinkage", "shrank",
                           effectiveRanksDF, [-250, 5000], -20)

    plotEffectiveRank_vs_N(axes[1][2], "g", "Erank", "erank",
                           effectiveRanksDF, [-500, 10000], -20)

    plotEffectiveRank_vs_N(axes[1][3], "h", "Rank", "rank",
                           effectiveRanksDF,
                           [-1000, 20000], -30)

    plt.subplots_adjust(right=0.5, left=0)


def main():

    effectiveRanksFilename = "C:/Users/thivi/Documents/GitHub/" \
                             "low-rank-hypothesis-complex-systems/" \
                             "singular_values/properties/effective_ranks.txt"
    header = \
        open(effectiveRanksFilename, 'r').readline().replace('#', ' ').split()
    effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                     comment="#", delimiter=r"\s+")
    effectiveRanksDF.set_index('Name', inplace=True)
    plot_effective_ranks_vs_N(effectiveRanksDF)
    plt.show()


if __name__ == "__main__":
    main()
