# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault

import pandas as pd
import matplotlib as plt
from numpy.linalg import norm
from scipy.optimize import minimize
from plots.config_rcparams import *
import numpy as np


# def linear_function(x, params):
#     return params[0]*x + params[1]


def linear_function(x, params):
    return params[0]*x + params[1]


def exponential_function(x, params):
    return params[0]*np.log(x) + params[1]


def objective_function(params, x, y, norm_choice):
    return norm(y - linear_function(x, params), norm_choice)


def objective_function2(params, x, y, norm_choice):
    return norm(y - exponential_function(x, params), norm_choice)


def coefficient_of_variation(y, haty, cv_choice="L1"):
    if cv_choice == "L1":
        cv = 1 - np.sum(np.abs(y - haty))/np.sum(np.abs(y - np.mean(y)))
    else:  # cv_choice = "L2"
        cv = 1 - np.sum((y - haty)**2) / np.sum((y - np.mean(y))**2)
    return cv


def plotEffectiveRanks_vs_N(effectiveRanksDF):
    color = "lightsteelblue"
    letter_posx, letter_posy = 0.1, 1    # -0.25, 1.08
    norm_choice = 1
    ylim2 = [-1000, 10500]
    s = 2

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 4))
    min_size = np.min(effectiveRanksDF['Size'])
    max_size = np.max(effectiveRanksDF['Size'])
    N_array = np.linspace(min_size, max_size, 1000)

    axes[0][0].scatter(effectiveRanksDF['Size'],
                       effectiveRanksDF['StableRank'], s=s)
    axes[0][0].text(letter_posx, letter_posy, "a", fontweight="bold",
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[0, 0].transAxes)
    axes[0][0].set_ylim([-20, 300])  # An outlier is not shown for viz
    args = (effectiveRanksDF['Size'], effectiveRanksDF['StableRank'],
            norm_choice)
    # reg = minimize(objective_function2, np.array([1, 1]), args)
    # line = exponential_function(N_array, reg.x)  #
    # reg = minimize(objective_function, np.array([0.05, 10]), args)
    # line = linear_function(N_array, reg.x)
    reg = minimize(objective_function2, np.array([1, 1]), args)
    line = exponential_function(N_array, reg.x)
    axes[0][0].plot(N_array, line, color=dark_grey)
    # axes[0][0].text(N_array[-1], line[-1] + 0.1, f"{np.round(reg.x[0], 3)}",
    #                 fontsize=10)
    print(reg.x)
    # print("relative L1 error / nb_sample = ", objective_function2(reg.x, effectiveRanksDF['Size'],
    #                                                       effectiveRanksDF['StableRank'], norm_choice)/len(effectiveRanksDF['Size'])/np.max(effectiveRanksDF['StableRank']))
    print(objective_function2(reg.x, effectiveRanksDF['Size'], effectiveRanksDF['StableRank'], norm_choice))
    print(coefficient_of_variation(effectiveRanksDF['StableRank'],
                                   linear_function(effectiveRanksDF['Size'],
                                                   reg.x)))
    axes[0][0].set_xscale("log")
    
    # args = (np.log(effectiveRanksDF['Size']),
    #         np.log(effectiveRanksDF['StableRank']), norm_choice)
    # reg = minimize(objective_function, np.array([0.05, 10]), args)
    # line = linear_function(np.log(effectiveRanksDF['Size']), reg.x)
    # axes[0][0].plot(effectiveRanksDF['Size'], np.exp(line), color=dark_grey)
    # axes[0][0].text(np.max(effectiveRanksDF['Size']),
    #  np.exp(np.max(line[-1]))+20, f"{np.round(reg.x[0], 3)}",
    #                 fontsize=10)
    # print(coefficient_of_variation(np.log(effectiveRanksDF['StableRank']),
    #                                linear_function(np.log(effectiveRanksDF['Size']),
    #                                                reg.x)))
    """ 
    
    Validation of the L2 regression with scipy:
    
    from scipy.stats import linregress  
    
    reg = linregress(effectiveRanksDF['Size'], effectiveRanksDF['StableRank'])           
    line = reg.intercept + reg.slope * N_array
    
    is equivalent to 
    
    args = (effectiveRanksDF['Size'], effectiveRanksDF['StableRank'], 2) 
    reg = minimize(objective_function, np.array([0.05, 10]), args)        
    line = linear_function(N_array, reg.x)                               
    """

    axes[0][1].scatter(effectiveRanksDF['Size'],
                       effectiveRanksDF['NuclearRank'], s=s)
    axes[0][1].text(letter_posx, letter_posy, "b", fontweight="bold",
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[0, 1].transAxes)
    axes[0][1].set_ylim([-90, 1100])  # An outlier is removed for viz
    args = (effectiveRanksDF['Size'], effectiveRanksDF['NuclearRank'],
            norm_choice)
    reg = minimize(objective_function, np.array([0.05, 10]), args)
    line = linear_function(N_array, reg.x)
    axes[0][1].plot(N_array, line, color=dark_grey)
    axes[0][1].text(N_array[-1], line[-1] + 0.1, f"{np.round(reg.x[0], 3)}",
                    fontsize=10)
    print(coefficient_of_variation(effectiveRanksDF['NuclearRank'],
                                   linear_function(effectiveRanksDF['Size'],
                                                   reg.x)))
    axes[0][1].set_xscale("log")



    axes[0][2].scatter(effectiveRanksDF['Size'],
                       effectiveRanksDF['Elbow'], s=s)
    axes[0][2].text(letter_posx, letter_posy, "c", fontweight="bold",
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[0, 2].transAxes)
    axes[0][2].set_ylim([-200, 3200])
    args = (effectiveRanksDF['Size'], effectiveRanksDF['Elbow'],
            norm_choice)
    reg = minimize(objective_function, np.array([0.05, 10]), args)
    line = linear_function(N_array, reg.x)
    axes[0][2].plot(N_array, line, color=dark_grey)
    axes[0][2].text(N_array[-1], line[-1] + 0.1, f"{np.round(reg.x[0], 3)}",
                    fontsize=10)
    print(coefficient_of_variation(effectiveRanksDF['Elbow'],
                                   linear_function(effectiveRanksDF['Size'],
                                                   reg.x)))
    axes[0][2].set_xscale("log")


    axes[0][3].scatter(effectiveRanksDF['Size'],
                       effectiveRanksDF['EnergyRatio'], s=s)
    axes[0][3].text(letter_posx, letter_posy, "d", fontweight="bold",
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[0, 3].transAxes)
    axes[0][3].set_ylim([-500, 7000])
    args = (effectiveRanksDF['Size'], effectiveRanksDF['EnergyRatio'],
            norm_choice)
    reg = minimize(objective_function, np.array([0.05, 10]), args)
    line = linear_function(N_array, reg.x)
    axes[0][3].plot(N_array, line, color=dark_grey)
    axes[0][3].text(N_array[-1]-1000, line[-1] + 100, f"{np.round(reg.x[0], 3)}",
                    fontsize=10)
    print(coefficient_of_variation(effectiveRanksDF['EnergyRatio'],
                                   linear_function(effectiveRanksDF['Size'],
                                                   reg.x)))
    axes[0][3].set_xscale("log")

    axes[1][0].scatter(effectiveRanksDF['Size'],
                       effectiveRanksDF['OptimalThreshold'], s=s)
    axes[1][0].text(letter_posx, letter_posy, "e", fontweight="bold",
                    horizontalalignment="center", verticalalignment="top",
                    transform=axes[1, 0].transAxes)
    axes[1][0].set_ylim(ylim2)
    args = (effectiveRanksDF['Size'], effectiveRanksDF['OptimalThreshold'],
            norm_choice)
    reg = minimize(objective_function, np.array([0.05, 10]), args)
    line = linear_function(N_array, reg.x)
    axes[1][0].plot(N_array, line, color=dark_grey)
    axes[1][0].text(N_array[-1], line[-1] + 0.1, f"{np.round(reg.x[0], 3)}",
                    fontsize=10)
    print(coefficient_of_variation(effectiveRanksDF['OptimalThreshold'],
                                   linear_function(effectiveRanksDF['Size'],
                                                   reg.x)))
    axes[1][0].set_xscale("log")


    axes[1][1].scatter(effectiveRanksDF['Size'],
                       effectiveRanksDF['OptimalShrinkage'], s=s)
    axes[1][1].text(letter_posx, letter_posy, "f", fontweight="bold",
                    horizontalalignment="center", verticalalignment="top",
                    transform=axes[1, 1].transAxes)
    axes[1][1].set_ylim(ylim2)
    args = (effectiveRanksDF['Size'],
            effectiveRanksDF['OptimalShrinkage'],
            norm_choice)
    reg = minimize(objective_function, np.array([0.05, 10]), args)
    line = linear_function(N_array, reg.x)
    axes[1][1].plot(N_array, line, color=dark_grey)
    axes[1][1].text(N_array[-1], line[-1] + 0.1, f"{np.round(reg.x[0], 3)}",
                    fontsize=10)
    print(coefficient_of_variation(effectiveRanksDF['OptimalShrinkage'],
                                   linear_function(effectiveRanksDF['Size'],
                                                   reg.x)))
    axes[1][1].set_xscale("log")

    axes[1][2].scatter(effectiveRanksDF['Size'],
                       effectiveRanksDF['Erank'], s=s)
    axes[1][2].text(letter_posx, letter_posy, "g", fontweight="bold",
                    horizontalalignment="center", verticalalignment="top",
                    transform=axes[1, 2].transAxes)
    axes[1][2].set_ylim(ylim2)
    args = (effectiveRanksDF['Size'], effectiveRanksDF['Erank'],
            norm_choice)
    reg = minimize(objective_function, np.array([0.05, 10]), args)
    line = linear_function(N_array, reg.x)
    axes[1][2].plot(N_array, line, color=dark_grey)
    axes[1][2].text(N_array[-1], line[-1] + 0.1, f"{np.round(reg.x[0], 3)}",
                    fontsize=10)
    print(coefficient_of_variation(effectiveRanksDF['Erank'],
                                   linear_function(effectiveRanksDF['Size'],
                                                   reg.x)))
    axes[1][2].set_xscale("log")

    axes[1][3].scatter(effectiveRanksDF['Size'],
                       effectiveRanksDF['Rank'], s=s)
    axes[1][3].plot(effectiveRanksDF['Size'], effectiveRanksDF['Size'],
                    linestyle="--", linewidth=1.5, color=reduced_grey)
    axes[1][3].text(letter_posx, letter_posy, "h", fontweight="bold",  # Rank",
                    horizontalalignment="center",
                    verticalalignment="top", transform=axes[1, 3].transAxes)
    axes[1][3].set_ylim([-2000, np.max(effectiveRanksDF['Size'])+3000])
    args = (effectiveRanksDF['Size'], effectiveRanksDF['Rank'],
            norm_choice)
    reg = minimize(objective_function, np.array([0.05, 10]), args)
    line = linear_function(N_array, reg.x)
    axes[1][3].plot(N_array, line, color=dark_grey)
    axes[1][3].text(N_array[-1]-1000, line[-1] + 500, f"{np.round(reg.x[0], 3)}",
                    fontsize=10)
    # print(reg, objective_function(reg.x, args[0], args[1], args[2]))
    print(coefficient_of_variation(effectiveRanksDF['Rank'],
                                   linear_function(effectiveRanksDF['Size'],
                                                   reg.x)))
    axes[1][3].set_xscale("log")

    # args = (np.log(effectiveRanksDF['Size']),
    #         np.log(effectiveRanksDF['Rank']), norm_choice)
    # reg = minimize(objective_function, np.array([0.05, 10]), args)
    # line = linear_function(np.log(effectiveRanksDF['Size']), reg.x)
    # axes[2][1].plot(effectiveRanksDF['Size'], np.exp(line), color=dark_grey)
    # axes[2][1].text(np.max(effectiveRanksDF['Size']),
    #                 np.exp(np.max(line[-1])) + 20,
    # f"{np.round(reg.x[0], 3)}",
    #                 fontsize=10)
    # print(coefficient_of_variation(np.log(effectiveRanksDF['Rank']),
    #                                linear_function(
    #                                    np.log(effectiveRanksDF['Size']),
    #                                    reg.x)))

    # nbVertices = effectiveRanksDF['Size']
    # nbVertices = nbVertices.values
    # x, bins, p = axes[2][2].hist(nbVertices, bins=np.logspace(np.log10(0.1),
    #                                                           np.log10(21000),
    #                                                           40),
    #                              density=True, color=color)
    # for item in p:
    #     item.set_height(item.get_height() / sum(x))
    # plt.text(letter_posx, letter_posy, "i", fontweight="bold",
    #          horizontalalignment="center",
    #          verticalalignment="top", transform=axes[2, 2].transAxes)
    # axes[2][2].set_xscale('log')
    # axes[2][2].set_ylim([-0.02 * 0.20, 0.20])
    # axes[2][2].set_xlim([0.5, 9 * 10 ** 4])
    # plt.xticks([1, 100, 10000])

    axes[0, 0].set_xlabel('N')
    axes[0, 1].set_xlabel('N')
    axes[0, 2].set_xlabel('N')
    axes[0, 3].set_xlabel('N')
    axes[1, 0].set_xlabel('N')
    axes[1, 1].set_xlabel('N')
    axes[1, 2].set_xlabel('N')
    axes[1, 3].set_xlabel('N')
    # axes[2, 2].set_xlabel('N')
    axes[0, 0].set_ylabel('srank', labelpad=-10)
    axes[0, 1].set_ylabel('nrank', labelpad=-20)
    axes[0, 2].set_ylabel('elbow', labelpad=-20)
    axes[0, 3].set_ylabel('energy', labelpad=-20)
    axes[1, 0].set_ylabel('thrank', labelpad=-30)
    axes[1, 1].set_ylabel('shrank', labelpad=-30)
    axes[1, 2].set_ylabel('erank', labelpad=-30)
    axes[1, 3].set_ylabel('rank', labelpad=-30)
    # axes[2, 2].set_ylabel('Fraction\nof networks', labelpad=-20)

    axes[0, 0].xaxis.set_label_coords(0.4, -0.08)
    axes[0, 1].xaxis.set_label_coords(0.4, -0.08)
    axes[0, 2].xaxis.set_label_coords(0.4, -0.08)
    axes[0, 3].xaxis.set_label_coords(0.4, -0.08)
    axes[1, 0].xaxis.set_label_coords(0.4, -0.08)
    axes[1, 1].xaxis.set_label_coords(0.4, -0.08)
    axes[1, 2].xaxis.set_label_coords(0.4, -0.08)
    axes[1, 3].xaxis.set_label_coords(0.4, -0.08)
    # axes[2, 2].xaxis.set_label_coords(1.05, -0.025)

    # axes[0, 0].set_yscale("log")
    # axes[0, 1].set_yscale("log")
    # axes[0, 2].set_yscale("log")
    # axes[1, 0].set_yscale("log")
    # axes[1, 1].set_yscale("log")
    # axes[1, 2].set_yscale("log")
    # axes[2, 0].set_yscale("log")
    # axes[2, 1].set_yscale("log")
    #
    # axes[0, 0].set_xscale("log")
    # axes[0, 1].set_xscale("log")
    # axes[0, 2].set_xscale("log")
    # axes[1, 0].set_xscale("log")
    # axes[1, 1].set_xscale("log")
    # axes[1, 2].set_xscale("log")
    # axes[2, 0].set_xscale("log")
    # axes[2, 1].set_xscale("log")
    #
    # axes[0, 0].tick_params(which="both", top=False, right=False)
    # axes[0, 1].tick_params(which="both", top=False, right=False)
    # axes[0, 2].tick_params(which="both", top=False, right=False)
    # axes[1, 0].tick_params(which="both", top=False, right=False)
    # axes[1, 1].tick_params(which="both", top=False, right=False)
    # axes[1, 2].tick_params(which="both", top=False, right=False)
    # axes[2, 0].tick_params(which="both", top=False, right=False)
    # axes[2, 1].tick_params(which="both", top=False, right=False)
    #
    # axes[0, 0].set_ylim([0.8, 25000])
    # axes[0, 1].set_ylim([0.8, 25000])
    # axes[0, 2].set_ylim([0.8, 25000])
    # axes[1, 0].set_ylim([0.8, 25000])
    # axes[1, 1].set_ylim([0.8, 25000])
    # axes[1, 2].set_ylim([0.8, 25000])
    # axes[2, 0].set_ylim([0.8, 25000])
    # axes[2, 1].set_ylim([0.8, 25000])
    #
    axes[0, 0].set_yticks([1, 300])
    axes[0, 1].set_yticks([1, 1000])
    axes[0, 2].set_yticks([1, 3000])
    axes[0, 3].set_yticks([1, 6000])
    axes[1, 0].set_yticks([1, 10000])
    axes[1, 1].set_yticks([1, 10000])
    axes[1, 2].set_yticks([1, 10000])
    axes[1, 3].set_yticks([1, 20000])
    # axes[2, 2].set_yticks([0, 0.2])

    xticks = [10, 20000]
    axes[0, 0].set_xticks(xticks)
    axes[0, 1].set_xticks(xticks)
    axes[0, 2].set_xticks(xticks)
    axes[0, 3].set_xticks(xticks)
    axes[1, 0].set_xticks(xticks)
    axes[1, 1].set_xticks(xticks)
    axes[1, 2].set_xticks(xticks)
    axes[1, 3].set_xticks(xticks)

    plt.subplots_adjust(right=0.5, left=0)

    return fig


def main():

    effectiveRanksFilename = "C:/Users/thivi/Documents/GitHub/" \
                             "low-rank-hypothesis-complex-systems/" \
                             "singular_values/properties/effective_ranks.txt"
    header = \
        open(effectiveRanksFilename, 'r').readline().replace('#', ' ').split()
    effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                     comment="#", delimiter=r"\s+")
    effectiveRanksDF.set_index('Name', inplace=True)
    plotEffectiveRanks_vs_N(effectiveRanksDF)
    # plt.subplots_adjust(wspace=20, left=0, right=1)
    plt.show()


if __name__ == "__main__":
    main()
