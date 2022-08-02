# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from scipy.linalg import svdvals
import numpy as np
from singular_values.compute_singular_values_dscm_upper_bounds import *
from plots.config_rcparams import *

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

""" ----------------------- <A_ij> < gamma/(1+gamma) ---------------------- """
for i, k in enumerate([8, 6, 4, 2]):
    N = 1000
    alpha = (np.ones((N, 1)) - np.random.uniform(0, 1, (N, 1)))/np.sqrt(k)
    beta = (np.ones((N, 1)) - np.random.uniform(0, 1, (N, 1)))/np.sqrt(k)
    meanA = alpha @ beta.T / (np.ones((N, N)) + alpha @ beta.T)
    gamma = np.max(alpha @ beta.T) + 1e-8
    # print(alpha, beta)
    # print(alpha@beta.T)
    assert np.all(alpha@beta.T < 1)
    assert np.all(meanA < 1/2)
    assert np.all(alpha@beta.T < gamma)
    assert np.all(meanA < gamma / (1 + gamma))
    print(f"gamma = {gamma}\n\n",
          f"min <A> = {np.min(meanA)}\n\n",
          f"max <A> = {np.max(meanA)}\n\n",
          f"mean <A> = {np.mean(meanA)}\n\n")
    # f"\nbar kin = {np.sum(meanA, axis=1)}",
    # f"\nbar kout = {np.sum(meanA, axis=0)}")

    singularValues = svdvals(meanA)
    normalized_singularValues = singularValues / singularValues[0]
    # """ For the sake of visualization, we set the zeros to ymin
    #         (symlog did not give the desired result) """
    # ymin = 10**(-5)
    # normalized_singularValues[normalized_singularValues < 1e-13] = ymin
    ax1.plot(np.arange(1, len(singularValues) + 1, 1),
             normalized_singularValues, color=deep[i], linewidth=1,
             linestyle="--")
    ax1.scatter(np.arange(1, len(singularValues) + 1, 1),
                normalized_singularValues, s=5,
                color=deep[i], zorder=10)
    # if ylabel_bool:
    #     plt.ylabel("$\\frac{\\sigma_n}{\\sigma_1}$", rotation=0, fontsize=16,
    #                color=first_community_color)
    #     ax.yaxis.labelpad = 20
    
    indices = np.linspace(1, len(singularValues), 10000)
    upper_bound_singvals = upper_bound_singvals_sparser(N, indices, gamma) / \
                           singularValues[0]
    ax1.plot(indices, upper_bound_singvals, color=deep[i],
             label=f"$\\gamma = {np.round(gamma,2)}$")
ax1.set_ylabel("Normalized singular values $\\sigma_n/\\sigma_1$")
ax1.set_xlabel("Dimension $n$")
ax1.legend(loc=1, fontsize=10)
ticks = ax1.get_xticks()
ticks[ticks.tolist().index(0)] = 1
ticks = [i for i in ticks
         if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
plt.xticks(ticks)
ax1.set_xlim([0.95, 10**2])
ax1.set_ylim([10**(-15), 10*np.max(upper_bound_singvals)])
ax1.set_yscale('log', nonposy="clip")
ax1.set_xscale('log')
# plt.legend(loc=1)
ax1.tick_params(axis='y', which='both', left=True,
                right=False, labelbottom=False)
ax1.minorticks_off()
# plt.text(-len(singularValues)/6.2, 0.0000035, "0 -")


""" ----------------------- <A_ij> > omega/(1+omega) ---------------------- """
for i, k in enumerate([10, 5, 2, 1.5]):
    N = 1000
    alpha = np.sqrt(k)*(np.ones((N, 1)) + np.random.uniform(0, 1, (N, 1)))
    beta = np.sqrt(k)*(np.ones((N, 1)) + np.random.uniform(0, 1, (N, 1)))
    meanA = alpha @ beta.T / (np.ones((N, N)) + alpha @ beta.T)
    omega = np.min(alpha @ beta.T) - 0.0000001
    assert np.all(alpha @ beta.T > 1)
    assert np.all(meanA > 1/2)
    assert np.all(meanA < 1)
    assert np.all(alpha@beta.T > omega)
    assert np.all(meanA > omega/(1 + omega))
    print(f"omega = {omega}\n\n",
          f"min <A> = {np.min(meanA)}\n\n",
          f"max <A> = {np.max(meanA)}\n\n",
          f"mean <A> = {np.mean(meanA)}\n\n")
    # f"\nbar kin = {np.sum(meanA, axis=1)}",
    # f"\nbar kout = {np.sum(meanA, axis=0)}")

    singularValues = svdvals(meanA)
    normalized_singularValues = singularValues / singularValues[0]
    # """ For the sake of visualization, we set the zeros to ymin
    #         (symlog did not give the desired result) """
    # ymin = 10**(-5)
    # normalized_singularValues[normalized_singularValues < 1e-13] = ymin
    ax2.plot(np.arange(2, len(singularValues) + 1, 1),
             normalized_singularValues[1:], color=deep[i], linewidth=1,
             linestyle="--")
    ax2.scatter(np.arange(1, len(singularValues) + 1, 1),
                normalized_singularValues, s=5,
                color=deep[i], zorder=10)
    # if ylabel_bool:
    #     plt.ylabel("$\\frac{\\sigma_n}{\\sigma_1}$", rotation=0, fontsize=16,
    #                color=first_community_color)
    #     ax.yaxis.labelpad = 20

    indices = np.linspace(1, len(singularValues), 10000)
    upper_bound_singvals = upper_bound_singvals_denser(N, indices, omega) / \
        singularValues[0]
    ax2.plot(indices, upper_bound_singvals, color=deep[i],
             label=f"$\\omega = {np.round(omega,1)}$")
ax2.set_ylabel("Normalized singular values $\\sigma_n/\\sigma_1$")
ax2.set_xlabel("Dimension $n$")
ax2.legend(loc=1, fontsize=10)
# ticks = ax2.get_xticks()
# ticks[ticks.tolist().index(0)] = 1
# ticks = [i for i in ticks
#          if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
# plt.xticks(ticks)
ax2.set_xlim([0.95, 10**2])
ax2.set_ylim([10**(-15), 10*np.max(upper_bound_singvals)])
ax2.set_yscale('log', nonposy="clip")
ax2.set_xscale('log')
# plt.legend(loc=1)
ax2.tick_params(axis='y', which='both', left=True,
                right=False, labelbottom=False)
ax2.minorticks_off()

plt.show()


# else:
#
#     N = 1000
#     alpha = np.sqrt(k)*(np.ones((N, 1)) + np.random.uniform(0, 1, (N, 1)))
#     beta = np.sqrt(k)*(np.ones((N, 1)) + np.random.uniform(0, 1, (N, 1)))
#     meanA = alpha @ beta.T / (np.ones((N, N)) + alpha @ beta.T)
#     omega = np.min(alpha @ beta.T) - 0.0000001
#     assert np.all(alpha @ beta.T > 1)
#     assert np.all(meanA > 1/2)
#     assert np.all(meanA < 1)
#     assert np.all(alpha@beta.T > omega)
#     assert np.all(meanA > omega/(1 + omega))
#     print(f"omega = {omega}\n\n",
#           f"min <A> = {np.min(meanA)}\n\n",
#           f"max <A> = {np.max(meanA)}\n\n",
#           f"mean <A> = {np.mean(meanA)}\n\n")
#     # f"\nbar kin = {np.sum(meanA, axis=1)}",
#     # f"\nbar kout = {np.sum(meanA, axis=0)}")
#
# else:
#     ax.plot(indices, upper_bound_singvals, color=deep[i],
#             label=f"$\\omega = {np.round(omega,2)}$")
