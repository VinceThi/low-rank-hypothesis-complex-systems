# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from scipy.linalg import svdvals
from singular_values.compute_effective_ranks import *
from plots.config_rcparams import *


setup1 = 1

fig, ax = plt.subplots(1, figsize=(6, 3.8))

# plt.axvline(x=rank, linestyle="--",
#             color=reduced_grey, label="Rank")
# plt.axvline(x=stableRank, linestyle="--",
#             color=deep[0],
#             label="Stable rank")
# plt.axvline(x=elbowPosition, linestyle="--",
#             color=deep[1],
#             label="Elbow position")
# plt.axvline(x=erank, linestyle="--",
#             color=deep[2],
#             label="erank")
# plt.axvline(x=energyRatio, linestyle="--",
#             color=deep[3],
#             label=f"Energy ratio ({percentage_threshold}%)")
# plt.axvline(x=optimalShrinkage, linestyle="--",
#             color=deep[4],
#             label=f"Optimal shrinkage ({norm_str})")
# plt.axvline(x=optimalThreshold, linestyle="--",
#             color=deep[5],
#             label=f"Optimal threshold")
for i, k in enumerate([20, 10, 5, 2]):
    if setup1:

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


        def upper_bound_singvals(i):
            return N*gamma**i/(1 - gamma)

    else:

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

        def upper_bound_singvals(i):
            return N*(i == 1) + N*omega**(1-i)/(omega - 1)

    singularValues = svdvals(meanA)
    normalized_singularValues = singularValues / singularValues[0]
    # """ For the sake of visualization, we set the zeros to ymin
    #         (symlog did not give the desired result) """
    # ymin = 10**(-5)
    # normalized_singularValues[normalized_singularValues < 1e-13] = ymin
    ax.plot(np.arange(1, len(singularValues) + 1, 1),
            normalized_singularValues, color=deep[i], linewidth=1,
            linestyle="--")
    ax.scatter(np.arange(1, len(singularValues) + 1, 1),
               normalized_singularValues, s=5,
               color=deep[i], zorder=10)
    # if ylabel_bool:
    #     plt.ylabel("$\\frac{\\sigma_n}{\\sigma_1}$", rotation=0, fontsize=16,
    #                color=first_community_color)
    #     ax.yaxis.labelpad = 20

    indices = np.linspace(1, len(singularValues), 10000)
    upper_bound_singvals = upper_bound_singvals(indices)/singularValues[0]
    ax.plot(indices, upper_bound_singvals, color=deep[i],
            label=f"$\\gamma = {np.round(gamma,2)}$")
plt.ylabel("Normalized singular values $\\sigma_n/\\sigma_1$")
plt.xlabel("Dimension $n$")
# plt.legend(loc=1, fontsize=8)
ticks = ax.get_xticks()
ticks[ticks.tolist().index(0)] = 1
ticks = [i for i in ticks
         if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
plt.xticks(ticks)
ax.set_xlim([0.95, 10**2])
ax.set_ylim([10**(-15), 10*np.max(upper_bound_singvals)])
ax.set_yscale('log', nonposy="clip")
ax.set_xscale('log')
plt.legend(loc=1)
plt.tick_params(axis='y', which='both', left=True,
                right=False, labelbottom=False)
plt.minorticks_off()
# plt.text(-len(singularValues)/6.2, 0.0000035, "0 -")
plt.show()
