# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from scipy.linalg import svdvals
# from scipy.stats import powerlaw
from singular_values.compute_effective_ranks import *
from plots.config_rcparams import *


setup1 = 1

if setup1:

    N = 100
    # alpha = (np.ones((N, 1)) - 0.05*powerlaw.rvs(2.1, size=(N,1)))/np.sqrt(N)
    # beta = (np.ones((N, 1)) - 0.1*  powerlaw.rvs(3.2, size=(N,1)))/np.sqrt(N)
    # alpha = (np.ones((N, 1)) - 0.05*np.random.random((N, 1)))/np.sqrt(N)
    # beta = (np.ones((N, 1)) - 0.06*np.random.random((N, 1)))/np.sqrt(N)
    # alpha = (np.ones((N, 1)) - 0.5*np.random.normal(0.7, 0.3, (N, 1)))\
    #     / np.sqrt(N) # *np.random.choice([0, 1], size=(N, 1), p=[1/10, 9/10])
    # beta = (np.ones((N, 1)) - 0.5*np.random.normal(0.7, 0.3, (N, 1)))\
    #     / np.sqrt(N) # *np.random.choice([0, 1], size=(N, 1), p=[1/10, 9/10])

    # alpha = (np.ones((N, 1)) - np.random.uniform(0, 0.1, (N, 1)))/np.sqrt(N)  # Case where we are not very good (less heterogeneous)
    # beta = (np.ones((N, 1)) - np.random.uniform(0, 0.1, (N, 1)))/np.sqrt(N)   # Case where we are not very good (less heterogeneous)

    alpha = (np.ones((N, 1)) - np.random.uniform(0, 1, (N, 1)))/np.sqrt(N)
    beta = (np.ones((N, 1)) - np.random.uniform(0, 1, (N, 1)))/np.sqrt(N)
    print(alpha*beta)
    assert np.all(alpha*beta < 1/N)

    sigma = np.linalg.norm(alpha)*np.linalg.norm(beta)

    print(sigma)

    def upper_bound_singvals(x, N):
        return sigma**x/(1 - sigma)
else:

    N = 100
    # alpha = np.sqrt(N)*(np.ones((N, 1)) + 5*powerlaw.rvs(2.1, size=(N, 1)))
    # beta = np.sqrt(N)*(np.ones((N, 1)) + 5*powerlaw.rvs(2.2, size=(N, 1)))
    alpha = np.sqrt(N)*(np.ones((N, 1)) + np.random.normal(0.8, 0.4, (N, 1)))
    beta = np.sqrt(N)*(np.ones((N, 1)) + np.random.normal(0.8, 0.4, (N, 1)))
    print(alpha*beta)
    assert np.all(alpha*beta >= N)

    s = np.linalg.norm(alpha**(-1))*np.linalg.norm(beta**(-1))

    print(s)

    upper_bound_singvals = np.concatenate((np.array([N + s/(1 - s)]),
                                           s**(np.arange(1, N, 1))/(1 - s)))

meanA = alpha @ beta.T / (np.ones((N, N)) + alpha @ beta.T)
if setup1:
    assert np.all(meanA < 1/(N+1))
print(f"<A> = {meanA}", f"\nbar kin = {np.sum(meanA, axis=1)}",
      f"\nbar kout = {np.sum(meanA, axis=0)}")
singularValues = svdvals(meanA)

plt.matshow(meanA)
plt.show()

numberSingularValues = len(singularValues)
rank = computeRank(singularValues)
stableRank = computeStableRank(singularValues)
optimalThreshold = computeOptimalThreshold(singularValues)
norm_str = 'frobenius'
optimalShrinkage = computeOptimalShrinkage(singularValues)
elbowPosition = findElbowPosition(singularValues)
erank = computeERank(singularValues)
threshold = 0.9
percentage_threshold = "%.0f" % (threshold*100)
energyRatio = computeEffectiveRankEnergyRatio(singularValues,
                                              threshold=threshold)
header = ['Size', 'Rank', 'Optimal threshold',
          'Optimal shrinkage',
          'Erank', 'Elbow', 'Energy ratio', 'Stable rank']
properties = [[numberSingularValues,
               rank, optimalThreshold, optimalShrinkage,
               erank, elbowPosition, energyRatio, stableRank]]
print("\n\n\n\n", tabulate.tabulate(properties, headers=header))

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
ax.scatter(np.arange(1, len(singularValues) + 1, 1),
           singularValues, s=10)

ax.plot(np.linspace(1, len(singularValues), 1000),
        upper_bound_singvals(np.linspace(1, len(singularValues), 1000), N), color=dark_grey)

plt.ylabel("Singular\n values $\\sigma_i$")
plt.xlabel("Index $i$")
plt.legend(loc=1, fontsize=8)
ticks = ax.get_xticks()
ticks[ticks.tolist().index(0)] = 1
ticks = [i for i in ticks
         if -0.1*len(singularValues) < i < 1.1*len(singularValues)]
plt.xticks(ticks)
plt.show()
