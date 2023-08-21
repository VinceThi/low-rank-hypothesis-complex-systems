# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from numpy.linalg import norm
from scipy.linalg import svdvals
from plots.config_rcparams import *
import time
import json
import tkinter.simpledialog
from tkinter import messagebox
from graphs.generate_s1_random_graph import *
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/singular_values/properties/" \
           "singular_values_random_graphs/"

""" Random graph parameters """
graph_str = "perturbed_gaussian"
directed = True
expected = True
N = 1000
nb_networks = 100
rank = 5
prng1 = np.random.RandomState(1234567890)
prng2 = np.random.RandomState(1334567890)
prng3 = np.random.RandomState(1434567890)
prng4 = np.random.RandomState(1534567890)
prng5 = np.random.RandomState(1634567890)
m1 = prng1.normal(0, 1/np.sqrt(N), (N, ))
n1 = prng1.normal(0, 0.1, (N, ))
m2 = prng2.normal(0, 1/np.sqrt(N), (N, ))
n2 = prng2.normal(0.1, 0.2, (N, ))
m3 = prng3.normal(0, 1/np.sqrt(N), (N, ))
n3 = prng3.normal(0.2, 0.3, (N, ))
m4 = prng4.normal(0, 1/np.sqrt(N), (N, ))
n4 = prng4.normal(0.3, 0.4, (N, ))
m5 = prng5.normal(0, 1/np.sqrt(N), (N, ))
n5 = prng5.normal(0.4, 0.5, (N, ))
P = np.array([m1, m2, m3, m4, m5]).T
Q = np.array([n1, n2, n3, n4, n5]).T
EW = P@Q.T  
g = 3  # 1 and 3 (norm ratio approx 1 and approx 0.25)
var = g**2/N


norm_choice = 2
norm_R = np.zeros(nb_networks)


""" Get singular values """
singularValues = np.zeros((nb_networks, N))
singularValues_R = np.zeros((nb_networks, N))
for i in tqdm(range(0, nb_networks)):
    R = np.random.normal(0, np.sqrt(var), (N, N))
    W = EW + R
    norm_R[i] = norm(R, ord=norm_choice)
    singularValues[i, :] = svdvals(W)
    singularValues_R[i, :] = svdvals(R)

norm_EW = norm(EW, ord=norm_choice)
singularValues_EW = svdvals(EW)
norm_ratio = np.mean(norm_R)/norm_EW
print(norm_ratio)


""" Plot singular values and Weyl's theorem"""

xlabel = "Index $i$"
# ylabel = "Average rescaled singular\n values $\\sigma_i/\\sigma_1$"
# ylabel = "Average singular values"

mean_norm_W = np.mean(singularValues[:, 0])
mean_singularValues = np.mean(singularValues, axis=0)
bar_singularValues = np.std(singularValues, axis=0)

mean_singularValues_R = np.mean(singularValues_R, axis=0)
bar_singularValues_R = np.std(singularValues_R,  axis=0)

fig = plt.figure(figsize=(4, 4))
ax1 = plt.subplot(111)
indices = np.arange(1, N + 1, 1)
# ax1.scatter(indices, mean_singularValues/mean_norm_W, s=30, color=deep[0],
#             label="$\\langle\\sigma_i(W)\\rangle\,/"
#                   "\,\\langle\,||W||_2\,\\rangle$")
# ax1.fill_between(indices,
#                  (mean_singularValues - std_singularValues)/mean_norm_W,
#                  (mean_singularValues + std_singularValues)/mean_norm_W,
#                  color=deep[0], alpha=0.2)
ax1.errorbar(x=indices, y=mean_singularValues/mean_norm_W,
             yerr=bar_singularValues/mean_norm_W, fmt="o", color=deep[0],
             zorder=-30, markersize=5, capsize=1, elinewidth=1,
             label="$\\langle\\sigma_i(W)\\rangle\,/"
                   "\,\\langle\,||W||_2\,\\rangle$")
# ax1.scatter(indices, mean_singularValues_R/mean_norm_W, s=2, color=deep[9],
#             label="$\\langle\\sigma_i(R)\\rangle\,/"
#                   "\,\\langle\,||W||_2\,\\rangle$", zorder=2)
# ax1.fill_between(indices,
#                  (mean_singularValues_R - bar_singularValues_R)/mean_norm_W,
#                  (mean_singularValues_R + bar_singularValues_R)/mean_norm_W,
#                  color=deep[2], alpha=0.2)
ax1.scatter(indices, singularValues_EW/mean_norm_W,
            label="$\\sigma_i(\\langle W \\rangle)\,/"
                  "\,\\langle\,||W||_2\,\\rangle$",
            s=5, color=deep[1], zorder=0)

ax1.errorbar(x=indices, y=mean_singularValues_R/mean_norm_W,
             yerr=bar_singularValues_R/mean_norm_W, fmt="o", color=deep[9],
             zorder=30, markersize=1.5, capsize=1, elinewidth=1,
             label="$\\langle\\sigma_i(R)\\rangle\,/"
                   "\,\\langle\,||W||_2\,\\rangle$")
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
           loc=4, bbox_to_anchor=(0, 0.1, 1, 1))
ticks = ax1.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])
plt.tick_params(axis='both', which='major')
# plt.ylim([0.01*np.min(singularValues), 1.5])
# ax1.set_yscale('log')
# plt.tick_params(axis='y', which='both', left=True,
#                 right=False, labelbottom=False)
plt.xlabel(xlabel)
plt.xlim([-50, N+50])

axins = inset_axes(ax1, width="50%", height="50%",
                   bbox_to_anchor=(-0.1, 0, 1, 1),
                   bbox_transform=ax1.transAxes, loc=1)
axins.plot(indices, np.mean(norm_R)*np.ones(N)/mean_norm_W,
           linewidth=1, linestyle="--", color=deep[9])
axins.text(0.92, 0.97, "$\\frac{\\langle\,||R||_2\,\\rangle}"
                       "{\\langle\,||W||_2\,\\rangle}$", fontweight="bold",
           horizontalalignment="center", verticalalignment="top",
           transform=ax1.transAxes, fontsize=8)
axins.scatter(indices,
              np.mean(np.abs(singularValues-np.outer(np.ones(nb_networks),
                                                     singularValues_EW)),
                      axis=0)/mean_norm_W,
              color=deep[7], s=2)
# axins.errorbar(x=indices, y=np.mean(np.abs(singularValues-singularValues_EW),
#                                     axis=0)/mean_norm_W,
#                yerr=np.std(np.abs(singularValues-singularValues_EW),
#                            axis=0)/mean_norm_W, fmt="o", color=deep[7],
#                markersize=1, capsize=0, elinewidth=1)
axins.set_ylabel("$\\langle\,\,|\,\\sigma_i(W)"
                 " - \\sigma_i(\\langle W\\rangle)\,|\,\,\\rangle"
                 "\,\,/\,\,\\langle\,||W||_2\,\\rangle$",
                 fontsize=8)
axins.set_xlabel("Index $i$", fontsize=8)
ticks = axins.get_xticks()
ticks[ticks.tolist().index(0)] = 1
axins.set_xticks(ticks[ticks > 0])
axins.set_xlim([-50, N+50])
for axis in ['top', 'bottom', 'left', 'right']:
    axins.spines[axis].set_linewidth(0.5)
axins.tick_params(axis='both', which='major', labelsize=8,
                  width=0.5, length=2)
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters,"
                       " the data, and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    path = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "singular_values/properties/singular_values_random_graphs/"
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    parameters_dictionary = {"graph_str": graph_str,
                             "directed": directed,
                             "N": N, "P": P.tolist(), "Q": Q.tolist(),
                             "norm_EW": norm_EW,
                             "norm_R": norm_R.tolist(),
                             "norm_ratio": norm_ratio.tolist(),
                             "nb_samples (nb_networks)": nb_networks,
                             "norm_choice": norm_choice
                             }

    fig.savefig(path + f'{timestr}_{file}_singular_values_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_singular_values_{graph_str}.png')

    with open(path + f'{timestr}_{file}_singular_values_W'
                     f'_{graph_str}.json', 'w') \
            as outfile:
        json.dump(singularValues.tolist(), outfile)
    with open(path + f'{timestr}_{file}_singular_values_EW'
                     f'_{graph_str}.json', 'w') \
            as outfile:
        json.dump(singularValues_EW.tolist(), outfile)
    with open(path + f'{timestr}_{file}_singular_values_R'
                     f'_{graph_str}.json', 'w') \
            as outfile:
        json.dump(singularValues_R.tolist(), outfile)
    with open(path + f'{timestr}_{file}_singular_values'
                     f'_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
