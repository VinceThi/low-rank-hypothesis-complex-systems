# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.dynamics import qmf_sis
from dynamics.reduced_dynamics import reduced_qmf_sis_vector_field
from dynamics.error_vector_fields import *
from graphs.get_real_networks import *
from plots.plot_singular_values import plot_singular_values
from scipy.linalg import pinv
from tqdm import tqdm
from plots.config_rcparams import *
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

plot_singvals = False

""" Graph parameters """
graph_str = "high_school_proximity"
A = get_epidemiological_weight_matrix(graph_str)
N = len(A[0])  # Dimension of the complete dynamics

""" Dynamical parameters """
dynamics_str = "qmf_sis"
t = 0  # It is not involved in the error computation
D = np.eye(N)

""" SVD """
U, S, Vh = np.linalg.svd(A)
if plot_singvals:
    plot_singular_values(S)

nb_samples = 10
mean_error_list = []
for n in tqdm(range(1, N, 1)):

    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign @ Vhn
    W = A / S[0]  # We normalize the network by the largest singular value
    Mp = pinv(M)
    x_samples = np.random.uniform(0, 1, (N, nb_samples))
    min_coupling, max_coupling = 0.01, 4
    coupling_samples = np.random.uniform(min_coupling, max_coupling,
                                         nb_samples)
    error_list = []
    for i in range(0, nb_samples):
        x = x_samples[:, i]
        coupling = coupling_samples[i]
        error_list.append(rmse(
            M@qmf_sis(t, x, W, coupling, D),
            reduced_qmf_sis_vector_field(t, M@x, W, coupling, M, Mp, D)))
    mean_error_list.append(np.mean(error_list))

fig = plt.figure(figsize=(4, 4))
ax = plt.subplot(111)
ax.scatter(np.arange(1, len(mean_error_list)+1, 1), mean_error_list,
           color=first_community_color, s=5)
ylab = plt.ylabel(f'Mean RMSE between\n the vector fields')
plt.xlabel('Dimension $n$')
ticks = ax.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters,"
                       " the data, and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    path = f'C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/' \
           f'simulations/simulations_data/{dynamics_str}_data/' \
           f'vector_field_errors/'
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    parameters_dictionary = {
        "graph_str": graph_str, "D": D.tolist(), "N": N,
        "nb_samples": nb_samples,
        "x_samples": "np.random.uniform(0, 1, (N, nb_samples))",
        "coupling_samples": f"np.random.uniform({min_coupling},"
                            f" {max_coupling}, nb_samples)"}

    fig.savefig(path + f'{timestr}_{file}_vector_field_rmse_error_vs_n'
                f'_{dynamics_str}_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_vector_field_rmse_error_vs_n'
                       f'_{dynamics_str}_{graph_str}.png')
    with open(path + f'{timestr}_{file}'
              f'_mean_error_list'
              f'_complete_{dynamics_str}_{graph_str}.json', 'w') \
            as outfile:
        json.dump(mean_error_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
