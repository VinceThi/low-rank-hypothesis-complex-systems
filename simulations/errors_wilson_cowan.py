# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.dynamics import wilson_cowan
from dynamics.reduced_dynamics import reduced_wilson_cowan_vector_field
from dynamics.error_vector_fields import *
from graphs.get_real_networks import *
from scipy.linalg import pinv
from tqdm import tqdm
from plots.config_rcparams import *
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

""" Graph parameters """
graph_str = "ciona"
A = get_connectome_weight_matrix(graph_str)
N = len(A[0])  # Dimension of the complete dynamics

""" Dynamical parameters """
dynamics_str = "wilson_cowan"
t = 0  # It is not involved in the error computation
D = np.eye(N)
a = 0.1
b = 1
c = 3

""" SVD """
U, S, Vh = np.linalg.svd(A)

nb_samples = 1000
mean_error_list = []
for n in tqdm(range(1, N, 1)):
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign@Vhn
    W = A / S[0]  # We normalize the network by the largest singular value
    Mp = pinv(M)
    calD = M@D@Mp
    x_samples = np.random.uniform(0, 1, (N, nb_samples))
    min_coupling = 12
    max_coupling = 15
    coupling_samples = np.random.uniform(min_coupling, max_coupling,
                                         nb_samples)
    error_list = []
    for i in range(0, nb_samples):
        x = x_samples[:, i]
        coupling = coupling_samples[i]
        error_list.append(rmse(
            M@wilson_cowan(t, x, W, coupling, D, a, b, c),
            reduced_wilson_cowan_vector_field(t, M@x, W, coupling,
                                              M, Mp, D, a, b, c)))
    mean_error_list.append(np.mean(error_list))

fig = plt.figure(figsize=(4, 4))
plt.subplot(111)
plt.scatter(np.arange(1, len(mean_error_list)+1, 1), mean_error_list,
            color=first_community_color, s=5)
ylab = plt.ylabel(f'Mean RMSE between\n the vector fields')
plt.xlabel('Dimension $n$')
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
