# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.dynamics import rnn
from dynamics.reduced_dynamics import reduced_rnn_vector_field
from dynamics.error_vector_fields import *
from singular_values.optimal_shrinkage import optimal_shrinkage
from singular_values.compute_effective_ranks import computeRank
from graphs.get_real_networks import *
from scipy.linalg import pinv
from tqdm import tqdm
from plots.config_rcparams import *
from plots.plot_singular_values import plot_singular_values
import time
import json
import tkinter.simpledialog
from tkinter import messagebox


plot_singvals = False

""" Graph parameters """
graph_str = "mouse_control_rnn"
A = get_learned_weight_matrix(graph_str)
N = len(A[0])  # Dimension of the complete dynamics

""" Dynamical parameters """
dynamics_str = "rnn"
t = 0  # Time is not involved in the vector-field error computation
D = np.eye(N)/0.625
# tau = 0.625 in Hadjiabadi et al. Maximally selective
# single-cell target for circuit control in epilepsy models

""" SVD """
U, S, Vh = np.linalg.svd(A)
if plot_singvals:
    plot_singular_values(S, effective_ranks=True, cum_explained_var=False,
                         ysemilog=False)
shrink_s = optimal_shrinkage(S, 1, 'operator')
A = U@np.diag(shrink_s)@Vh

N_arange = np.arange(1, computeRank(shrink_s), 1)
"""                                                                              
Comment                                                                          
-------                                                                          
Above, we use computeRank(shrink_s) because the error 0 for n = rank. 
We remove it to do a semilog plot.                                                            
See https://www.youtube.com/watch?v=ydxy3fEar9M for transforming an error for    
a semilog plot.                                                                  
"""
nb_samples = 100
error_array = np.zeros((nb_samples, len(N_arange)))
error_upper_bound_array = np.zeros((nb_samples, len(N_arange)))
for n in tqdm(N_arange):

    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign@Vhn
    W = A   # /S[0]  # We normalize the network by the largest singular value
    Mp = pinv(M)
    x_samples = np.random.uniform(0, 1, (N, nb_samples))
    min_coupling, max_coupling = 0.1/0.625, 3/0.625
    coupling_samples = np.random.uniform(min_coupling, max_coupling,
                                         nb_samples)
    for i in range(0, nb_samples):
        x = x_samples[:, i]
        coupling = coupling_samples[i]
        error_array[i, n-1] = \
            rmse(M@rnn(t, x, W, coupling, D),
                 reduced_rnn_vector_field(t, M@x, W, coupling, M, Mp, D))
        # xp = x_prime_SIS(x, W, M)
        # error_upper_bound_array[i, n-1] = \
        #     error_upper_bound_SIS(x, xp, W, coupling, D, n, S/S[0], M)

mean_error = np.mean(error_array, axis=0)
mean_log10_error = np.log10(mean_error)
relative_std_semilogplot_error = \
    np.std(error_array, axis=0) / (np.log(10) * mean_error)
fill_between_error_1 = 10**(
        mean_log10_error - relative_std_semilogplot_error)
fill_between_error_2 = 10**(
        mean_log10_error + relative_std_semilogplot_error)

# mean_upper_bound_error = np.mean(error_upper_bound_array[:, :-1], axis=0)
# mean_log10_upper_bound_error = np.log10(mean_upper_bound_error)
# relative_std_semilogplot_upper_bound_error = \
#     np.std(error_upper_bound_array[:, :-1], axis=0) / \
#     (np.log(10) * mean_upper_bound_error)
# fill_between_ub1 = 10**(mean_log10_upper_bound_error -
#                         relative_std_semilogplot_upper_bound_error)
# fill_between_ub2 = 10**(mean_log10_upper_bound_error +
#                         relative_std_semilogplot_upper_bound_error)

fig = plt.figure(figsize=(4, 4))
ax = plt.subplot(111)
ax.scatter(N_arange, mean_error, s=5, color=deep[3],
           label="RMSE $\\mathcal{E}_f\,(x)$")
# for i in range(1, nb_samples):
#     ax.scatter(N_arange, error_array[i, :],
#                color=deep[3], s=5, alpha=0.1)
# ax.plot(N_arange, mean_upper_bound_error, color=dark_grey,
#         label="Upper bound")
ax.fill_between(N_arange, fill_between_error_1, fill_between_error_2,
                color=deep[3], alpha=0.5)
# ax.fill_between(N_arange, fill_between_ub1, fill_between_ub2,
#                 color=dark_grey, alpha=0.5)
plt.xlabel('Dimension $n$')
ticks = ax.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])
plt.xlim([-0.01*len(N_arange), 1.01*len(N_arange)])
# plt.ylim([0.9*np.min(fill_between_error_1),
#           1.1*np.max(fill_between_ub2)])
ax.legend(loc=1, frameon=True, edgecolor="#e0e0e0")
ax.set_yscale('log')
plt.tick_params(axis='y', which='both', left=True,
                right=False, labelbottom=False)
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
    with open(path + f'{timestr}_{file}_RMSE_vector_field'
                     f'_{dynamics_str}_{graph_str}.json', 'w') \
            as outfile:
        json.dump(error_array.tolist(), outfile)
    with open(path + f'{timestr}_{file}_upper_bound_RMSE_vector_field'
                     f'_{dynamics_str}_{graph_str}.json', 'w') \
            as outfile:
        json.dump(error_upper_bound_array.tolist(), outfile)
    with open(path + f'{timestr}_{file}'
              f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
