# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.dynamics import rnn
from dynamics.reduced_dynamics import reduced_rnn_vector_field
from dynamics.error_vector_fields import *
from singular_values.optimal_shrinkage import optimal_shrinkage
from singular_values.compute_effective_ranks import computeRank, \
    computeEffectiveRanks
from graphs.get_real_networks import *
from tqdm import tqdm
from plots.config_rcparams import *
from plots.plot_singular_values import plot_singular_values
import time
import json
import tkinter.simpledialog
from tkinter import messagebox


plot_singvals = True

""" Graph parameters """
graph_str = "mouse_control_rnn"
A = get_learned_weight_matrix(graph_str)
N = len(A[0])  # Dimension of the complete dynamics

""" Dynamical parameters """
dynamics_str = "rnn"
t = 0  # Time is not involved in the vector-field error computation

""" SVD """
U, S, Vh = np.linalg.svd(A)
if plot_singvals:
    print(computeEffectiveRanks(S, graph_str, N))
    plot_singular_values(S, effective_ranks=False, cum_explained_var=False,
                         ysemilog=False)
shrink_s = optimal_shrinkage(S, 1, 'frobenius')
A = U@np.diag(shrink_s)@Vh

N_arange = np.arange(1, computeRank(shrink_s)+1, 1)
"""                                                                              
Comment                                                                          
-------                                                                          
Above, we use computeRank(shrink_s) because the error is 0 for n = rank in the
case of a RNN. 
We remove it to do a semilog plot.                                                            
See https://www.youtube.com/watch?v=ydxy3fEar9M for transforming an error for    
a semilog plot.                                                                  
"""
nb_samples = 1000
error_array = np.zeros((nb_samples, len(N_arange)))
error_upper_bound_array = np.zeros((nb_samples, len(N_arange)))
# error_upper_bound_triangle_array = np.zeros((nb_samples, len(N_arange)))
# error_upper_bound_induced_array = np.zeros((nb_samples, len(N_arange)))
for n in tqdm(N_arange):

    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign@Vhn
    W = A   # /S[0]  # To normalize the network by the largest singular value
    Mp = np.linalg.pinv(M)
    P = Mp@M
    x_samples = np.random.uniform(-1, 1, (N, nb_samples))
    # tau = 0.625 in Hadjiabadi et al. Maximally selective
    # single-cell target for circuit control in epilepsy models
    min_coupling, max_coupling = 0.1/0.625, 3/0.625
    coupling_samples = np.random.uniform(min_coupling, max_coupling,
                                         nb_samples)
    mean_D, std_D = 1/0.625, 0.001
    for i in range(0, nb_samples):
        x = x_samples[:, i]
        coupling = coupling_samples[i]
        D = np.diag(np.random.normal(mean_D, std_D, N))
        error_array[i, n-1] = \
            rmse(M@rnn(t, x, W, coupling, D),
                 reduced_rnn_vector_field(t, M@x, W, coupling, M, Mp, D))

        dsig_prime = derivative_sigmoid_prime_rnn(x, W, P, coupling)
        Jx = -D
        Jy = jacobian_y_rnn(dsig_prime, coupling)
        error_upper_bound_array[i, n-1] = \
            error_vector_fields_upper_bound(x, Jx, Jy, shrink_s, M, P)

        # error_upper_bound_triangle_array[i, n-1] = \
        #     error_vector_fields_upper_bound_triangle(x, Jx, Jy, W, M, P)
        # error_upper_bound_induced_array[i, n-1] = \
        #     error_vector_fields_upper_bound_induced_norm(x, Jx, Jy, W, M, P)

        # print(norm(M@Jy@W@(np.eye(N) - P)@x),
        #       norm(M@Jy@W@(np.eye(N) - P), ord=2)*norm(x),
        #       norm(M@Jy, ord=2)*norm(W@(np.eye(N) - P), ord=2)*norm(x),
        #       norm(Jy, ord=2))

mean_error = np.mean(error_array, axis=0)
mean_log10_error = np.log10(mean_error)
relative_std_semilogplot_error = \
    np.std(error_array, axis=0) / (np.log(10)*mean_error)
fill_between_error_1 = 10**(
        mean_log10_error - relative_std_semilogplot_error)
fill_between_error_2 = 10**(
        mean_log10_error + relative_std_semilogplot_error)

mean_upper_bound_error = np.mean(error_upper_bound_array, axis=0)
mean_log10_upper_bound_error = np.log10(mean_upper_bound_error)
relative_std_semilogplot_upper_bound_error = \
    np.std(error_upper_bound_array, axis=0) / \
    (np.log(10) * mean_upper_bound_error)
fill_between_ub1 = 10**(mean_log10_upper_bound_error -
                        relative_std_semilogplot_upper_bound_error)
fill_between_ub2 = 10**(mean_log10_upper_bound_error +
                        relative_std_semilogplot_upper_bound_error)

# mean_upper_bound_triangle_error = \
#     np.mean(error_upper_bound_triangle_array, axis=0)
# mean_upper_bound_induced_error = \
#     np.mean(error_upper_bound_induced_array, axis=0)

fig = plt.figure(figsize=(4, 4))
ax = plt.subplot(111)
# for i in range(1, nb_samples):
#     ax.scatter(N_arange, error_array[i, :],
#                color=deep[3], s=5, alpha=0.1)
ax.scatter(N_arange, mean_error, s=5, color=deep[3],
           label="RMSE $\\mathcal{E}\,(x)$")
ax.plot(N_arange, mean_upper_bound_error, color=dark_grey,
        label="Upper bound")
# ax.plot(N_arange, mean_upper_bound_triangle_error, color=deep[4],
#         label="Upper bound (triangle)")
# ax.plot(N_arange, mean_upper_bound_induced_error, color=deep[6],
#         label="Upper bound (triangle+induced)")
ax.fill_between(N_arange, fill_between_error_1, fill_between_error_2,
                color=deep[3], alpha=0.5)
ax.fill_between(N_arange, fill_between_ub1, fill_between_ub2,
                color=dark_grey, alpha=0.5)
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
    path = f'C:/Users/thivi/Documents/GitHub/' \
           f'low-rank-hypothesis-complex-systems/' \
           f'simulations/simulations_data/{dynamics_str}_data/' \
           f'vector_field_errors/'
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    parameters_dictionary = {
        "graph_str": graph_str, "N": N,   # "D": D.tolist(),
        "nb_samples": nb_samples, "N_arange": N_arange.tolist(),
        "x_samples": "np.random.uniform(0, 1, (N, nb_samples))",
        "coupling_samples": f"np.random.uniform({min_coupling},"
                            f" {max_coupling}, nb_samples)",
        "D_samples": f"np.diag(np.random.normal({mean_D}, {std_D}, N))"}

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
