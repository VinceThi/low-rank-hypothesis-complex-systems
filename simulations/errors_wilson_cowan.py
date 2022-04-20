# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.dynamics import wilson_cowan
from dynamics.reduced_dynamics import reduced_wilson_cowan_vector_field
from dynamics.error_vector_fields import *
from graphs.get_real_networks import *
from tqdm import tqdm
from plots.config_rcparams import *
from plots.plot_singular_values import plot_singular_values
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

plot_singvals = False

""" Graph parameters """
graph_str = "celegans_signed"
A = get_connectome_weight_matrix(graph_str)
N = len(A[0])  # Dimension of the complete dynamics

""" Dynamical parameters """
dynamics_str = "wilson_cowan"
t = 0  # Time is not involved in the vector-field error computation
# D = np.eye(N)
# a = 1
# b = 1
# c = 3

""" SVD """
U, S, Vh = np.linalg.svd(A)
if plot_singvals:
    plot_singular_values(S, effective_ranks=True, cum_explained_var=False,
                         ysemilog=True)

N_arange = np.arange(1, N, 1)
nb_samples = 1000
error_array = np.zeros((nb_samples, len(N_arange)))
error_upper_bound_array = np.zeros((nb_samples, len(N_arange)))
# error_upper_bound_array2 = np.zeros((nb_samples, len(N_arange)))
for n in tqdm(N_arange):

    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float))\
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign@Vhn
    W = A   # /S[0]  # We normalize the network by the largest singular value
    Mp = np.linalg.pinv(M)
    P = Mp@M
    x_samples = np.random.uniform(0, 1, (N, nb_samples))
    min_coupling, max_coupling = 0.01, 1
    coupling_samples = np.random.uniform(min_coupling, max_coupling,
                                         nb_samples)
    min_a, max_a = 0.001, 0.1
    a_samples = np.random.uniform(min_a, max_a, nb_samples)

    min_b, max_b = 0.5, 2
    b_samples = np.random.uniform(min_b, max_b, nb_samples)

    min_c, max_c = 2, 4
    c_samples = np.random.uniform(min_c, max_c, nb_samples)

    mean_D, std_D = 1, 0.001

    for i in range(0, nb_samples):
        x = x_samples[:, i]
        Px = P@x
        coupling = coupling_samples[i]
        a, b, c = a_samples[i], b_samples[i], c_samples[i]
        D = np.diag(np.random.normal(mean_D, std_D, N))
        error_array[i, n-1] = \
            rmse(M@wilson_cowan(t, x, W, coupling, D, a, b, c),
                 reduced_wilson_cowan_vector_field(t, M@x, W, coupling,
                                                   M, Mp, D, a, b, c))

        """ Least-square method """
        # xp = x_prime_wilson_cowan(x, W, P, coupling, a, b, c)
        # # The problem is that there are many solutions for xp and
        # # we might not find the xp that minimizes the upper bound.
        # # At least, one can find a correct xp, up to some tolerance.
        # Jx = jacobian_x_wilson_cowan(xp, W, coupling, D, a, b, c)
        # Jy = jacobian_y_wilson_cowan(xp, W, coupling, a, b, c)
        #
        # error_upper_bound_array[i, n - 1] = \
        #     error_vector_fields_upper_bound(x, Jx, Jy, S, M, P)

        """ Approximate method (parameter 'a' small) """
        # # The problem is that for large n, the error depending on the
        # # parameter 'a' become non negligeable and the approximate error
        # # upper bound might be below the true error.
        # dsigp = derivative_sigmoid_prime_wilson_cowan(x, W, P, coupling, b,c)
        # Jx = -D
        # Jy = jacobian_y_approx_wilson_cowan(dsigp, coupling, b)
        # error_upper_bound_array[i, n - 1] = \
        #     error_vector_fields_upper_bound(x, Jx, Jy, S, M, P)

        """ Approximate method 2 : choose x' = x or Px"""
        # Problem we do not know x', we naively choose the max in {x, Px}
        # but it should be the max in the ***interval*** between x and Px.
        # Yet, if x is close to Px, this is not a bad choice. This will happen
        # especially for large n.
        Jx = jacobian_x_wilson_cowan(x, W, coupling, D, a, b, c)
        Jy = jacobian_y_wilson_cowan(x, W, coupling, a, b, c)
        e_x = error_vector_fields_upper_bound(x, Jx, Jy, S, M, P)

        Jx_tilde = jacobian_x_wilson_cowan(P@x, W, coupling, D, a, b, c)
        Jy_tilde = jacobian_y_wilson_cowan(P@x, W, coupling, a, b, c)
        e_Px = error_vector_fields_upper_bound(P@x, Jx_tilde, Jy_tilde,
                                               S, M, P)
        error_upper_bound_array[i, n - 1] = max(e_x, e_Px)


"""                                                                              
Comment                                                                          
-------                                                                          
Below, we use error_...[0, :-1] because the column is error 0 (n = N). 
We remove it to do a semilog plot.                                                            
See https://www.youtube.com/watch?v=ydxy3fEar9M for transforming an error for    
a semilog plot.                                                                  
"""
mean_error = np.mean(error_array[:, :-1], axis=0)
mean_log10_error = np.log10(mean_error)
relative_std_semilogplot_error = \
    np.std(error_array[:, :-1], axis=0)/(np.log(10)*mean_error)
fill_between_error_1 = 10**(
        mean_log10_error - relative_std_semilogplot_error)
fill_between_error_2 = 10**(
        mean_log10_error + relative_std_semilogplot_error)

mean_upper_bound_error = np.mean(error_upper_bound_array[:, :-1], axis=0)
mean_log10_upper_bound_error = np.log10(mean_upper_bound_error)
relative_std_semilogplot_upper_bound_error = \
    np.std(error_upper_bound_array[:, :-1], axis=0) / \
    (np.log(10) * mean_upper_bound_error)
fill_between_ub1 = 10**(mean_log10_upper_bound_error -
                        relative_std_semilogplot_upper_bound_error)
fill_between_ub2 = 10**(mean_log10_upper_bound_error +
                        relative_std_semilogplot_upper_bound_error)

# mean_upper_bound_error2 = np.mean(error_upper_bound_array2[:, :-1], axis=0)

# mean_upper_bound_triangle_error = \
#     np.mean(error_upper_bound_triangle_array, axis=0)
# mean_upper_bound_induced_error = \
#     np.mean(error_upper_bound_induced_array, axis=0)

fig = plt.figure(figsize=(4, 4))
ax = plt.subplot(111)
ax.scatter(N_arange[:-1], mean_error, s=5, color=deep[3],
           label="RMSE $\\mathcal{E}_f\,(x)$")
ax.plot(N_arange[:-1], mean_upper_bound_error, color=dark_grey,
        label="Approximate upper bound")
# ax.plot(N_arange[:-1], mean_upper_bound_error2, color=deep[9],
#         label="Upper bound (part)")
# ax.plot(N_arange[:-1], np.sqrt(N)*np.ones(len(N_arange[:-1]))/2,
#  color=deep[9], label="Upper bound (part)")

ax.fill_between(N_arange[:-1], fill_between_error_1, fill_between_error_2,
                color=deep[3], alpha=0.5)
ax.fill_between(N_arange[:-1], fill_between_ub1, fill_between_ub2,
                color=dark_grey, alpha=0.5)
# ax.plot(N_arange[:-1], S[:-2]/S[0], color=deep[9],
#         label="$\\sigma_n/\\sigma_1$")
# ax.plot(N_arange[:-1], S[:n-1] - S[1:n], color=deep[8],
#         label="$\\sigma_n - \\sigma_{n-1}$")
plt.xlabel('Dimension $n$')
ticks = ax.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])
plt.xlim([-0.01*N, 1.01*N])
plt.ylim([0.9*np.min(fill_between_error_1),
          1.1*np.max(fill_between_ub2)])
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
        "graph_str": graph_str, "N": N,  # "D": D.tolist(),
        "nb_samples": nb_samples,  # "a": a, "b": b, "c": c,
        "N_arange": N_arange.tolist(),
        "x_samples": "np.random.uniform(0, 1, (N, nb_samples))",
        "coupling_samples": f"np.random.uniform({min_coupling},"
                            f" {max_coupling}, nb_samples)",
        "a_samples": f"np.random.uniform({min_a}, {max_a}, nb_samples)",
        "b_samples": f"np.random.uniform({min_b}, {max_b}, nb_samples)",
        "c_samples": f"np.random.uniform({min_c}, {max_c}, nb_samples)",
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
