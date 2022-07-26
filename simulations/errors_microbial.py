# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.dynamics import microbial
from dynamics.reduced_dynamics import reduced_microbial_vector_field
from dynamics.error_vector_fields import *
from graphs.get_real_networks import *
from tqdm import tqdm
from plots.config_rcparams import *
from plots.plot_singular_values import plot_singular_values
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

""" Graph parameters """
graph_str = "gut"
A = get_microbiome_weight_matrix(graph_str)
N = len(A[0])  # Dimension of the complete dynamics

""" Dynamical parameters """
dynamics_str = "microbial"
t = 0  # It is not involved in the error computation

# 1. The ones used in SI and in Sanhedrai et al., Nat. Phys. (2022).
# a, b, c, D = 5, 13, 10/3, 30*np.eye(N)

# 2. The ones used in the paper to compute the upper bound on the error.
a, b, c, D = 0.00005, 0.1, 0.9, 0.01*np.eye(N)

# 3.
# a, b, c, D = 0, 0.15, 72, 0*np.eye(N)


""" SVD """
U, S, Vh = np.linalg.svd(A)
# plot_singular_values(S)


""" Simulations """
N_arange = np.arange(1, N, 1)
nb_samples = 1
# for 1000 samples, it took 20h30min
error_array = np.zeros((nb_samples, N))
error_upper_bound_array = np.zeros((nb_samples, N))
for n in tqdm(N_arange):
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign @ Vhn
    W = A / S[0]  # We normalize the network by the largest singular value
    Mp = np.linalg.pinv(M)
    P = Mp@M
    x_samples = np.random.uniform(0, 1, (N, nb_samples))

    # Test zone
    # -----------------------------------
    # xtest = x_samples[:, 0]
    # gamma = 3
    # P = Mp @ M
    # Dchi = np.diag((np.eye(N) - P)@xtest)
    # DWchi = np.diag(W@(np.eye(N) - Mp @ M)@xtest)
    # d = b*(xtest**2 - (P@xtest)**2) + c*(xtest**3 - (P@xtest)**3)\
    #     + gamma*(xtest*(W@xtest) - (P@xtest)*(W@P@xtest))
    # print(f"\n\nmax a ={np.max(3*c*Dchi@xtest**2)}\n\n",
    #       f"max b = {np.max((2*b*Dchi + gamma*DWchi
    #  + gamma*Dchi@W)@xtest)}\n\n",
    #       f"max bcoup = {np.max(gamma*Dchi@W@xtest)}\n\n",
    #       f"max c = {np.max(d)}")
    # print(f"\n\nmean a ={np.mean(3*c*Dchi@xtest**2)}\n\n",
    #       f"mean b = {np.mean((2*b*Dchi + gamma*DWchi
    #  + gamma*Dchi@W)@xtest)}\n\n",
    #       f"mean bcoup = {np.mean(gamma*Dchi@W@xtest)}\n\n",
    #       f"mean c = {np.mean(d)}")

    # print(f"\n\nmax a ={np.max(xtest**2)}\n\n",
    #       f"max b = {np.max(Dchinv@(2*b*Dchi
    # + gamma*Dchi@W + gamma*DWchi)@xtest/(3*c))}\n\n",
    #       f"max c = {np.max(Dchinv@d/(3*c))}")
    # print(f"\n\nmean a ={np.mean(xtest**2)}\n\n",
    #       f"mean b = {np.mean(Dchinv@(2*b*Dchi
    # + gamma*Dchi@W + gamma*DWchi)@xtest/(3*c))}\n\n",
    #       f"mean c = {np.mean(Dchinv@d/(3*c))}")
    # -----------------------------------
    min_coupling = 0.1   # 1  # 0.1 # 0.5
    max_coupling = 5     # 6  # 5   # 3
    coupling_samples = np.random.uniform(min_coupling, max_coupling,
                                         nb_samples)

    # a, b, c, D = 0.00005, 0.1, 0.9, 0.01*np.eye(N)
    min_a, max_a = 0.00001, 0.0001
    a_samples = np.random.uniform(min_a, max_a, nb_samples)

    min_b, max_b = 0.05, 2
    b_samples = np.random.uniform(min_b, max_b, nb_samples)

    min_c, max_c = 0.5, 1.5
    c_samples = np.random.uniform(min_c, max_c, nb_samples)

    mean_D, std_D = 0.05, 0.0005

    for i in range(0, nb_samples):
        x = x_samples[:, i]
        coupling = coupling_samples[i]
        a, b, c = a_samples[i], b_samples[i], c_samples[i]
        D = np.diag(np.random.normal(mean_D, std_D, N))

        # RMSE
        error_array[i, n - 1] = \
            rmse(M@microbial(t, x, W, coupling, D, a, b, c),
                 reduced_microbial_vector_field(t, M@x, W, coupling,
                                                M, Mp, D, a, b, c))

        # Upper bound RMSE
        xp = x_prime_microbial(x, W, P, coupling, b, c)
        """ Test zone """
        # print(xp)
        # chi = (np.eye(N) - P)@x
        # print(
        #   coupled_quadratic_equations_microbial(xp, x, W, P, coupling, b, c))
        # p = P@x
        # AA = 3*c*chi
        # print(np.shape(W@chi), chi)
        # print(np.diag(W@chi))
        # BB = 2*b*np.diag(chi) + coupling*np.diag(chi)@W
        # + coupling*np.diag(W@chi)
        # CC = b*(x**2 - p**2) + c*(x**3 - p**3) + coupling*(x*(W@x) - p*(W@p))
        # print(f"<A> = {np.mean(AA)}, <B> = {np.mean(BB)},
        #  <C> = {np.mean(CC)}")
        Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
        Jy = jacobian_y_microbial(xp, coupling)
        evec = error_vector_fields_upper_bound(x, Jx, Jy, S/S[0], M, P)
        if np.isnan(evec):
            error_upper_bound_array[i, n - 1] = \
                error_upper_bound_array[i, n - 2]
        else:
            error_upper_bound_array[i, n - 1] = evec

# fig = plt.figure(figsize=(4, 4))
# plt.subplot(111)
# plt.scatter(np.arange(1, len(mean_error_list) + 1, 1), mean_error_list,
#             color=first_community_color, s=5)
# ylab = plt.ylabel(f'Mean RMSE between\n the vector fields')
# plt.xlabel('Dimension $n$')
# plt.tight_layout()
# plt.show()
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
    np.std(error_array[:, :-1], axis=0) / (np.log(10) * mean_error)
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

fig = plt.figure(figsize=(4, 4))
ax = plt.subplot(111)
ax.scatter(N_arange, mean_error, s=5, color=deep[3],
           label="RMSE $\\mathcal{E}_f\,(x)$")
ax.plot(N_arange, mean_upper_bound_error, color=dark_grey,
        label="Upper bound")
ax.fill_between(N_arange, fill_between_error_1, fill_between_error_2,
                color=deep[3], alpha=0.5)
ax.fill_between(N_arange, fill_between_ub1, fill_between_ub2,
                color=dark_grey, alpha=0.5)
plt.xlabel('Dimension $n$')
ticks = ax.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])
plt.xlim([-0.01*N, 1.01*N])
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
