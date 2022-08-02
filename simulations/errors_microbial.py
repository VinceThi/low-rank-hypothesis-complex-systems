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

plot_singular_values_bool = False
""" 
                     Choice of approximation methods:
            "neglect_linear_term", "max_x_Px", or "optimization"
            
- neglect_linear_term: it doesn't give good results
- max_x_Px: good enough result, but not always above the error for an instance
- optimization: very long simulations, does not guarantee a good x'            
"""
approximation_method = "optimization"


""" 
                           Choice of setup:
                               1 or 2

- setup = 1 uses the parameters of Sanhedrai et al. 2022. The trajectories
  are not bounded between 0 and 1, but it is useful to get bifurcations.
- setup = 2 is what we use in the paper for the upper bound, since the
  trajectories are below one for the set of parameters used                
"""
#
setup = 2


""" Graph parameters """
graph_str = "gut"
adj = get_microbiome_weight_matrix(graph_str)
N = len(adj[0])  # Dimension of the complete dynamics

""" Dynamical parameters """
dynamics_str = "microbial"
t = 0  # It is not involved in the error computation
# The parameters are define below

""" SVD """
U, S, Vh = np.linalg.svd(adj)
if plot_singular_values_bool:
    plot_singular_values(S)


""" Simulations """
if approximation_method == "optimization":
    N_arange = np.array([1, 50, 200, 400, 600, 800, 830])
    nb_samples = 10
else:
    N_arange = np.arange(1, N, 1)  # We do not compute the case n = N
    nb_samples = 1000
    # for 1000 samples, it took 35h for max_x_Px on a personal laptop (i7)

error_array = np.zeros((nb_samples, len(N_arange)))
error_upper_bound_array = np.zeros((nb_samples, len(N_arange)))
for count, n in tqdm(enumerate(N_arange)):
    print(f"{(count+1)}/{len(N_arange)}")
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign @ Vhn
    if setup == 1:
        W = adj
    elif setup == 2:
        W = adj / S[0]  # We normalize the network by the largest sing. value
    Mp = np.linalg.pinv(M)
    P = Mp@M

    # Test zone
    # -----------------------------------
    # xtest = x_samples[:, 0]
    # gamma = 3
    # chi = (np.eye(N) - P)@xtest
    # Dchi = np.diag(chi)
    # DWchi = np.diag(W@chi)
    # d = b*(xtest**2 - (P@xtest)**2) - c*(xtest**3 - (P@xtest)**3)\
    #     + gamma*(xtest*(W@xtest) - (P@xtest)*(W@P@xtest))
    # print(f"\n\nmean |a| ={np.mean(np.abs(-3*c*Dchi@(xtest**2)))}\n\n",
    #       f"mean |b| = "
    #       f"{np.mean(np.abs((2*b*Dchi+gamma*DWchi+gamma*Dchi@W)@xtest))}\n\n",
    #       f"mean |bcoup| = {np.mean(np.abs(gamma*Dchi@W@xtest))}\n\n",
    #       f"mean |c| = {np.mean(np.abs(d))}")

    if setup == 1:
        W = adj
        singvals_W = S

        max_x = 20
        x_samples = np.random.uniform(0, max_x, (N, nb_samples))

        min_coupling = 0.5
        max_coupling = 3
        coupling_samples = np.random.uniform(min_coupling, max_coupling,
                                             nb_samples)

        min_a, max_a = 4, 6
        a_samples = np.random.uniform(min_a, max_a, nb_samples)

        min_b, max_b = 12, 13
        b_samples = np.random.uniform(min_b, max_b, nb_samples)

        min_c, max_c = 2, 4
        c_samples = np.random.uniform(min_c, max_c, nb_samples)

        mean_D, std_D = 0.01, 0.0001

    elif setup == 2:
        W = adj/S[0]
        singvals_W = S/S[0]

        max_x = 1
        x_samples = np.random.uniform(0, max_x, (N, nb_samples))

        min_coupling = 0.1
        max_coupling = 5
        coupling_samples = np.random.uniform(min_coupling, max_coupling,
                                             nb_samples)

        min_a, max_a = 0.00001, 0.0001
        a_samples = np.random.uniform(min_a, max_a, nb_samples)

        min_b, max_b = 0.05, 2
        b_samples = np.random.uniform(min_b, max_b, nb_samples)

        min_c, max_c = 0.5, 1.5
        c_samples = np.random.uniform(min_c, max_c, nb_samples)

        D = 0.01*np.eye(N)

    for i in range(0, nb_samples):
        x = x_samples[:, i]
        coupling = coupling_samples[i]
        a, b, c = a_samples[i], b_samples[i], c_samples[i]
        if setup == 1:
            D = np.diag(np.random.normal(mean_D, std_D, N))

        # RMSE
        error_array[i, count] = \
            rmse(M@microbial(t, x, W, coupling, D, a, b, c),
                 reduced_microbial_vector_field(t, M@x, W, coupling,
                                                M, Mp, D, a, b, c))

        # Upper bound RMSE
        if approximation_method == "neglect_linear_term":
            """ Neglect the the linear term Bx' """
            p = P@x
            A = A_microbial(x, p, c)
            B = B_microbial(x, p, W, b, coupling)
            C = C_microbial(x, p, W, b, c, coupling)
            xp = x_prime_microbial_neglect_linear_term(A, C)
            # print(f"<|A|>= {np.mean(np.abs(A))}, "
            #       f"<|B|>= {np.mean(np.abs(B))},"
            #       f" <|C|>= {np.mean(np.abs(C))}")
            Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
            Jy = jacobian_y_microbial(xp, coupling)
            evec = error_vector_fields_upper_bound(x, Jx, Jy, singvals_W, M, P)
            if np.isnan(evec):
                error_upper_bound_array[i, count] = \
                    error_upper_bound_array[i, count - 1]
            else:
                error_upper_bound_array[i, count] = evec

        elif approximation_method == "max_x_Px":
            """ We choose x' as x or Px, choosing the one that gives the 
            maximum value of the error bound. """
            # We naively choose the max in {x, Px} even if it should be
            # the max in the ***interval*** between x and Px.
            # Yet, if x is close to Px, this is not a bad choice.
            # This gives better results especially for large n.
            # For the population dynamics on the gut microbiome, this gives a
            # very crude results for small n.
            Jx = jacobian_x_microbial(x, W, coupling, D, b, c)
            Jy = jacobian_y_microbial(x, coupling)

            Jx_tilde = jacobian_x_microbial(P@x, W, coupling, D, b, c)
            Jy_tilde = jacobian_y_microbial(P@x, coupling)
            e_x = error_vector_fields_upper_bound(x, Jx, Jy, singvals_W, M, P)
            e_Px = error_vector_fields_upper_bound(P@x, Jx_tilde, Jy_tilde,
                                                   singvals_W, M, P)

            error_upper_bound_array[i, count] = max(e_x, e_Px)

        elif approximation_method == "optimization":
            p = P@x
            A = A_microbial(x, p, c)
            B = B_microbial(x, p, W, b, coupling)
            C = C_microbial(x, p, W, b, c, coupling)
            xp = x_prime_microbial_optimize(A, B, C, max_x=max_x)
            Jx = jacobian_x_microbial(xp, W, coupling, D, b, c)
            Jy = jacobian_y_microbial(xp, coupling)

            evec = error_vector_fields_upper_bound(x, Jx, Jy, singvals_W, M, P)

            error_upper_bound_array[i, count] = evec

        else:
            raise ValueError("The variable approximation_method should be"
                             " 'optimization', 'neglect_linear_term'"
                             " or 'max_x_Px'.")

"""                                                                              
Comment                                                                          
-------                                                                                                                                  
See https://www.youtube.com/watch?v=ydxy3fEar9M for transforming an error for    
a semilog plot.                                                                  
"""
mean_error = np.mean(error_array, axis=0)
mean_log10_error = np.log10(mean_error)
relative_std_semilogplot_error = \
    np.std(error_array, axis=0) / (np.log(10) * mean_error)
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

# print(mean_error[:3], mean_upper_bound_error[:3])

fig = plt.figure(figsize=(4, 4))
ax = plt.subplot(111)
ax.scatter(N_arange, mean_error, s=5, color=deep[3],
           label="RMSE $\\mathcal{E}\,(x)$")
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
        "D": f"{D[0, 0]}I"}

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
