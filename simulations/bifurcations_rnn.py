# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
from dynamics.dynamics import rnn
from dynamics.reduced_dynamics import reduced_rnn_vector_field
from graphs.get_real_networks import get_learned_weight_matrix
from singular_values.compute_effective_ranks import computeEffectiveRanks
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from singular_values.optimal_shrinkage import optimal_shrinkage
from plots.config_rcparams import *
from plots.plot_weight_matrix import plot_weight_matrix
from plots.plot_singular_values import plot_singular_values
from scipy.linalg import pinv, svdvals
from tqdm import tqdm
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

# /!\ This script is also useful to look at the trajectories
# (plot_time_series = True) than to get bifurcations in themselves
# when we use learned weight matrices (with negative weights), because of
# the potentially chaotic trajectories [see Sompolinsky et al., PRL, 1988]
plot_time_series = True
compute_error = False
compute_backward_branch = False
plot_weight_matrix_bool = False
plot_singvals_bool = False
save_weight_matrix_no_d_reduction_matrix = False

""" Time parameters """
t0, t1, dt = 0, 200, 0.001
# t0, t1, dt = 0, 300, 0.2
timelist = np.linspace(t0, t1, int(t1 / dt))

""" Graph parameters """
graph_str = "mouse_rnn"
A = get_learned_weight_matrix(graph_str)
U, S, Vh = np.linalg.svd(A)
shrink_s = optimal_shrinkage(S, 1, 'frobenius')
A = U@np.diag(shrink_s)@Vh
N = len(A[0])

if plot_weight_matrix_bool:
    plot_weight_matrix(A)

if plot_singvals_bool:
    plot_singular_values(svdvals(A))


""" Dynamical parameters """
dynamics_str = "rnn"
D = np.eye(N)/0.625
# tau = 0.625 in Hadjiabadi et al. Maximally selective
# single-cell target for circuit control in epilepsy models

coupling_constants = np.linspace(10, 20, 4) / 0.625
# coupling_constants = np.linspace(1.8, 2, 4) / 0.625

""" SVD and dimension reduction """
n = 47  # Dimension of the reduced dynamics
Un, Sn, Vhn = computeTruncatedSVD_more_positive(A, n)
L, M = Un@Sn, Vhn
print("\n", computeEffectiveRanks(svdvals(A), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")

W = A  # /Sn[0][0]We do not normalize the network by the largest singular value
L = L  # /Sn[0][0]

Mp = pinv(M)
s = np.array([np.eye(n, n)[0, :]])
# ^ This yields the dominant singular vector observable
m = s@M/np.sum(s@M)
ell = (s/np.sum(s@M))[0]
calD = M@D@Mp


""" Get bifurcations """
# ------ Forward branch ------
x_forward_equilibrium_points_list = []
redx_forward_equilibrium_points_list = []

x0 = np.linspace(-1, 1, N)   # 2*np.random.random(N) - 1
# x0 = -10*np.random.random(N)
redx0 = M@x0
# print("\nIterating on coupling constants for equilibrium point diagram(f)\n")
for coupling in tqdm(coupling_constants):
    
    # Integrate complete dynamics
    args_dynamics = (coupling, D)
    x = np.array(integrate_dopri45(t0, t1, dt, rnn,
                                   W, x0, *args_dynamics))
    x_glob = np.sum(m*x, axis=1)

    # /!\ Look carefully if the dynamics reach an equilibrium point
    equilibrium_point = x_glob[-1]
    x_forward_equilibrium_points_list.append(equilibrium_point)
    x0 = x[-1, :]

    # Integrate reduced dynamics
    args_reduced_dynamics = (coupling, M, Mp, D)
    redx = np.array(integrate_dopri45(t0, t1, dt, reduced_rnn_vector_field,
                                      W, redx0, *args_reduced_dynamics))
    redx0 = M@x0

    # Get global observables
    redX_glob = np.zeros(len(redx[:, 0]))
    for nu in range(n):
        redX_nu = redx[:, nu]
        redX_glob = redX_glob + ell[nu] * redX_nu
    # /!\ Look carefully if the dynamics reach an equilibrium point
    red_equilibrium_point = redX_glob[-1]
    redx_forward_equilibrium_points_list.append(red_equilibrium_point)

    if plot_time_series:
        plt.figure(figsize=(12, 5))
        linewidth = 0.3
        redlinewidth = 2
        plt.subplot(131)
        for j in range(0, N):
            plt.plot(timelist, x[:, j], color=reduced_first_community_color,
                     linewidth=linewidth)
        for nu in range(n):
            plt.plot(timelist, M[nu, :]@x.T, color=first_community_color,
                     linewidth=redlinewidth)
            plt.plot(timelist, redx[:, nu], color=second_community_color,
                     linewidth=redlinewidth, linestyle="--")
        plt.plot(timelist, M[0, :] @ x.T, color="k",
                 linewidth=redlinewidth)
        plt.plot(timelist, redx[:, 0], color=dark_grey,
                 linewidth=redlinewidth, linestyle="--")
        plt.plot(timelist, M[-1, :] @ x.T, color="r",
                 linewidth=redlinewidth)
        plt.plot(timelist, redx[:, -1], color="r",
                 linewidth=redlinewidth, linestyle="--")
        plt.xlabel("Time $t$")
        plt.ylabel("$X_{\\mu}$ for all $\\mu$")

        plt.subplot(132)
        plt.scatter(M[0, :] @ x.T, M[1, :] @ x.T, color="k",
                    linewidth=redlinewidth, s=2)
        plt.ylabel("$X_2$")                          
        plt.xlabel("$X_1$")
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])

        plt.subplot(133)
        plt.scatter(redx[:, 0], redx[:, 1], color=dark_grey,
                    linewidth=redlinewidth, linestyle="--", s=2)
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])

        plt.ylabel("$X_2$")
        plt.xlabel("$X_1$")
        plt.tight_layout()
        plt.show()

if compute_backward_branch:
    # ----- Backward branch -------
    x_backward_equilibrium_points_list = []
    redx_backward_equilibrium_points_list = []
    x0_b = 10*np.random.random(N)
    redx0_b = M @ x0_b
    print("\n Iterating on coupling constants"
          " for equilibrium point diagram(b)\n")
    for coupling in tqdm(coupling_constants[::-1]):

        # Integrate complete dynamics
        args_dynamics = (coupling, D)
        x = np.array(integrate_dopri45(t0, t1, dt, rnn,
                                       W, x0_b, *args_dynamics))
        x_glob = np.sum(m*x, axis=1)
        equilibrium_point = x_glob[-1]
        x_backward_equilibrium_points_list.append(equilibrium_point)
        x0_b = x[-1, :]

        # Integrate reduced dynamics
        args_reduced_dynamics = (coupling, M, Mp, D)
        redx = np.array(integrate_dopri45(t0, t1, dt, reduced_rnn_vector_field,
                                          W, redx0_b, *args_reduced_dynamics))
        redx0_b = M@x0_b

        # Get global observables
        redX_glob = np.zeros(len(redx[:, 0]))
        redX_sub_glob = np.zeros(len(redx[:, 0]))
        for nu in range(n):
            redX_nu = redx[:, nu]
            redX_glob = redX_glob + ell[nu]*redX_nu
        red_equilibrium_point = redX_glob[-1]
        redx_backward_equilibrium_points_list.append(red_equilibrium_point)
        red_sub_equilibrium_point = redX_sub_glob[-1]

        if plot_time_series:

            plt.figure(figsize=(4, 4))
            linewidth = 0.3
            redlinewidth = 2
            plt.subplot(111)
            # for j in range(0, N):
            #     plt.plot(timelist, x[:, j],
            #              color=reduced_first_community_color,
            #              linewidth=linewidth)
            for nu in range(n):
                plt.plot(timelist, M[nu, :] @ x.T, color=first_community_color,
                         linewidth=redlinewidth)
                plt.plot(timelist, redx[:, nu], color=second_community_color,
                         linewidth=redlinewidth, linestyle="--")
            # plt.ylim([-0.2, 1.2])
            # ylab = plt.ylabel('$x_i$', fontsize=fontsize, labelpad=20)
            ylab = plt.ylabel('$X_{\\mu}$')
            ylab.set_rotation(0)
            plt.tight_layout()
            plt.show()

    x_backward_equilibrium_points_list.reverse()
    redx_backward_equilibrium_points_list.reverse()


fig = plt.figure(figsize=(4, 4))
redlinewidth = 2
plt.subplot(111)
plt.plot(coupling_constants, x_forward_equilibrium_points_list,
         color=first_community_color, label="Complete")
plt.plot(coupling_constants, redx_forward_equilibrium_points_list,
         color=second_community_color, label="Reduced")
if compute_backward_branch:
    plt.plot(coupling_constants, x_backward_equilibrium_points_list,
             color=first_community_color)
    plt.plot(coupling_constants, redx_backward_equilibrium_points_list,
             color=second_community_color)
ylab = plt.ylabel('Global activity equilibrium point $X^*$')
plt.xlabel('Coupling constant')
plt.tick_params(axis='both', which='major')
plt.legend(loc=4, fontsize=fontsize_legend)
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
           f'simulations/simulations_data/{dynamics_str}_data/'
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    parameters_dictionary = {"graph_str": graph_str,
                             "D": D.tolist(), "n": n, "N": N,
                             "t0": t0, "t1": t1, "dt": dt,
                             "coupling_constants": coupling_constants.tolist()}

    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
                f'_{dynamics_str}_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
                       f'_{dynamics_str}_{graph_str}.png')
    with open(path + f'{timestr}_{file}'
              f'_x_forward_equilibrium_points_list'
              f'_complete_{dynamics_str}_{graph_str}.json', 'w') \
            as outfile:
        json.dump(x_forward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_redx_forward_equilibrium_points_list'
              f'_reduced_{dynamics_str}_{graph_str}.json',
              'w') as outfile:
        json.dump(redx_forward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_x_backward_equilibrium_points_list'
              f'_complete_{dynamics_str}_{graph_str}.json', 'w') \
            as outfile:
        json.dump(x_backward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_redx_backward_equilibrium_points_list'
              f'_reduced_{dynamics_str}_{graph_str}.json',
              'w') as outfile:
        json.dump(redx_backward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
