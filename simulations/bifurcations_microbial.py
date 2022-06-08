# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
from dynamics.dynamics import microbial
from dynamics.reduced_dynamics import reduced_microbial,\
    reduced_microbial_vector_field
from graphs.get_real_networks import get_microbiome_weight_matrix
from graphs.compute_tensors import compute_tensor_order_3, \
    compute_tensor_order_4
from singular_values.compute_effective_ranks import computeEffectiveRanks
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from plots.config_rcparams import *
from plots.plot_weight_matrix import plot_weight_matrix
from scipy.linalg import pinv, svdvals
import networkx as nx
from tqdm import tqdm
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

plot_time_series = True
compute_error = False
plot_weight_matrix_bool = False
integrate_complete_dynamics = True
integrate_reduced_dynamics = False
integrate_reduced_dynamics_tensor = False
forward_branch = True
backward_branch = False

# Desmos exploration: https://www.desmos.com/calculator/pdtj9p2mdi

""" Time parameters """
# Three setup for for "gut" network
# 1.
# t0, t1, dt = 0, 4, 0.0001 ou 0, 1.5, 0.0001
# 2.
t0, t1, dt = 0, 3000, 0.5
# 3.
# t0, t1, dt = 0, 6, 0.001

timelist = np.linspace(t0, t1, int(t1 / dt))

""" Graph parameters """
graph_str = "gut"

if graph_str == "SBM":
    n1 = 60
    n2 = 40
    sizes = [n1, n2]
    N = sum(sizes)
    p11, p22 = 0.4, 0.4
    pout = 0.2
    pq = [[p11, pout], [pout, p22]]

    random_weights = np.random.random((N, N))
    random_symmetric_weights = (random_weights + random_weights.T) / 2
    binaryD = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
    weight_str = "binary"  # "random_symmetric"

    if weight_str == "random":
        A = binaryD * random_weights
    elif weight_str == "random_symmetric":
        A = binaryD * random_symmetric_weights
    elif weight_str == "binary":
        A = binaryD
    else:
        print("WARNING: this weight_str is not among the available choice.")

else:
    A = get_microbiome_weight_matrix(graph_str)
    N = len(A[0])
    # B = np.where(A > 0, A, 0)
    # print(np.sum(B, axis=1))

if plot_weight_matrix_bool:
    plot_weight_matrix(A)


""" Dynamical parameters """
dynamics_str = "microbial"
# For real gut microbiome network
# 1.
# a, b, c, D = 5, 13, 10/3, 30*np.eye(N)
# coupling_constants =  np.linspace(0.5, 3, 50)

# 2.
a, b, c, D = 0.00005, 0.1, 0.9, 0.01*np.eye(N)
coupling_constants = np.linspace(1.5, 4.5, 10)

# 3.
# a, b, c, D = 0, 0.15, 72, 0.0002*np.eye(N)
# coupling_constants = np.linspace(25, 40, 5)

# a, b, c, D = 0.00005, 0.01, 0.1, 0.9*np.eye(N)
# coupling_constants = np.linspace(5, 50, 5)
# a, b, c, D = 1/4, 13/3, 1/3, 10*np.eye(N)


""" SVD and dimension reduction """
n = 25  # Dimension of the reduced dynamics
Un, Sn, M = computeTruncatedSVD_more_positive(A, n)
print("\n", computeEffectiveRanks(svdvals(A), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")

# 1.
# W = A

# 2. and 3.
W = A/Sn[0][0]  # We normalize the network by the largest singular value

Mp = pinv(M)
s = np.array([np.eye(n, n)[0, :]])
# ^ This yields the dominant singular vector observable
m = -s@M/np.sum(s@M)  # /!\ the minus sign is very particular here
ell = -(s/np.sum(s@M))[0]

if integrate_reduced_dynamics_tensor:
    calD = M@D@np.linalg.pinv(M)
    calM = np.sum(M, axis=1)
    calW_tensor3 = compute_tensor_order_3(M, W)
    calM_tensor3 = compute_tensor_order_3(M, np.eye(N))
    calM_tensor4 = compute_tensor_order_4(M, np.eye(N))

timestr_M_D = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")


""" Get bifurcations """

if forward_branch:
    # ------ Forward branch ------
    x_forward_equilibrium_points_list = []
    redx_forward_equilibrium_points_list = []

    x0 = np.random.random(N)
    redx0 = M@x0

    print("\n Iterating on coupling constants"
          " for equilibrium point diagram(f) \n")
    for coupling in tqdm(coupling_constants):

        if integrate_complete_dynamics:
            """ Integrate complete dynamics """
            args_dynamics = (coupling, D, a, b, c)
            x = np.array(integrate_dopri45(t0, t1, dt, microbial,
                                           W, x0, *args_dynamics))
            x_glob = np.sum(m*x, axis=1)

            """ /!\ Look carefully if the dynamics reach an eq. point """
            equilibrium_point = x_glob[-1]
            x_forward_equilibrium_points_list.append(equilibrium_point)
            # x0 = x[-1, :]

        """ Integrate reduced dynamics """
        if integrate_reduced_dynamics:
            if integrate_reduced_dynamics_tensor:
                args_reduced_dynamics = (coupling, calD, calM, calM_tensor3,
                                         calM_tensor4, a, b, c)
                redx = np.array(integrate_dopri45(t0, t1, dt,
                                                  reduced_microbial,
                                                  calW_tensor3, redx0,
                                                  *args_reduced_dynamics))
            else:
                args_reduced_dynamics = (coupling, M, Mp, D, a, b, c)
                redx = np.array(
                    integrate_dopri45(t0, t1, dt,
                                      reduced_microbial_vector_field,
                                      W, redx0, *args_reduced_dynamics))
            redx0 = redx[-1, :]

            """ Get global observables """
            redX_glob = np.zeros(len(redx[:, 0]))
            for nu in range(n):
                redX_nu = redx[:, nu]
                redX_glob = redX_glob + ell[nu]*redX_nu
            # /!\ Look carefully if the dynamics reach an equilibrium point
            red_equilibrium_point = redX_glob[-1]
            redx_forward_equilibrium_points_list.append(red_equilibrium_point)

        if plot_time_series:
            plt.figure(figsize=(4, 4))
            linewidth = 0.3
            redlinewidth = 2
            plt.subplot(111)
            if integrate_complete_dynamics:
                for j in range(0, N):
                    plt.plot(timelist, x[:, j],
                             color=reduced_first_community_color,
                             linewidth=linewidth)
            if integrate_reduced_dynamics or integrate_reduced_dynamics_tensor:
                for nu in range(n):
                    # f integrate_complete_dynamics:
                    plt.plot(timelist, M[nu, :]@x.T,
                             color=first_community_color,
                             linewidth=redlinewidth)
                    # if integrate_reduced_dynamics:
                    plt.plot(timelist, redx[:, nu],
                             color=second_community_color,
                             linewidth=redlinewidth, linestyle="--")
            # if integrate_complete_dynamics:
            #     plt.plot(timelist, x_glob,
            #              linewidth=redlinewidth, color="r")
            # if integrate_reduced_dynamics or
            #  integrate_reduced_dynamics_tensor:
            #     plt.plot(timelist, redX_glob,
            #              linewidth=redlinewidth, color="g")
            ylab = plt.ylabel('$X_{\\mu}$', labelpad=20)
            ylab.set_rotation(0)
            plt.tight_layout()
            plt.show()

if backward_branch:
    # ----- Backward branch -------
    x_backward_equilibrium_points_list = []
    redx_backward_equilibrium_points_list = []
    x0_b = 0.1*np.random.random(N)
    redx0_b = M@x0_b
    print("\n Iterating on coupling constants"
          " for equilibrium point diagram(b)\n")
    for coupling in tqdm(coupling_constants[::-1]):

        if integrate_complete_dynamics:
            """ Integrate complete dynamics """
            args_dynamics = (coupling, D, a, b, c)
            x = np.array(integrate_dopri45(t0, t1, dt, microbial,
                                           W, x0_b, *args_dynamics))
            x_glob = np.sum(m*x, axis=1)
            equilibrium_point = x_glob[-1]
            x_backward_equilibrium_points_list.append(equilibrium_point)
            x0_b = x[-1, :]

        """ Integrate reduced dynamics """
        if integrate_reduced_dynamics:
            if integrate_reduced_dynamics_tensor:
                args_reduced_dynamics = (coupling, calD, calM, calM_tensor3,
                                         calM_tensor4, a, b, c)
                redx = np.array(
                    integrate_dopri45(t0, t1, dt, reduced_microbial,
                                      calW_tensor3, redx0_b,
                                      *args_reduced_dynamics))
            else:
                args_reduced_dynamics = (coupling, M, Mp, D, a, b, c)
                redx = np.array(
                    integrate_dopri45(t0, t1, dt,
                                      reduced_microbial_vector_field,
                                      W, redx0_b, *args_reduced_dynamics))
            redx0_b = redx[-1, :]

            """ Get global observables """
            redX_glob = np.zeros(len(redx[:, 0]))
            redX_sub_glob = np.zeros(len(redx[:, 0]))
            for nu in range(n):
                redX_nu = redx[:, nu]
                redX_glob = redX_glob + ell[nu]*redX_nu
            red_equilibrium_point = redX_glob[-1]
            redx_backward_equilibrium_points_list.append(red_equilibrium_point)

        if plot_time_series:

            plt.figure(figsize=(4, 4))
            linewidth = 0.3
            redlinewidth = 2
            plt.subplot(111)
            if integrate_complete_dynamics:
                for j in range(0, N):
                    plt.plot(timelist, x[:, j],
                             color=reduced_first_community_color,
                             linewidth=linewidth)
            if integrate_reduced_dynamics or integrate_reduced_dynamics_tensor:
                for nu in range(n):
                    # f integrate_complete_dynamics:
                    plt.plot(timelist, M[nu, :] @ x.T,
                             color=first_community_color,
                             linewidth=redlinewidth)
                    # if integrate_reduced_dynamics:
                    plt.plot(timelist, redx[:, nu],
                             color=second_community_color,
                             linewidth=redlinewidth, linestyle="--")
            if integrate_complete_dynamics:
                plt.plot(timelist, x_glob,
                         linewidth=redlinewidth, color="r")
            if integrate_reduced_dynamics or integrate_reduced_dynamics_tensor:
                plt.plot(timelist, redX_glob,
                         linewidth=redlinewidth, color="g")
            ylab = plt.ylabel('$X_{\\mu}$', labelpad=20)
            ylab.set_rotation(0)
            plt.tight_layout()
            plt.show()

    x_backward_equilibrium_points_list.reverse()
    redx_backward_equilibrium_points_list.reverse()

fig = plt.figure(figsize=(4, 4))
redlinewidth = 2
plt.subplot(111)
if integrate_complete_dynamics:
    if forward_branch:
        plt.plot(coupling_constants, x_forward_equilibrium_points_list,
                 color=first_community_color, label="Complete")
    if backward_branch:
        plt.plot(coupling_constants, x_backward_equilibrium_points_list,
                 color=first_community_color)
if integrate_reduced_dynamics:
    if forward_branch:
        plt.plot(coupling_constants, redx_forward_equilibrium_points_list,
                 color=second_community_color, label="Reduced")
    if backward_branch:
        plt.plot(coupling_constants, redx_backward_equilibrium_points_list,
                 color=second_community_color)
ylab = plt.ylabel('Global activity equilibrium point $X^*$')
plt.xlabel('Coupling constant')
# plt.ylim([-0.02, 1.02])
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

    parameters_dictionary = {"graph_str": graph_str, "a": a, "b": b, "c": c,
                             "D": D.tolist(), "n": n, "N": N,
                             "t0": t0, "t1": t1, "dt": dt,
                             "coupling_constants": coupling_constants.tolist()}

    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
                f'_{dynamics_str}_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
                       f'_{dynamics_str}_{graph_str}.png')
    if integrate_complete_dynamics:
        with open(path + f'{timestr}_{file}'
                  f'_x_forward_equilibrium_points_list'
                  f'_complete_{dynamics_str}_{graph_str}.json', 'w')\
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
