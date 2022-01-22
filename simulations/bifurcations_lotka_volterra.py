# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
from dynamics.dynamics import lotka_volterra
from dynamics.reduced_dynamics import reduced_lotka_volterra,\
    reduced_lotka_volterra_vector_field
from graphs.compute_tensors import compute_tensor_order_3
from singular_values.compute_effective_ranks import computeEffectiveRanks
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from scipy.linalg import pinv, svdvals
import networkx as nx
from tqdm import tqdm
from plots.config_rcparams import *
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

plot_time_series = True

""" Time parameters """
t0, t1, dt = 0, 100, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))

""" Graph parameters """
graph_str = "foodweb_little_rock"
path_str = f"C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
           f"graphs/graph_data/{graph_str}/"
G = nx.read_edgelist(path_str + "edges.csv", delimiter=',',
                     create_using=nx.DiGraph)
A = nx.to_numpy_array(G).T
# A_{ij} = 1, if edge j -> i (other convention in the data)
N = len(A[0])
A = A - A.T
# plt.matshow(A, aspect="auto")
# plt.show()

""" Dynamical parameters """
dynamics_str = "lotka_volterra"
D = 0.2*np.eye(N)
coupling_constants = np.linspace(0.1, 1, 20)

""" SVD and dimension reduction """
n = 80  # Dimension of the reduced dynamics
Un, Sn, Vhn = computeTruncatedSVD_more_positive(A, n)
L, M = Un@Sn, Vhn
print(computeEffectiveRanks(svdvals(A), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")

W = A/Sn[0][0]  # We normalize the network by the largest singular value

Mp = pinv(M)
s = np.array([np.eye(n, n)[0, :]])
m = s@M/np.sum(s@M)
ell = (s/np.sum(s@M))[0]
calD = M@D@Mp


""" Forward branch  """
x_forward_equilibrium_points_list = []
redx_forward_equilibrium_points_list = []

for coupling in tqdm(coupling_constants):

    x0 = np.random.random(N)
    redx0 = M@x0

    # Integrate complete dynamics
    args_dynamics = (coupling, D)
    x = np.array(integrate_dopri45(t0, t1, dt, lotka_volterra,
                                   W, x0, *args_dynamics))
    x_glob = np.sum(m*x, axis=1)

    # /!\ Look carefully if the dynamics reach an equilibrium point
    equilibrium_point = x_glob[-1]
    x_forward_equilibrium_points_list.append(equilibrium_point)
    # x0 = x[-1, :]

    # Integrate reduced dynamics
    WD = W - D/coupling
    calWD_tensor3 = compute_tensor_order_3(M, W)
    args_reduced_dynamics = (coupling, calD)
    redx = np.array(integrate_dopri45(t0, t1, dt, reduced_lotka_volterra,
                                      calWD_tensor3, redx0,
                                      *args_reduced_dynamics))
    # args_reduced_dynamics = (coupling, M, Mp, D)
    # redx = np.array(integrate_dopri45(t0, t1, dt,
    #                                   reduced_lotka_volterra_vector_field,
    #                                   W, redx0, *args_reduced_dynamics))
    # redx0 = redx[-1, :]

    # Get global observables
    redX_glob = np.zeros(len(redx[:, 0]))
    for nu in range(n):
        redX_nu = redx[:, nu]
        redX_glob = redX_glob + ell[nu] * redX_nu
    # we assume that the dynamics reach an equilibrium point
    red_equilibrium_point = redX_glob[-1]
    redx_forward_equilibrium_points_list.append(red_equilibrium_point)

    if plot_time_series:
        plt.figure(figsize=(4, 4))
        linewidth = 0.3
        redlinewidth = 2
        plt.subplot(111)
        for j in range(0, N):
            plt.plot(timelist, x[:, j], color=reduced_first_community_color,
                     linewidth=linewidth)
        for nu in range(n):
            plt.plot(timelist, M[nu, :]@x.T, color=first_community_color,
                     linewidth=redlinewidth)
            plt.plot(timelist, redx[:, nu], color=second_community_color,
                     linewidth=redlinewidth, linestyle="--")
        ylab = plt.ylabel('$X_{\\mu}$', labelpad=20)
        ylab.set_rotation(0)
        plt.tight_layout()
        # plt.ylim([-0.1, 1.1])
        plt.show()

# """ Backward branch """
# x_backward_equilibrium_points_list = []
# redx_backward_equilibrium_points_list = []
# x0_b = 0*np.random.random(N)
# redx0_b = M@x0_b
# for coupling in tqdm(coupling_constants[::-1]):
#
#     # Integrate complete dynamics
#     args_dynamics = (coupling, D)
#     x = np.array(integrate_dopri45(t0, t1, dt, lotka_volterra,
#                                    W, x0_b, *args_dynamics))
#     x_glob = np.sum(m*x, axis=1)
#     equilibrium_point = x_glob[-1]
#     x_backward_equilibrium_points_list.append(equilibrium_point)
#     x0_b = x[-1, :]
#
#     # Integrate reduced dynamics
#     # args_reduced_dynamics = (coupling, calD, calD_tensor3)
#     # redx = np.array(integrate_dopri45(t0, t1, dt, reduced_lotka_volterra,
#     #                                   calW_tensor3, redx0_b,
#     #                 *args_reduced_dynamics))
#     args_reduced_dynamics = (coupling, M, Mp, D)
#     redx = np.array(integrate_dopri45(t0, t1, dt, reduced_lotka_volterra,
#                                       W, redx0_b, *args_reduced_dynamics))
#     redx0_b = redx[-1, :]
#
#     # Get global observables
#     redX_glob = np.zeros(len(redx[:, 0]))
#     redX_sub_glob = np.zeros(len(redx[:, 0]))
#     for nu in range(n):
#         redX_nu = redx[:, nu]
#         redX_glob = redX_glob + ell[nu] * redX_nu
#     red_equilibrium_point = redX_glob[-1]
#     redx_backward_equilibrium_points_list.append(red_equilibrium_point)
#     red_sub_equilibrium_point = redX_sub_glob[-1]
#
#     if plot_time_series:
#
#         plt.figure(figsize=(4, 4))
#         linewidth = 0.3
#         redlinewidth = 2
#         plt.subplot(111)
#         # for j in range(0, N):
#         #    plt.plot(timelist, x[:, j], color=reduced_first_community_color,
#         #             linewidth=linewidth)
#         for nu in range(n):
#             plt.plot(timelist, M[nu, :]@x.T, color=first_community_color,
#                      linewidth=redlinewidth)
#             plt.plot(timelist, redx[:, nu], color=second_community_color,
#                      linewidth=redlinewidth, linestyle="--")
#         # plt.ylim([-0.2, 1.2])
#         # ylab = plt.ylabel('$x_i$', fontsize=fontsize, labelpad=20)
#         ylab = plt.ylabel('$X_{\\mu}$')
#         ylab.set_rotation(0)
#         # plt.ylim([-0.1, 1.1])
#         plt.tight_layout()
#         plt.show()
#
# x_backward_equilibrium_points_list.reverse()
# redx_backward_equilibrium_points_list.reverse()

fig = plt.figure(figsize=(4, 4))
redlinewidth = 2
plt.subplot(111)
plt.plot(coupling_constants, x_forward_equilibrium_points_list,
         color=first_community_color, label="Complete")
# plt.plot(coupling_constants, x_backward_equilibrium_points_list,
#          color=first_community_color)
plt.plot(coupling_constants, redx_forward_equilibrium_points_list,
         color=second_community_color, label="Reduced", linestyle="--")
# plt.plot(coupling_constants, redx_backward_equilibrium_points_list,
#          color=second_community_color, linestyle="--")
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
    path = f'C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/' \
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
              f'_complete_{dynamics_str}_graph_str_{graph_str}.json', 'w') \
            as outfile:
        json.dump(x_forward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_redx_forward_equilibrium_points_list'
              f'_reduced_{dynamics_str}_{graph_str}.json',
              'w') as outfile:
        json.dump(redx_forward_equilibrium_points_list, outfile)
    # with open(path + f'{timestr}_{file}'
    #           f'_x_backward_equilibrium_points_list'
    #           f'_complete_{dynamics_str}_{graph_str}.json', 'w') \
    #         as outfile:
    #     json.dump(x_backward_equilibrium_points_list, outfile)
    # with open(path + f'{timestr}_{file}'
    #           f'_redx_backward_equilibrium_points_list'
    #           f'_reduced_{dynamics_str}_{graph_str}.json',
    #           'w') as outfile:
    #     json.dump(redx_backward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
