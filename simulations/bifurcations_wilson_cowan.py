# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
from dynamics.dynamics import wilson_cowan
from dynamics.reduced_dynamics import reduced_wilson_cowan
from graphs.get_real_networks import get_connectome_weight_matrix
from singular_values.compute_effective_ranks import computeEffectiveRanks
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from plots.config_rcparams import *
from plots.plot_weight_matrix import plot_weight_matrix
from plots.plot_singular_values import plot_singular_values
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
plot_singvals_bool = False
save_weight_matrix_no_d_reduction_matrix = False

""" Time parameters """
t0, t1, dt = 0, 250, 0.2
timelist = np.linspace(t0, t1, int(t1 / dt))

""" Graph parameters """
graph_str = "celegans_signed"

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
    A = get_connectome_weight_matrix(graph_str)
    N = len(A[0])

if plot_weight_matrix_bool:
    plot_weight_matrix(A)

if plot_singvals_bool:
    plot_singular_values(svdvals(A))


""" Dynamical parameters """
dynamics_str = "wilson_cowan"
D = np.eye(N)
# a = 0.1
# b = 1
# c = 3
a = 1
b = 1
c = -3

coupling_constants = np.linspace(0.01, 0.5, 20)  # c. elegans signed
# coupling_constants = np.linspace(12, 15, 50)  # ciona weighted

# Notes when W is not normalized by the largest singular value
# coupling_constants = np.linspace(0.35, 0.9, 10)  # celegans,ciona,zebrafish
# coupling_constants = np.linspace(0.5, 8.5, 100)# mouse_meso (not connex ?)
# coupling_constants = np.linspace(0.6, 1.2, 100)# mouse_meso hysteresis 1
# coupling_constants = np.linspace(5, 10, 100)   # mouse_meso hysteresis 2


""" SVD and dimension reduction """
n = 5  # Dimension of the reduced dynamics
Un, Sn, Vhn = computeTruncatedSVD_more_positive(A, n)
L, M = Un@Sn, Vhn
print("\n", computeEffectiveRanks(svdvals(A), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")

W = A  # /Sn[0][0]  # We normalize the network by the largest singular value
L = L  # /Sn[0][0]

Mp = pinv(M)
s = np.array([np.eye(n, n)[0, :]])
# ^ This yields the dominant singular vector observable
m = s @ M / np.sum(s @ M)
ell = (s / np.sum(s @ M))[0]
calD = M@D@Mp
timestr_M_D = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")


""" Get bifurcations """
# ------ Forward branch ------
x_forward_equilibrium_points_list = []
redx_forward_equilibrium_points_list = []

# x0 = -10*np.random.random(N)
# redx0 = M @ x0
print("\n Iterating on coupling constants for equilibrium point diagram(f) \n")
for coupling in tqdm(coupling_constants):

    x0 = np.random.random(N)
    redx0 = M @ x0
    
    # Integrate complete dynamics
    args_dynamics = (coupling, D, a, b, c)
    x = np.array(integrate_dopri45(t0, t1, dt, wilson_cowan,
                                   W, x0, *args_dynamics))
    x_glob = np.sum(m*x, axis=1)

    # /!\ Look carefully if the dynamics reach an equilibrium point
    equilibrium_point = x_glob[-1]
    x_forward_equilibrium_points_list.append(equilibrium_point)
    # x0 = x[-1, :]

    # Integrate reduced dynamics
    args_reduced_dynamics = (M, coupling, calD, a, b, c)
    redx = np.array(integrate_dopri45(t0, t1, dt, reduced_wilson_cowan,
                                      L, redx0, *args_reduced_dynamics))
    # args_reduced_dynamics = (coupling, M, Mp, D, a, b, c)
    # redx = np.array(integrate_dopri45(t0, t1, dt, reduced_wilson_cowan,
    #                                   W, redx0, *args_reduced_dynamics))
    # redx0 = redx[-1, :]

    # Get global observables
    redX_glob = np.zeros(len(redx[:, 0]))
    for nu in range(n):
        redX_nu = redx[:, nu]
        redX_glob = redX_glob + ell[nu] * redX_nu
    # /!\ Look carefully if the dynamics reach an equilibrium point
    red_equilibrium_point = redX_glob[-1]
    redx_forward_equilibrium_points_list.append(red_equilibrium_point)

    if plot_time_series:
        plt.figure(figsize=(4, 4))
        linewidth = 0.3
        redlinewidth = 2
        plt.subplot(111)
        # for j in range(0, N):
        #     plt.plot(timelist, x[:, j], color=reduced_first_community_color,
        #              linewidth=linewidth)
        for nu in range(n):
            plt.plot(timelist, M[nu, :]@x.T, color=first_community_color,
                     linewidth=redlinewidth)
            plt.plot(timelist, redx[:, nu], color=second_community_color,
                     linewidth=redlinewidth, linestyle="--")
        ylab = plt.ylabel('$X_{\\mu}$', labelpad=20)
        ylab.set_rotation(0)
        plt.tight_layout()
        plt.show()

# ----- Backward branch -------
x_backward_equilibrium_points_list = []
redx_backward_equilibrium_points_list = []
x0_b = 10 * np.random.random(N)
redx0_b = M @ x0_b
print("\n Iterating on coupling constants for equilibrium point diagram(b)\n")
for coupling in tqdm(coupling_constants[::-1]):

    # Integrate complete dynamics
    args_dynamics = (coupling, D, a, b, c)
    x = np.array(integrate_dopri45(t0, t1, dt, wilson_cowan,
                                   W, x0_b, *args_dynamics))
    x_glob = np.sum(m * x, axis=1)
    equilibrium_point = x_glob[-1]
    x_backward_equilibrium_points_list.append(equilibrium_point)
    x0_b = x[-1, :]

    # Integrate reduced dynamics
    args_reduced_dynamics = (M, coupling, calD, a, b, c)
    redx = np.array(integrate_dopri45(t0, t1, dt, reduced_wilson_cowan,
                                      L, redx0_b, *args_reduced_dynamics))
    # args_reduced_dynamics = (coupling, M, Mp, D, a, b, c)
    # redx = np.array(integrate_dopri45(t0, t1, dt, reduced_wilson_cowan,
    #                                   W, redx0_b, *args_reduced_dynamics))
    redx0_b = redx[-1, :]

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
        #     plt.plot(timelist, x[:, j], color=reduced_first_community_color,
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
plt.plot(coupling_constants, x_backward_equilibrium_points_list,
         color=first_community_color)
plt.plot(coupling_constants, redx_forward_equilibrium_points_list,
         color=second_community_color, label="Reduced")
plt.plot(coupling_constants, redx_backward_equilibrium_points_list,
         color=second_community_color)
ylab = plt.ylabel('Global activity equilibrium point $X^*$')
plt.xlabel('Coupling constant')
plt.ylim([-0.02, 1.02])
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

    parameters_dictionary = {"graph_str": graph_str, "a": a, "b": b, "c": c,
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
