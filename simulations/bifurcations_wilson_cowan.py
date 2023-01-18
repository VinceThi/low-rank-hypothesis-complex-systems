# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
from dynamics.error_vector_fields import \
    jacobian_x_wilson_cowan, jacobian_y_wilson_cowan
from scipy.integrate import solve_ivp
from dynamics.dynamics import wilson_cowan
from dynamics.reduced_dynamics import reduced_wilson_cowan
from graphs.get_real_networks import get_connectome_weight_matrix
from singular_values.compute_effective_ranks import computeEffectiveRanks
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from plots.config_rcparams import *
from plots.plot_weight_matrix import plot_weight_matrix
from plots.plot_singular_values import plot_singular_values
from scipy.linalg import pinv, svdvals
from tqdm import tqdm
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

plot_time_series = False
compute_error = False
plot_weight_matrix_bool = False
plot_singvals_bool = False
save_weight_matrix_no_d_reduction_matrix = False

""" Time and integration parameters """
t0, t1 = 0, 250
t_span = [t0, t1]
integration_method = 'BDF'
rtol = 1e-8
atol = 1e-12


def jacobian_complete(t, x, W, coupling, D, a, b, c):
    Jx = jacobian_x_wilson_cowan(x, W, coupling, D, a, b, c)
    Jy = jacobian_y_wilson_cowan(x, W, coupling, a, b, c)
    return Jx + Jy@W


def jacobian_reduced(t, X, W, coupling, M, Mp, D, a, b, c):
    return M@jacobian_complete(t, Mp@X, W, coupling, D, a, b, c)@Mp


""" Graph parameters """
graph_str = "celegans_signed"

W = get_connectome_weight_matrix(graph_str)
N = len(W[0])

if plot_weight_matrix_bool:
    plot_weight_matrix(W)

if plot_singvals_bool:
    plot_singular_values(svdvals(W))

""" Dynamical parameters """
dynamics_str = "wilson_cowan"
D = np.eye(N)
a = 0.05
b = 1
c = 3
# Coupling constants for graph_str = "celegans_signed"
coupling_constants = np.linspace(0.01, 0.14, 2000)


""" SVD and dimension reduction """
n = 80  # Dimension of the reduced dynamics
Un, Sn, Vhn = computeTruncatedSVD_more_positive(W, n)
L, M = Un@Sn, Vhn
print("\n", computeEffectiveRanks(svdvals(W), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")

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

x0 = -10*np.random.random(N)
redx0 = M @ x0
print("\n Iterating on coupling constants for equilibrium point diagram(f) \n")
for coupling in tqdm(coupling_constants):
    
    # Integrate complete dynamics
    args_dynamics = (W, coupling, D, a, b, c)
    sol = solve_ivp(wilson_cowan, t_span, x0, integration_method,
                    args=args_dynamics, rtol=rtol, atol=atol,
                    vectorized=True, jac=jacobian_complete)
    x = sol.y.T
    tc = sol.t
    x_glob = np.sum(m*x, axis=1)

    equilibrium_point = x_glob[-1]
    x_forward_equilibrium_points_list.append(equilibrium_point)
    x0 = x[-1, :]

    # Integrate reduced dynamics
    args_reduced_dynamics = (L, M, coupling, calD, a, b, c)
    sol = solve_ivp(reduced_wilson_cowan, t_span, redx0,
                    integration_method, args=args_reduced_dynamics,
                    rtol=rtol, atol=atol, vectorized=True)
    redx = sol.y.T
    tr = sol.t
    redx0 = redx[-1, :]

    # Get global observables
    redX_glob = np.zeros(len(redx[:, 0]))
    for nu in range(n):
        redX_nu = redx[:, nu]
        redX_glob = redX_glob + ell[nu] * redX_nu
    red_equilibrium_point = redX_glob[-1]
    redx_forward_equilibrium_points_list.append(red_equilibrium_point)

    if plot_time_series:
        plt.figure(figsize=(4, 4))
        linewidth = 0.3
        redlinewidth = 2
        plt.subplot(111)
        # for j in range(0, N):
        #     plt.plot(tc, x[:, j], color=reduced_first_community_color,
        #              linewidth=linewidth)
        for nu in range(n):
            plt.scatter(tc, M[nu, :]@x.T, color=first_community_color, s=2)
            plt.scatter(tr, redx[:, nu], color=second_community_color, s=2)
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
    args_dynamics = (W, coupling, D, a, b, c)
    sol = solve_ivp(wilson_cowan, t_span, x0_b, integration_method,
                    args=args_dynamics, rtol=rtol, atol=atol, vectorized=True,
                    jac=jacobian_complete)
    x = sol.y.T
    tc = sol.t
    x_glob = np.sum(m*x, axis=1)
    equilibrium_point = x_glob[-1]
    x_backward_equilibrium_points_list.append(equilibrium_point)
    x0_b = x[-1, :]

    # Integrate reduced dynamics
    args_reduced_dynamics = (L, M, coupling, calD, a, b, c)
    sol = solve_ivp(reduced_wilson_cowan, t_span, redx0_b,
                    integration_method, args=args_reduced_dynamics,
                    rtol=rtol, atol=atol, vectorized=True)
    redx = sol.y.T
    tr = sol.t
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
        #     plt.plot(tc, x[:, j], color=reduced_first_community_color,
        #              linewidth=linewidth)
        for nu in range(n):
            plt.scatter(tc, M[nu, :]@x.T, color=first_community_color, s=2)
            plt.scatter(tr, redx[:, nu], color=second_community_color, s=2)
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
plt.ylim([-0.02, 0.35])
plt.tick_params(axis='both', which='major')
plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters,"
                       " the data, and the plot?"):
    window = tkinter.Tk()
    window.withdraw()
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    path = f'C:/Users/thivi/Documents/GitHub/' \
           f'low-rank-hypothesis-complex-systems/' \
           f'simulations/simulations_data/{dynamics_str}_data/'
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    parameters_dictionary = {"graph_str": graph_str, "a": a, "b": b, "c": c,
                             "D": D.tolist(), "n": n, "N": N,
                             "t0": t0, "t1": t1,
                             "integration_method": integration_method,
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
