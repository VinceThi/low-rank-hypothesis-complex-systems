# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
from scipy.integrate import solve_ivp
from dynamics.error_vector_fields import jacobian_x_microbial,\
    jacobian_y_microbial
from dynamics.dynamics import microbial
from dynamics.reduced_dynamics import reduced_microbial_vector_field
from graphs.get_real_networks import get_microbiome_weight_matrix
from singular_values.compute_effective_ranks import computeEffectiveRanks
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from plots.config_rcparams import *
from plots.plot_weight_matrix import plot_weight_matrix
from scipy.linalg import pinv, svdvals
from tqdm import tqdm
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

plot_time_series = False
plot_weight_matrix_bool = False
integrate_complete_dynamics = True
integrate_reduced_dynamics = True
forward_branch = True
backward_branch = True
uniform_observable = False

# Desmos exploration: https://www.desmos.com/calculator/pdtj9p2mdi

""" Time and integration parameters """
t0, t1 = 0, 50
t_span = [t0, t1]
t_span_red = [t0, t1]
integration_method = 'BDF'
rtol = 1e-8
atol = 1e-12


def jacobian_complete(t, x, W, coupling, D, a, b, c):
    Jx = jacobian_x_microbial(x, W, coupling, D, b, c)
    Jy = jacobian_y_microbial(x, coupling)
    return Jx + Jy@W


def jacobian_reduced(t, X, W, coupling, M, Mp, D, a, b, c):
    return M@jacobian_complete(t, Mp@X, W, coupling, D, a, b, c)@Mp


""" 
Note: The ODE is very stiff and BDF is particularly well suited for 
      stiff problems. We also provide the jacobian matrices as recommanded
      in the documentation of 'solve_ivp': 
      https://docs.scipy.org/doc/scipy/reference/generated/
      scipy.integrate.solve_ivp.html

      In the reference 
      --- P. Städter, Y. Schälte, L. Schmiester, J. Hasenauer & P. L. Stapor,
      'Benchmarking of numerical integration methods for ODE models of
      biological systems', Scientific Reports, 11, 2696 (2021) ---
      one will find many arguments to use BDF, such as better integration time
      and failure rate. Also, rtol and atol need to be carefully chosen. Our 
      numerical simulations seem to corroborate their results.

"""


def plot_time_series_observables(x, x_glob, tr, redx, tc, M):
    plt.figure(figsize=(4, 4))
    linewidth = 0.3
    redlinewidth = 2
    if integrate_complete_dynamics:
        for j in range(0, N):
            plt.plot(tc, x[:, j],
                     color=reduced_first_community_color,
                     linewidth=linewidth)
        for nu in range(n):
            plt.scatter(tc, M[nu, :]@x.T,
                        color=first_community_color, s=2)
            plt.plot(tc, x_glob,
                     linewidth=redlinewidth, color=deep[2])
    if integrate_reduced_dynamics:
        for nu in range(n):
            plt.scatter(tr, redx[:, nu],
                        color=second_community_color,
                        s=2, linestyle="--")
        plt.plot(tr, redX_glob,
                 linewidth=redlinewidth, color=deep[3])
    ylab = plt.ylabel('$X_{\\mu}$', labelpad=20)
    ylab.set_rotation(0)
    plt.xlabel("Time $t$")
    plt.tight_layout()


""" Graph parameters """
graph_str = "gut"
W = get_microbiome_weight_matrix(graph_str)
N = len(W[0])

if plot_weight_matrix_bool:
    plot_weight_matrix(W)


""" Dynamical parameters """
dynamics_str = "microbial"
# For real gut microbiome network
a, b, c, D = 5, 13, 1, 30*np.eye(N)
coupling_constants_forward = np.linspace(0.05, 3, 50)
coupling_constants_backward = np.linspace(0.05, 3, 50)


""" SVD and dimension reduction """
n = 76  # 76, 203, 400  # Dimension of the reduced dynamics
Un, Sn, M = computeTruncatedSVD_more_positive(W, n)
print("\n", computeEffectiveRanks(svdvals(W), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")
Mp = pinv(M)
# We combine all observable weighted by the singular values
# to get a global observable
s = np.diag(Sn)
normalization_constant = np.sum(s@M)*20
m = s@M/normalization_constant
ell = s/normalization_constant

norm_cte_uniform = 20

timestr_M_D = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")


""" Get bifurcations """

if forward_branch:
    x_forward_equilibrium_points_list = []
    x_forward_uniform_eq_pts_list = []
    redx_forward_equilibrium_points_list = []
    x0 = np.random.random(N)
    redx0 = M@x0
    print("\n Iterating on coupling constants"
          " for equilibrium point diagram(f) \n")
    for coupling in tqdm(coupling_constants_forward):

        if integrate_complete_dynamics:
            args_dynamics = (W, coupling, D, a, b, c)
            sol = solve_ivp(microbial, t_span, x0, integration_method,
                            args=args_dynamics, rtol=rtol, atol=atol,
                            vectorized=True, jac=jacobian_complete)
            x = sol.y.T
            tc = sol.t

            """ Verify if the equilibrium point is reached """
            eval_f = microbial(0, x[-1, :], W, coupling, D, a, b, c)
            if not np.allclose(eval_f, np.zeros(N)):
                raise ValueError("An equilibrium point is not reached for the "
                                 "complete dynamics.")

            """ Get global observables """
            x_glob = np.sum(m*x, axis=1)
            equilibrium_point = x_glob[-1]
            x_forward_equilibrium_points_list.append(equilibrium_point)

            """ Reset initial condition at the last equilibrium point """
            x0 = x[-1, :]

            if uniform_observable:
                x_glob_uni = np.sum(x/N, axis=1)/norm_cte_uniform
                x_forward_uniform_eq_pts_list.append(x_glob_uni[-1])

        if integrate_reduced_dynamics:
            args_reduced_dynamics = (W, coupling, M, Mp, D, a, b, c)
            sol = solve_ivp(reduced_microbial_vector_field, t_span, redx0,
                            integration_method, args=args_reduced_dynamics,
                            rtol=rtol, atol=atol, vectorized=True,
                            jac=jacobian_reduced)
            redx = sol.y.T
            tr = sol.t

            """ Verify if the equilibrium point is reached """
            eval_redf = reduced_microbial_vector_field(0, redx[-1, :], W,
                                                       coupling, M, Mp,
                                                       D, a, b, c)
            if not np.allclose(eval_redf, np.zeros(n)):
                raise ValueError("An equilibrium point is not reached for the "
                                 "reduced dynamics.")

            """ Get global observables """
            redX_glob = np.zeros(len(redx[:, 0]))
            for nu in range(n):
                redX_nu = redx[:, nu]
                redX_glob = redX_glob + ell[nu]*redX_nu
            red_equilibrium_point = redX_glob[-1]
            redx_forward_equilibrium_points_list.append(
                red_equilibrium_point)

            """ Reset initial condition at the last equilibrium point """
            redx0 = redx[-1, :]

        if plot_time_series:
            plot_time_series_observables(x, x_glob, tr, redx, tc, M)
            if uniform_observable:
                plt.plot(tc, x_glob_uni, color="k")
            plt.show()

if backward_branch:
    x_backward_equilibrium_points_list = []
    x_backward_uniform_eq_pts_list = []
    redx_backward_equilibrium_points_list = []
    if forward_branch:
        x0_b = x[-1, :]
        redx0_b = M@x[-1, :]
    else:
        x0_b = 0.1*np.random.random(N)
        redx0_b = M@x0_b
    print("\n Iterating on coupling constants"
          " for equilibrium point diagram(b)\n")
    for coupling in tqdm(coupling_constants_backward[::-1]):

        if integrate_complete_dynamics:
            args_dynamics = (W, coupling, D, a, b, c)
            sol = solve_ivp(microbial, t_span, x0_b, integration_method,
                            args=args_dynamics, rtol=rtol, atol=atol,
                            vectorized=True, jac=jacobian_complete)
            x = sol.y.T
            tc = sol.t

            """ Verify if the equilibrium point is reached """
            eval_f = microbial(0, x[-1, :], W, coupling, D, a, b, c)
            if not np.allclose(eval_f, np.zeros(N)):
                raise ValueError("An equilibrium point is not reached for the "
                                 "complete dynamics.")

            """ Get global observables """
            x_glob = np.sum(m*x, axis=1)
            equilibrium_point = x_glob[-1]
            x_backward_equilibrium_points_list.append(equilibrium_point)

            """ Reset initial condition at the last equilibrium point """
            x0_b = x[-1, :]

            if uniform_observable:
                x_glob_uni = np.sum(x/N, axis=1)
                x_backward_uniform_eq_pts_list.append(x_glob_uni[-1])

        if integrate_reduced_dynamics:
            args_reduced_dynamics = (coupling, M, Mp, D, a, b, c)
            args_reduced_dynamics = (W, coupling, M, Mp, D, a, b, c)
            sol = solve_ivp(reduced_microbial_vector_field, t_span,
                            redx0_b, integration_method,
                            args=args_reduced_dynamics,
                            rtol=rtol, atol=atol, vectorized=True,
                            jac=jacobian_reduced)
            redx = sol.y.T
            tr = sol.t

            """ Verify if the equilibrium point is reached """
            eval_redf = reduced_microbial_vector_field(0, redx[-1, :], W,
                                                       coupling, M, Mp,
                                                       D, a, b, c)
            if not np.allclose(eval_redf, np.zeros(n)):
                raise ValueError("An equilibrium point is not reached for the "
                                 "reduced dynamics.")

            """ Get global observables """
            redX_glob = np.zeros(len(redx[:, 0]))
            redX_sub_glob = np.zeros(len(redx[:, 0]))
            for nu in range(n):
                redX_nu = redx[:, nu]
                redX_glob = redX_glob + ell[nu]*redX_nu
            red_equilibrium_point = redX_glob[-1]
            redx_backward_equilibrium_points_list.append(
                red_equilibrium_point)

            """ Reset initial condition at the last equilibrium points"""
            redx0_b = redx[-1, :]

        if plot_time_series:
            plot_time_series_observables(x, x_glob, tr, redx, tc, M)
            if uniform_observable:
                plt.plot(tc, x_glob_uni, color="k")
            plt.show()

    x_backward_equilibrium_points_list.reverse()
    if uniform_observable:
        x_backward_uniform_eq_pts_list.reverse()
    redx_backward_equilibrium_points_list.reverse()

if uniform_observable:
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(coupling_constants_forward, x_forward_uniform_eq_pts_list,
                color=deep[3], s=2)
    plt.scatter(coupling_constants_backward, x_backward_uniform_eq_pts_list,
                color=deep[3], s=2)
    ylab = plt.ylabel('Global (uniform) equilibrium point $X^*$')
    plt.xlabel('Global microbial\n interaction weight')
    plt.tick_params(axis='both', which='major')
    plt.tight_layout()
    plt.show()

fig = plt.figure(figsize=(4, 4))
redlinewidth = 2
norm_cte = 15
plt.subplot(111)
if integrate_complete_dynamics:
    if forward_branch:
        plt.scatter(coupling_constants_forward,
                    x_forward_equilibrium_points_list,
                    color=deep[0], label="Complete", s=2)
    if backward_branch:
        plt.scatter(coupling_constants_backward,
                    x_backward_equilibrium_points_list,
                    color=deep[2], s=2)
if integrate_reduced_dynamics:
    if forward_branch:
        plt.scatter(coupling_constants_forward,
                    redx_forward_equilibrium_points_list,
                    color=deep[1], label="Reduced", s=2)
    if backward_branch:
        plt.scatter(coupling_constants_backward,
                    redx_backward_equilibrium_points_list,
                    color=deep[3], s=2)
ylab = plt.ylabel('Global activity equilibrium point $X^*$')
plt.xlabel('Global microbial\n interaction weight')
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
    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram' 
                f'_{dynamics_str}_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram' 
                       f'_{dynamics_str}_{graph_str}.png')
    parameters_dictionary = {"graph_str": graph_str, "a": a,
                             "b": b, "c": c,
                             "D": D.tolist(), "n": n, "N": N,
                             "t0": t0, "t1": t1, "t_span": t_span,
                             "t_span_red": t_span_red,
                             "coupling_constants_forward":
                                 coupling_constants_forward.tolist(),
                             "coupling_constants_backward":
                                 coupling_constants_backward.tolist()}

    if integrate_complete_dynamics:
        if forward_branch:
            with open(path + f'{timestr}_{file}'
                      f'_x_forward_equilibrium_points_list'
                      f'_complete_{dynamics_str}_{graph_str}.json', 'w')\
                    as outfile:
                json.dump(x_forward_equilibrium_points_list, outfile)
        if backward_branch:
            with open(path + f'{timestr}_{file}'
                             f'_x_backward_equilibrium_points_list'
                             f'_complete_{dynamics_str}_{graph_str}.json',
                      'w') \
                    as outfile:
                json.dump(x_backward_equilibrium_points_list, outfile)
    if integrate_reduced_dynamics:
        if forward_branch:
            with open(path + f'{timestr}_{file}'
                      f'_redx_forward_equilibrium_points_list'
                      f'_reduced_{dynamics_str}_{graph_str}.json',
                      'w') as outfile:
                json.dump(redx_forward_equilibrium_points_list, outfile)
        if backward_branch:
            with open(path + f'{timestr}_{file}'
                      f'_redx_backward_equilibrium_points_list'
                      f'_reduced_{dynamics_str}_{graph_str}.json',
                      'w') as outfile:
                json.dump(redx_backward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
