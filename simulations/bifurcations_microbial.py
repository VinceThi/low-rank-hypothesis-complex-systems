# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
# from matrix_factorization.snmf import snmf_multiple_inits
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
integrate_complete_dynamics = False
integrate_reduced_dynamics = True
forward_branch = True
backward_branch = True

# Desmos exploration: https://www.desmos.com/calculator/pdtj9p2mdi

""" Time and integration parameters """
t0, t1 = 0, 200
t_span = [t0, t1]
t_span_red = [t0, t1]
integration_method = 'BDF'
rtol = 1e-6   # 1e-8
atol = 1e-10  # 1e-12
tol_eq_pt = 1e-7


def jacobian_complete(t, x, W, coupling, D, a, b, c):
    Jx = jacobian_x_microbial(x, W, coupling, D, b, c)
    Jy = jacobian_y_microbial(x, coupling)
    return Jx + Jy@W


def jacobian_reduced(t, X, W, coupling, M, Mp, D, a, b, c):
    return M@jacobian_complete(t, Mp@X, W, coupling, D, a, b, c)@Mp


""" 
Note: The ODE is stiff and BDF is particularly well suited for 
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


def plot_time_series_observables(x, x_glob, redx, redX_glob, tc, tr, M):
    plt.figure(figsize=(4, 4))
    lw = 0.3
    redlw = 2
    if integrate_complete_dynamics:
        for j in range(0, N):
            plt.plot(tc, x[:, j], color=reduced_first_community_color,
                     linewidth=lw)
        for tau in range(n):
            plt.scatter(tc, M[tau, :]@x.T, color=first_community_color, s=2)
            plt.plot(tc, x_glob, linewidth=redlw, color=deep[2])
    if integrate_reduced_dynamics:
        for tau in range(n):
            plt.scatter(tr, redx[:, tau], color=second_community_color, s=2,
                        linestyle="--")
        plt.plot(tr, redX_glob, linewidth=redlw, color=deep[3])
    ylabel = plt.ylabel('$X_{\\mu}$', labelpad=20)
    ylabel.set_rotation(0)
    plt.xlabel("Time $t$")
    plt.tight_layout()
    plt.show()


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
number_coupling = 50
coupling_constants_forward = np.linspace(0.1, 3, number_coupling)
coupling_constants_backward = np.linspace(0.1, 3, number_coupling)
coupling_constants = []

number_initial_conditions = 300
max_randint_CI = 15


""" SVD and dimension reduction """
n_min = 76
n = 76  # 76, 203, 400, 735  # Dimension of the reduced dynamics >= n_min
Un, Sn, M = computeTruncatedSVD_more_positive(W, n)
print("\n", computeEffectiveRanks(svdvals(W), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")
Mp = pinv(M)

if n < n_min:
    raise ValueError("n < n_min. "
                     "For Fig. 3 of the paper, we want global observables "
                     "that does not vary with $n$, so we set a minimal value"
                     "to compare equilibrium points branches at different "
                     "values of n greater or equal to n_min.")

# We try to be as close as possible to the uniform observable with n_min
# right singular vectors
normalization_constant = 10
ell_bar = np.sum(M[:n_min, :], axis=1)/(normalization_constant*N)
ell = np.concatenate([ell_bar, np.zeros(n - n_min)])
m = ell@M

timestr_M_D = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")


""" Get equilibrium points """
xf = np.zeros((number_initial_conditions, number_coupling))
redxf = np.zeros((number_initial_conditions, number_coupling))
xb = np.zeros((number_initial_conditions, number_coupling))
redxb = np.zeros((number_initial_conditions, number_coupling))
for i in tqdm(range(number_initial_conditions)):
    if forward_branch:
        x_forward_equilibrium_points_list = []
        redx_forward_equilibrium_points_list = []
        x0 = np.random.random(N)
        redx0 = M@x0
        print("\n Iterating on coupling constants"
              " for equilibrium point diagram(f) \n")
        for coupling in tqdm(coupling_constants_forward):

            if integrate_complete_dynamics:
                eq_pt_negative = True
                eq_pt_unreach = True
                while eq_pt_negative or eq_pt_unreach:
                    args_dynamics = (W, coupling, D, a, b, c)
                    sol = solve_ivp(microbial, t_span, x0, integration_method,
                                    args=args_dynamics, rtol=rtol, atol=atol,
                                    vectorized=True, jac=jacobian_complete)
                    x = sol.y.T
                    tc = sol.t

                    """ Verify if the equilibrium point is reached """
                    eval_f = microbial(0, x[-1, :], W, coupling, D, a, b, c)
                    if not np.all(eval_f < tol_eq_pt):
                        print(f"An equilibrium point is not reached "
                              f"for the complete dynamics (forward branch)."
                              f" \n The maximum absolute error is"
                              f" {np.max(np.abs(eval_f))}. \n "
                              f"Another initial condition is sampled at the "
                              f"coupling value {coupling}.")
                        x0 = np.random.random(N)
                    else:
                        eq_pt_unreach = False
                        """ Get global observables """
                        x_glob = np.sum(m*x, axis=1)
                        equilibrium_point = x_glob[-1]
                        if equilibrium_point > 0:
                            eq_pt_negative = False
                            x_forward_equilibrium_points_list.append(
                                equilibrium_point)
                            """ Reset initial condition at last eq. point """
                            x0 = x[-1, :]
                        else:
                            x0 = np.random.random(N)

            if integrate_reduced_dynamics:
                eq_pt_negative = True
                eq_pt_unreach = True
                while eq_pt_negative or eq_pt_unreach:
                    args_reduced_dynamics = (W, coupling, M, Mp, D, a, b, c)
                    sol = solve_ivp(reduced_microbial_vector_field, t_span,
                                    redx0, integration_method,
                                    args=args_reduced_dynamics,
                                    rtol=rtol, atol=atol, vectorized=True,
                                    jac=jacobian_reduced)
                    redx = sol.y.T
                    tr = sol.t

                    """ Verify if the equilibrium point is reached """
                    eval_redf = reduced_microbial_vector_field(0, redx[-1, :],
                                                               W, coupling, M,
                                                               Mp, D, a, b, c)
                    if not np.all(eval_redf < tol_eq_pt):
                        print(f"An equilibrium point is not reached "
                              f"for the reduced dynamics (forw. branch). \n\n "
                              f"The maximum absolute error is"
                              f" {np.max(np.abs(eval_redf))}. \n\n "
                              f"Another initial condition is sampled at the "
                              f"coupling value {coupling}.")
                        redx0 = M@np.random.random(N)
                    else:
                        eq_pt_unreach = False

                        """ Get global observables """
                        redX_glob = np.zeros(len(redx[:, 0]))
                        for nu in range(n):
                            redX_nu = redx[:, nu]
                            redX_glob = redX_glob + ell[nu]*redX_nu

                        red_equilibrium_point = redX_glob[-1]
                        if red_equilibrium_point > 0:
                            eq_pt_negative = False
                            redx_forward_equilibrium_points_list.append(
                                red_equilibrium_point)
                            """ Reset initial condition at last eq. point """
                            redx0 = redx[-1, :]
                        else:
                            redx0 = M@np.random.random(N)

            if plot_time_series and integrate_complete_dynamics\
                    and integrate_reduced_dynamics:
                plot_time_series_observables(x, x_glob, redx, redX_glob,
                                             tc, tr, M)
        if integrate_complete_dynamics:
            xf[i, :] = x_forward_equilibrium_points_list
        if integrate_reduced_dynamics:
            redxf[i, :] = redx_forward_equilibrium_points_list

    if backward_branch:
        x_backward_equilibrium_points_list = []
        redx_backward_equilibrium_points_list = []
        x0_upper_limit = np.random.randint(1, max_randint_CI)
        x0_b = np.random.uniform(0, x0_upper_limit, N)
        redx0_b = M@x0_b
        print("\n Iterating on coupling constants"
              " for equilibrium point diagram(b)\n")
        for coupling in tqdm(coupling_constants_backward[::-1]):

            if integrate_complete_dynamics:
                eq_pt_negative = True
                eq_pt_unreach = True
                while eq_pt_negative or eq_pt_unreach:
                    args_dynamics = (W, coupling, D, a, b, c)
                    sol = solve_ivp(microbial, t_span, x0_b,
                                    integration_method,
                                    args=args_dynamics, rtol=rtol, atol=atol,
                                    vectorized=True, jac=jacobian_complete)
                    x = sol.y.T
                    tc = sol.t

                    """ Verify if the equilibrium point is reached """
                    eval_f = microbial(0, x[-1, :], W, coupling, D, a, b, c)
                    if not np.all(eval_f < tol_eq_pt):
                        print(f"An equilibrium point is not reached "
                              f"for the complete dynamics (backward branch)."
                              f" \n The maximum absolute error is"
                              f" {np.max(np.abs(eval_f))}. \n "
                              f"Another initial condition is sampled at the "
                              f"coupling value {coupling}.")
                        x0_upper_limit = np.random.randint(1, max_randint_CI)
                        x0 = np.random.uniform(0, x0_upper_limit, N)
                    else:
                        eq_pt_unreach = False

                        """ Get global observables """
                        x_glob = np.sum(m*x, axis=1)
                        equilibrium_point = x_glob[-1]
                        if equilibrium_point > 0:
                            eq_pt_negative = False
                            x_backward_equilibrium_points_list.append(
                                equilibrium_point)
                            """ Reset initial condition at last eq. point """
                            x0_b = x[-1, :]
                        else:
                            print(f"The equilibrium point is negative "
                                  f"for the complete dynamics(backward branch)"
                                  f".\n Another initial condition is sampled"
                                  f" at the coupling value {coupling}.")
                            x0_upper_limit = \
                                np.random.randint(1, max_randint_CI)
                            x0_b = np.random.uniform(0, x0_upper_limit, N)

            if integrate_reduced_dynamics:
                eq_pt_negative = True
                eq_pt_unreach = True
                while eq_pt_negative or eq_pt_unreach:
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
                    eval_redf = reduced_microbial_vector_field(0, redx[-1, :],
                                                               W, coupling, M,
                                                               Mp, D, a, b, c)
                    if not np.all(eval_redf < tol_eq_pt):
                        print(f"An equilibrium point is not reached "
                              f"for the reduced dynamics (back. branch). \n\n "
                              f"The maximum absolute error is"
                              f" {np.max(np.abs(eval_redf))}. \n\n "
                              f"Another initial condition is sampled at the "
                              f"coupling value {coupling}.")
                        upper = np.random.randint(1, max_randint_CI)
                        x0_upper_limit = \
                            np.random.randint(1, max_randint_CI)
                        redx0_b = M@np.random.uniform(0, x0_upper_limit, N)

                    else:
                        eq_pt_unreach = False
                        """ Get global observables """
                        redX_glob = np.zeros(len(redx[:, 0]))
                        redX_sub_glob = np.zeros(len(redx[:, 0]))
                        for nu in range(n):
                            redX_nu = redx[:, nu]
                            redX_glob = redX_glob + ell[nu]*redX_nu
                        red_equilibrium_point = redX_glob[-1]
                        if red_equilibrium_point > 0:
                            eq_pt_negative = False
                            redx_backward_equilibrium_points_list.append(
                                red_equilibrium_point)
                            """ Reset initial condition at last eq. points"""
                            redx0_b = redx[-1, :]
                        else:
                            print(f"The equilibrium point is negative "
                                  f"for the reduced dynamics (back. branch)."
                                  f"\n\n Another initial condition is sampled "
                                  f"at the coupling value {coupling}.")
                            x0_upper_limit = \
                                np.random.randint(1, max_randint_CI)
                            redx0_b = M@np.random.uniform(0, x0_upper_limit, N)

            if plot_time_series and integrate_complete_dynamics \
                    and integrate_reduced_dynamics:
                plot_time_series_observables(x, x_glob, redx,
                                             redX_glob, tc, tr, M)
        if integrate_complete_dynamics:
            x_backward_equilibrium_points_list.reverse()
            xb[i, :] = x_backward_equilibrium_points_list
        if integrate_reduced_dynamics:
            redx_backward_equilibrium_points_list.reverse()
            redxb[i, :] = redx_backward_equilibrium_points_list
              
fig = plt.figure(figsize=(4, 4))
redlinewidth = 2
plt.subplot(111)
if integrate_complete_dynamics:
    if forward_branch:
        for i in range(number_initial_conditions):
            plt.scatter(coupling_constants_forward, xf[i, :],
                        color=deep[0], s=3)
        plt.plot(coupling_constants_forward, np.mean(xf, axis=0),
                 color=deep[0], linewidth=1, label="Complete (f)")
    if backward_branch:
        for i in range(number_initial_conditions):
            plt.scatter(coupling_constants_backward, xb[i, :],
                        color=deep[4], s=3)
        plt.plot(coupling_constants_backward, np.mean(xb, axis=0),
                 color=deep[4], linewidth=1, label="Complete (b)")
if integrate_reduced_dynamics:
    if forward_branch:
        for i in range(number_initial_conditions):
            plt.scatter(coupling_constants_forward, redxf[i, :],
                        color=deep[1], s=3)
        plt.plot(coupling_constants_forward, np.mean(redxf, axis=0),
                 color=deep[1], linewidth=1, label="Reduced (f)")
    if backward_branch:
        for i in range(number_initial_conditions):
            plt.scatter(coupling_constants_backward, redxb[i, :],
                        color=deep[3], s=3)
        plt.plot(coupling_constants_backward, np.mean(redxb, axis=0),
                 color=deep[3], linewidth=1, label="Reduced (b)")
ylab = plt.ylabel('Global activity equilibrium point $\mathcal{X}^*$')
plt.xlabel('Microbial\n interaction weight')
plt.tick_params(axis='both', which='major')
plt.legend(loc=2, fontsize=8)
plt.ylim([-0.02, 1.02])
plt.yticks([0, 0.5, 1])
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
                             "rtol": rtol, "atol": atol,
                             "tol_eq_pt": tol_eq_pt,
                             "number initial conditions":
                                 number_initial_conditions,
                             "number coupling constants": number_coupling,
                             "coupling_constants_forward":
                                 coupling_constants_forward.tolist(),
                             "coupling_constants_backward":
                                 coupling_constants_backward.tolist(),
                             "initial conditions distribution "
                             "(forward branch)": "np.random.uniform(0, 1, N)",
                             "initial conditions distribution"
                             " (backward branch)":
                             f"np.random.uniform(0, "
                             f"randint(1, {max_randint_CI}), N)"}

    if integrate_complete_dynamics:
        if forward_branch:
            with open(path + f'{timestr}_{file}'
                      f'_x_forward_equilibrium_points_list'
                      f'_complete_{dynamics_str}_{graph_str}.json', 'w')\
                    as outfile:
                json.dump(xf.tolist(), outfile)
        if backward_branch:
            with open(path + f'{timestr}_{file}'
                             f'_x_backward_equilibrium_points_list'
                             f'_complete_{dynamics_str}_{graph_str}.json',
                      'w') \
                    as outfile:
                json.dump(xb.tolist(), outfile)
    if integrate_reduced_dynamics:
        if forward_branch:
            with open(path + f'{timestr}_{file}'
                      f'_redx_forward_equilibrium_points_list'
                      f'_reduced_{dynamics_str}_{graph_str}.json',
                      'w') as outfile:
                json.dump(redxf.tolist(), outfile)
        if backward_branch:
            with open(path + f'{timestr}_{file}'
                      f'_redx_backward_equilibrium_points_list'
                      f'_reduced_{dynamics_str}_{graph_str}.json',
                      'w') as outfile:
                json.dump(redxb.tolist(), outfile)
    with open(path + f'{timestr}_{file}'
              f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
