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
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import time
import json
import tkinter.simpledialog
from tkinter import messagebox


plot_weight_matrix_bool = False
plot_singvals_bool = False

""" Time parameters """
t0, t1 = 0, 300
t_span = [t0, t1]
integration_method = "BDF"
rtol = 1e-8
atol = 1e-12

""" Graph parameters """
graph_str = "mouse_control_rnn"
A = get_learned_weight_matrix(graph_str)
U, S, Vh = np.linalg.svd(A)
print("\n", computeEffectiveRanks(svdvals(A), graph_str, len(A)))
shrink_s = optimal_shrinkage(S, 1, 'frobenius')
W = U@np.diag(shrink_s)@Vh
N = len(A[0])

if plot_weight_matrix_bool:
    plot_weight_matrix(A)

if plot_singvals_bool:
    plot_singular_values(svdvals(A))

""" Dynamical parameters """
dynamics_str = "rnn"
D = 1*np.eye(N)
coupling = 2
# Interesting parameters for graph_str = "mouse_control_rnn"
# n = 100 : D = I, coupling = 2
# n = 90 : D = I, coupling = 2

# Interesting parameters for graph_str = "mouse_rnn"
# n = 46 : D = 1.5I , coupling = 18 , t = 210
# n = 47, D = 1.5I, coupling = 15 , t1 = 200
#         D = 1.6I, coupling = 16 , t1 = 200
#         D = 2I, coupling = 20   , t1 = 200

""" SVD and dimension reduction """
n = 80  # Dimension of the reduced dynamics
Un, Sn, Vhn = computeTruncatedSVD_more_positive(A, n)
L, M = Un@Sn, Vhn
Mp = pinv(M)
print(f"\n After the shrinkage: \n")
print("\n", computeEffectiveRanks(svdvals(W), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")
print(f"\ncoupling = {coupling} \n")

""" Get trajectories """
x0 = np.linspace(-1, 1, N)


# Integrate complete dynamics
args_dynamics2 = (W, coupling, D)
sol = solve_ivp(rnn, t_span, x0, integration_method, args=args_dynamics2,
                rtol=rtol, atol=atol, vectorized=True)
x = sol.y
tc = sol.t
X1c, X2c, X3c = M[0, :]@x, M[1, :]@x, M[2, :]@x


# Integrate reduced dynamics
redx0 = M@x0
# redx0 = M@x[:, -1]
args_reduced_dynamics = (coupling, M, Mp, D)
args_reduced_dynamics2 = (W, coupling, M, Mp, D)
red_sol = solve_ivp(reduced_rnn_vector_field, t_span, redx0,
                    integration_method, args=args_reduced_dynamics2,
                    rtol=rtol, atol=atol, vectorized=True)
redx = red_sol.y
tr = red_sol.t
X1r, X2r, X3r = redx[0, :], redx[1, :], redx[2, :]

fig = plt.figure(figsize=(12, 6))
linewidth = 0.3
redlinewidth = 2

plt.subplot(231)
plt.scatter(tc, X1c, color=first_community_color, s=1, label=f"Complete")
plt.scatter(tr, X1r, color=second_community_color, s=1, label=f"Reduced")
plt.xlabel("Time $t$")
plt.ylabel("$X_1$")
plt.legend()

plt.subplot(232)
plt.scatter(tc, X2c, color=first_community_color, s=1, label=f"Complete")
plt.scatter(tr, X2r, color=second_community_color, s=1, label=f"Reduced")
plt.xlabel("Time $t$")
plt.ylabel("$X_2$")
plt.legend()

plt.subplot(233)
plt.scatter(tc, X3c, color=first_community_color, s=1, label=f"Complete")
plt.scatter(tr, X3r, color=second_community_color, s=1, label=f"Reduced")
plt.xlabel("Time $t$")
plt.ylabel("$X_3$")
plt.legend()

time_cut = 0.5

ax4 = plt.subplot(234)
ax4.plot(X1c[int(time_cut*len(tc)):],
         X2c[int(time_cut*len(tc)):],
         color=first_community_color,
         linewidth=redlinewidth, label="Complete dynamics")
ax4.plot(X1r[int(time_cut*len(tr)):],
         X2r[int(time_cut*len(tr)):],
         color=second_community_color,
         linewidth=redlinewidth, linestyle="--", label="Reduced dynamics")
ax4.set_ylabel("$X_2$")
ax4.set_xlabel("$X_1$")
for j in range(0, N):
    plt.plot(tc, x[j, :], color=reduced_first_community_color,
             linewidth=linewidth)
# for nu in range(n):
#     plt.plot(tc, M[nu, :]@x, color=first_community_color,
#              linewidth=redlinewidth)
#     plt.plot(tr, redx[nu, :], color=second_community_color,
#              linewidth=redlinewidth, linestyle="--")
# # plt.plot(timelist, M[0, :] @ x, color="k",
# #          linewidth=redlinewidth)
# # plt.plot(timelist, redx[:, 0], color=dark_grey,
# #          linewidth=redlinewidth, linestyle="--")
# # plt.plot(timelist, M[-1, :] @ x, color="r",
# #          linewidth=redlinewidth)
# # plt.plot(timelist, redx[-1, :], color="r",
# #          linewidth=redlinewidth, linestyle="--")
# plt.xlabel("Time $t$")
# plt.ylabel("$X_{\\mu}$ for all $\\mu$")

ax5 = plt.subplot(235)
ax5.plot(X1c[int(time_cut*len(tc)):],
         X3c[int(time_cut*len(tc)):],
         color=first_community_color,
         linewidth=redlinewidth, label="Complete dynamics")
ax5.plot(X1r[int(time_cut*len(tr)):],
         X3r[int(time_cut*len(tr)):],
         color=second_community_color,
         linewidth=redlinewidth, linestyle="--", label="Reduced dynamics")
ax5.set_ylabel("$X_3$")
ax5.set_xlabel("$X_1$")

ax6 = fig.add_subplot(236, projection='3d')
# ax5.set_title("Complete dynamics")
ax6.scatter(X1c[int(time_cut*len(tc)):],
            X2c[int(time_cut*len(tc)):],
            X3c[int(time_cut*len(tc)):], color=first_community_color,
            linewidth=redlinewidth, label="Complete dynamics", s=1)
ax6.plot(X1r[int(time_cut*len(tr)):],
         X2r[int(time_cut*len(tr)):],
         X3r[int(time_cut*len(tr)):], color=second_community_color,
         linewidth=redlinewidth, linestyle="--", label="Reduced dynamics")
ax6.set_zlabel("$X_3$")
ax6.set_ylabel("$X_2$")
ax6.set_xlabel("$X_1$")
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

    parameters_dictionary = {"graph_str": graph_str,
                             "D": D.tolist(), "n": n, "N": N,
                             "t0": t0, "t1": t1,  # "dt": dt,
                             "coupling": coupling}

    fig.savefig(path + f'{timestr}_{file}_trajectory'
                       f'_{dynamics_str}_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_trajectory'
                       f'_{dynamics_str}_{graph_str}.png')
    with open(path + f'{timestr}_{file}'
                     f'_x_time_series'
                     f'_complete_{dynamics_str}_{graph_str}.json', 'w') \
            as outfile:
        json.dump(x.tolist(), outfile)
    with open(path + f'{timestr}_{file}'
                     f'_redx_time_series'
                     f'_reduced_{dynamics_str}_{graph_str}.json',
              'w') as outfile:
        json.dump(redx.tolist(), outfile)
    with open(path + f'{timestr}_{file}'
                     f'_time_points'
                     f'_complete_{dynamics_str}_{graph_str}.json', 'w') \
            as outfile:
        json.dump(tc.tolist(), outfile)
    with open(path + f'{timestr}_{file}'
                     f'_time_points'
                     f'_reduced_{dynamics_str}_{graph_str}.json',
              'w') as outfile:
        json.dump(tr.tolist(), outfile)
    with open(path + f'{timestr}_{file}'
                     f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
