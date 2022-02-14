# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
from dynamics.dynamics import qmf_sis
from dynamics.reduced_dynamics import reduced_qmf_sis, reduced_qmf_sis_vector_field
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

plot_time_series = False

""" Time parameters """
t0, t1, dt = 0, 30, 0.1
timelist = np.linspace(t0, t1, int(t1 / dt))

""" Graph parameters """
graph_str = "high_school_proximity"
path_str = f"C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
           f"graphs/graph_data/{graph_str}/"
#  import csv
#  with open(path_str + "edges.csv", "rt") as source:
#     rdr = csv.reader(source)
#     with open(path_str + "edges_no_time.csv", "wt") as result:
#         wtr = csv.writer(result)
#         for r in rdr:
#             wtr.writerow((r[0], r[1]))
G = nx.read_edgelist(path_str + "edges_no_time.csv", delimiter=',',
                     create_using=nx.Graph)
A = nx.to_numpy_array(G)
N = len(A[0])  # Dimension of the complete dynamics
# plt.matshow(A, aspect="auto")
# plt.show()

""" Dynamical parameters """
dynamics_str = "qmf_sis"
D = np.eye(N)
coupling_constants = np.linspace(0.01, 4, 50)

""" SVD and dimension reduction """
n = 10  # Dimension of the reduced dynamics
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
calW, calW_tensor3 = M@W@Mp, compute_tensor_order_3(M, W)

""" Integration """
x_equilibrium_points_list = []
redx_equilibrium_points_list = []
print("\n Iterating on coupling constants for equilibrium point diagram... \n")
for coupling in tqdm(coupling_constants):

    x0 = np.random.random(N)
    redx0 = M@x0

    # Integrate complete dynamics
    args_dynamics = (coupling, D)
    x = np.array(integrate_dopri45(t0, t1, dt, qmf_sis,
                                   W, x0, *args_dynamics))
    x_glob = np.sum(m*x, axis=1)

    # /!\ Look carefully if the dynamics reach an equilibrium point
    equilibrium_point = x_glob[-1]
    x_equilibrium_points_list.append(equilibrium_point)
    # x0 = x[-1, :]

    # Integrate reduced dynamics
    # args_reduced_dynamics = (calW_tensor3, coupling, calD)
    # redx = np.array(integrate_dopri45(t0, t1, dt, reduced_qmf_sis,
    #                                   calW, redx0,
    #                                   *args_reduced_dynamics))
    args_reduced_dynamics = (coupling, M, Mp, D)
    redx = np.array(integrate_dopri45(t0, t1, dt, reduced_qmf_sis_vector_field,
                                      W, redx0, *args_reduced_dynamics))
    # redx0 = redx[-1, :]

    # Get global observables
    redX_glob = np.zeros(len(redx[:, 0]))
    for nu in range(n):
        redX_nu = redx[:, nu]
        redX_glob = redX_glob + ell[nu]*redX_nu
    # /!\ Look carefully if the dynamics reach an equilibrium point
    red_equilibrium_point = redX_glob[-1]
    redx_equilibrium_points_list.append(red_equilibrium_point)

    if plot_time_series:
        plt.figure(figsize=(4, 4))
        # for j in range(0, N):
        #     plt.plot(timelist, x[:, j], color=reduced_first_community_color,
        #              linewidth=0.3)
        for nu in range(n):
            plt.plot(timelist, M[nu, :]@x.T, color=first_community_color)
            plt.plot(timelist, redx[:, nu], color=second_community_color,
                     linestyle="--")
        ylab = plt.ylabel('$X_{\\mu}$', labelpad=20)
        ylab.set_rotation(0)
        plt.tight_layout()
        plt.show()

fig = plt.figure(figsize=(4, 4))
redlinewidth = 2
plt.subplot(111)
plt.plot(coupling_constants, x_equilibrium_points_list,
         color=first_community_color, label="Complete")
plt.plot(coupling_constants, redx_equilibrium_points_list,
         color=second_community_color, label="Reduced")
plt.plot(coupling_constants, np.zeros(len(coupling_constants)),
         color=first_community_color, linestyle='--')
plt.plot(coupling_constants, np.zeros(len(coupling_constants)),
         color=second_community_color, linestyle='--')
ylab = plt.ylabel('Global activity equilibrium point $X^*$')
plt.xlabel('Infection rate')
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

    parameters_dictionary = {"graph_str": graph_str,
                             "D": D.tolist(), "n": n, "N": N,
                             "t0": t0, "t1": t1, "dt": dt,
                             "coupling_constants": coupling_constants.tolist()}

    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
                f'_{dynamics_str}_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
                       f'_{dynamics_str}_{graph_str}.png')
    with open(path + f'{timestr}_{file}'
              f'_x_equilibrium_points_list'
              f'_complete_{dynamics_str}_{graph_str}.json', 'w') \
            as outfile:
        json.dump(x_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_redx_equilibrium_points_list'
              f'_reduced_{dynamics_str}_{graph_str}.json',
              'w') as outfile:
        json.dump(redx_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
