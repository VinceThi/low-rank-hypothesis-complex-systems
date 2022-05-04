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

# TODO normalize the network and find the right coupling range ?
# TODO Go back the the equations and change the timescale for instance <<<<<<<<<<<---------- ESPECIALLY THIS !
# TODO - The optimal threshold in the gut micro is 0.67, why ? A mistake ?
# TODO What is the domain of the dynamics !? I didn't even look. It's not bounded by 1 ! ^^^^ adimensionalize the dynamics

""" Graph parameters """
graph_str = "gut"
A = get_microbiome_weight_matrix(graph_str)
N = len(A[0])  # Dimension of the complete dynamics

""" Dynamical parameters """
dynamics_str = "microbial"
t = 0  # It is not involved in the error computation
D = 30*np.eye(N)
a = 5
b = 13
c = 10/3

""" SVD """
U, S, Vh = np.linalg.svd(A)

plot_singular_values(S)

nb_samples = 10
mean_error_list = []
for n in tqdm(range(1, N, 1)):
    Vhn = Vh[:n, :]
    D_sign = np.diag(-(np.sum(Vhn, axis=1) < 0).astype(float)) \
        + np.diag((np.sum(Vhn, axis=1) >= 0).astype(float))
    M = D_sign @ Vhn
    W = A  # / S[0]  # We normalize the network by the largest singular value
    Mp = np.linalg.pinv(M)
    calD = M@D@Mp
    x_samples = np.random.uniform(0, 1, (N, nb_samples))

    # Test zone
    # -----------------------------------
    xtest = x_samples[:, 0]
    gamma = 1
    P = Mp @ M
    Dchi = np.diag((np.eye(N) - P)@xtest)
    Dchinv = np.diag(((np.eye(N) - P)@xtest)**(-1))
    DWchi = np.diag(W@(np.eye(N) - Mp @ M)@xtest)
    d = b*(xtest**2 - (P@xtest)**2) + c*(xtest**3 - (P@xtest)**3)\
        + gamma*(xtest*(W@xtest) - (P@xtest)*(W@P@xtest))
    print(f"\n\nmax a ={np.max(3*c*Dchi@xtest**2)}\n\n",
          f"max b = {np.max((2*b*Dchi + gamma*DWchi)@xtest)}\n\n",
          f"max bcoup = {np.max(gamma*Dchi@W@xtest)}\n\n",
          f"max c = {np.max(d)}")
    print(f"\n\nmean a ={np.mean(3*c*Dchi@xtest**2)}\n\n",
          f"mean b = {np.mean((2*b*Dchi + gamma*DWchi)@xtest)}\n\n",
          f"mean bcoup = {np.mean(gamma*Dchi@W@xtest)}\n\n",
          f"mean c = {np.mean(d)}")

    # print(f"\n\nmax a ={np.max(xtest**2)}\n\n",
    #       f"max b = {np.max(Dchinv@(2*b*Dchi  + gamma*Dchi@W + gamma*DWchi)@xtest/(3*c))}\n\n",
    #       f"max c = {np.max(Dchinv@d/(3*c))}")
    # print(f"\n\nmean a ={np.mean(xtest**2)}\n\n",
    #       f"mean b = {np.mean(Dchinv@(2*b*Dchi  + gamma*Dchi@W + gamma*DWchi)@xtest/(3*c))}\n\n",
    #       f"mean c = {np.mean(Dchinv@d/(3*c))}")
    # -----------------------------------
    min_coupling = 0.1
    max_coupling = 5
    coupling_samples = np.random.uniform(min_coupling, max_coupling,
                                         nb_samples)
    error_list = []
    for i in range(0, nb_samples):
        x = x_samples[:, i]
        coupling = coupling_samples[i]
        error_list.append(rmse(
            M @ microbial(t, x, W, coupling, D, a, b, c),
            reduced_microbial_vector_field(t, M @ x, W, coupling,
                                           M, Mp, D, a, b, c)))
    mean_error_list.append(np.mean(error_list))

fig = plt.figure(figsize=(4, 4))
plt.subplot(111)
plt.scatter(np.arange(1, len(mean_error_list) + 1, 1), mean_error_list,
            color=first_community_color, s=5)
ylab = plt.ylabel(f'Mean RMSE between\n the vector fields')
plt.xlabel('Dimension $n$')
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters,"
                       " the data, and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    path = f'C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/' \
           f'simulations/simulations_data/{dynamics_str}_data/' \
           f'vector_field_errors/'
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    parameters_dictionary = {
        "graph_str": graph_str, "D": D.tolist(), "N": N,
        "nb_samples": nb_samples,
        "x_samples": "np.random.uniform(0, 1, (N, nb_samples))",
        "coupling_samples": f"np.random.uniform({min_coupling},"
                            f" {max_coupling}, nb_samples)"}

    fig.savefig(path + f'{timestr}_{file}_vector_field_rmse_error_vs_n'
                       f'_{dynamics_str}_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_vector_field_rmse_error_vs_n'
                       f'_{dynamics_str}_{graph_str}.png')
    with open(path + f'{timestr}_{file}'
                     f'_mean_error_list'
                     f'_complete_{dynamics_str}_{graph_str}.json', 'w') \
            as outfile:
        json.dump(mean_error_list, outfile)
    with open(path + f'{timestr}_{file}'
                     f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
