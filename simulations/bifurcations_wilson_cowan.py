# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.integrate import *
from dynamics.dynamics import wilson_cowan
from dynamics.reduced_dynamics import reduced_wilson_cowan
from singular_values.compute_effective_ranks import computeEffectiveRanks
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from plots.config_rcparams import *
from scipy.linalg import pinv, svdvals
import pandas as pd
import networkx as nx
from tqdm import tqdm
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

plot_time_series = False
compute_error = False
plot_weight_matrix = False
save_weight_matrix_no_d_reduction_matrix = False

""" Time parameters """
t0, t1, dt = 0, 250, 0.2
timelist = np.linspace(t0, t1, int(t1 / dt))

""" Graph parameters """
graph_str = "celegans"
path_str = "C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
           "graphs/graph_data/connectomes/"

if graph_str == "Gilbert":
    n1 = 60
    n2 = 40
    sizes = [n1, n2]
    N = sum(sizes)
    p = 0.5
    pq = [[p, p], [p, p]]

elif graph_str == "SBM":
    n1 = 60
    n2 = 40
    sizes = [n1, n2]
    N = sum(sizes)
    # p11, p22 = 0.7, 0.5
    # pout = 0.05
    p11, p22 = 0.4, 0.4
    pout = 0.2
    pq = [[p11, pout], [pout, p22]]

if graph_str == "Gilbert" or graph_str == "SBM":

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
        A = None
        print("WARNING: this weight_str is not"
              " among the available choice.")

elif graph_str == "mouse_meso":
    # Oh, S., Harris, J., Ng, L. et al.
    # A mesoscale connectome of the mouse brain.
    # Nature 508, 207–214 (2014) doi:10.1038/nature13186

    # To binary matrix  (with "> 0")
    A = (np.loadtxt(path_str + "ABA_weight_mouse.txt") > 0).astype(float)
    # W = (W + W.T) / 2
    G_mouse_meso = nx.from_numpy_array(A)
    N = G_mouse_meso.number_of_nodes()
    print(f"N = {N}")
    # N = 213
    # rank_mouse_meso = 185

elif graph_str == "zebrafish_meso":
    # Kunst et al. "A Cellular-Resolution Atlas of the Larval Zebrafish Brain",
    # (2019) avec le traitement de Antoine Légaré
    # On a pas exactement les mêmes régions que l'article non plus,
    #  où la matrice est faite avec 36 régions. Ici, on en a 71 qui sont
    #  mutually exclusive et collectively exhaustive (je reprends les
    # termes du dude dans le courriel) donc ça couvre tout le volume au
    #  complet sans overlap

    df = pd.read_csv(path_str +
                     'Connectivity_matrix_zebra_fish_mesoscopic.csv')
    dictio = {'X': 0}  # We put zeros temporarily on the diagonal
    df = df.replace(dictio)

    volumes = np.array(1 * np.load(path_str + "volumes_zebrafish_meso.npy"))
    relativeVolumes = volumes / sum(volumes)
    adjacency = df.to_numpy()[:, 1:-1].astype(float)
    N = len(adjacency[0])
    # """ To get an undirected graph """
    # for i in range(adjacency.shape[0]):
    #     for j in range(i+1, adjacency.shape[0]):
    #         adjacency[i, j] = (adjacency[i, j] + adjacency[j, i]) /
    #  (relativeVolumes[i] + relativeVolumes[j])
    #         adjacency[j, i] = adjacency[i, j]
    """ To get a directed graph """
    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[0]):
            adjacency[i, j] = adjacency[i, j] / (
                    relativeVolumes[i] + relativeVolumes[j])
    adjacency = adjacency / np.amax(adjacency)
    adjacency = np.log(adjacency + 0.00001)
    adjacency -= np.amin(adjacency)
    adjacency = adjacency / np.amax(adjacency)
    A = adjacency + np.eye(N)

    print(f"N = {N}")
    # N = 71
    # rank_zebrafish_meso = 71

elif graph_str == "celegans":
    # Data obtained from Mohamed Bahdine, extracted as described in the
    # supplementary material of the article : Network control principles
    # predict neuron function in the C. elegans connectome - Yan,..., Barabasi
    # The data come from Wormatlas.
    A = np.array(1 * np.load(path_str + "C_Elegans.npy"))
    G_celegans = nx.from_numpy_array(A)
    N = G_celegans.number_of_nodes()
    print(f"N = {N}")
    # N = 279
    # rank_celegans = 273

# elif graph_str == "drosophila":
#     df = pd.read_csv(path_str +'drosophila_exported-traced-adjacencies-v1.1/'
#                                 'traced-total-connections.csv')
#     Graphtype = nx.DiGraph()
#     G_drosophila = nx.from_pandas_edgelist(df,
#                                            source='bodyId_pre',
#                                            target='bodyId_post',
#                                            edge_attr='weight',
#                                            create_using=Graphtype)
#     W = nx.to_numpy_array(G_drosophila)
#     N = G_drosophila.number_of_nodes()
#     print(f"N = {N}")
#     # N = 21733
#     # rank_drosophila =

elif graph_str == "ciona":
    A_from_xlsx = pd.read_excel(path_str +
                                'ciona_intestinalis_lavaire_elife'
                                '-16962-fig16-data1-v1_modified.xlsx').values
    A_ciona_nan = np.array(A_from_xlsx[0:, 1:])
    A_ciona = np.array(A_ciona_nan, dtype=float)
    where_are_NaNs = np.isnan(A_ciona)
    A_ciona[where_are_NaNs] = 0
    A = (A_ciona > 0).astype(float)
    G_ciona = nx.from_numpy_array(A_ciona)

    N = G_ciona.number_of_nodes()
    print(f"N = {N}")
    # N = 213
    # rank_ciona = 203

else:
    #
    raise ValueError("This graph_str is not an option.")

if plot_weight_matrix:
    fig = plt.figure(figsize=(4.5, 4))
    ax = plt.subplot(111)
    cax = ax.matshow(A, aspect='auto')
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("$W_{ij}$", fontsize=14, rotation=0, labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()


""" Dynamical parameters """
D = np.eye(N)
a = 0.1
b = 1
c = 3

# sigma1 times these linspace
coupling_constants = np.linspace(0.35, 0.9, 10)  # celegans,ciona,zebrafish
# coupling_constants = np.linspace(0.5, 8.5, 100)# mouse_meso (not connex ?)
# coupling_constants = np.linspace(0.6, 1.2, 100)# mouse_meso hysteresis 1
# coupling_constants = np.linspace(5, 10, 100)   # mouse_meso hysteresis 2


""" SVD and dimension reduction """
n = 1  # Dimension of the reduced dynamics
Un, Sn, Vhn = computeTruncatedSVD_more_positive(A, n)
L, M = Un@Sn, Vhn
print(computeEffectiveRanks(svdvals(A), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")

W = A/Sn[0][0]  # We normalize the network by the largest singular value

Mp = pinv(M)
s = np.array([np.eye(n, n)[0, :]])
# ^ This yields the dominant singular vector observable
m = s @ M / np.sum(s @ M)
ell = (s / np.sum(s @ M))[0]
calD = M@D@Mp
timestr_M_D = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

""" Forward branch  """
x_forward_equilibrium_points_list = []
redx_forward_equilibrium_points_list = []

x0 = -10 * np.random.random(N)
redx0 = M @ x0
for coupling in tqdm(coupling_constants):

    # Integrate complete dynamics
    args_dynamics = (coupling, D, a, b, c)
    x = np.array(integrate_dopri45(t0, t1, dt, wilson_cowan,
                                   W, x0, *args_dynamics))
    x_glob = np.sum(m*x, axis=1)

    # /!\ Look carefully if the dynamics reach an equilibrium point
    equilibrium_point = x_glob[-1]
    x_forward_equilibrium_points_list.append(equilibrium_point)
    x0 = x[-1, :]

    # Integrate reduced dynamics
    args_reduced_dynamics = (M, coupling, calD, a, b, c)
    redx = np.array(integrate_dopri45(t0, t1, dt, reduced_wilson_cowan,
                                      L, redx0, *args_reduced_dynamics))
    # args_reduced_dynamics = (coupling, M, Mp, D, a, b, c)
    # redx = np.array(integrate_dopri45(t0, t1, dt, reduced_wilson_cowan,
    #                                   W, redx0, *args_reduced_dynamics))
    redx0 = redx[-1, :]

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

""" Backward branch """
x_backward_equilibrium_points_list = []
redx_backward_equilibrium_points_list = []
x0_b = 10 * np.random.random(N)
redx0_b = M @ x0_b
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
    # args_reduced_dynamics = (M, coupling, calD, a, b, c)
    # redx = np.array(integrate_dopri45(t0, t1, dt, reduced_wilson_cowan,
    #                                   L, redx0_b, *args_reduced_dynamics))
    args_reduced_dynamics = (coupling, M, Mp, D, a, b, c)
    redx = np.array(integrate_dopri45(t0, t1, dt, reduced_wilson_cowan,
                                      W, redx0_b, *args_reduced_dynamics))
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
    path = 'C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/' \
           'simulations/simulations_data/wilson_cowan_data/'
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    parameters_dictionary = {"graph_str": graph_str, "a": a, "b": b, "c": c,
                             "D": D, "n": n, "N": N,
                             "t0": t0, "t1": t1, "dt": dt,
                             "coupling_constants": coupling_constants.tolist()}

    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
                f'_wilson_cowan_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
                       f'_wilson_cowan_{graph_str}.png')
    with open(path + f'{timestr}_{file}'
              f'_x_forward_equilibrium_points_list'
              f'_complete_wilson_cowan_graph_str_{graph_str}.json', 'w') \
            as outfile:
        json.dump(x_forward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_redx_forward_equilibrium_points_list'
              f'_reduced_wilson_cowan_{graph_str}.json',
              'w') as outfile:
        json.dump(redx_forward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_x_backward_equilibrium_points_list'
              f'_complete_wilson_cowan_{graph_str}.json', 'w') \
            as outfile:
        json.dump(x_backward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_redx_backward_equilibrium_points_list'
              f'_reduced_wilson_cowan_{graph_str}.json',
              'w') as outfile:
        json.dump(redx_backward_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_wilson_cowan_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
