# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import networkx as nx
import numpy as np
from graphs.compute_tensors import compute_tensor_order_3
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from singular_values.compute_effective_ranks import computeEffectiveRanks
from scipy.linalg import svdvals , pinv
import matplotlib.pyplot as plt

# graph_str = "high_school_proximity"
# path_str = f"C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
#            f"graphs/graph_data/{graph_str}/"
# G = nx.read_edgelist(path_str + "edges_no_time.csv", delimiter=',',
#                      create_using=nx.Graph)
# A = nx.to_numpy_array(G)
graph_str = "caribbean"  # "little_rock"
path_str = f"C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/" \
           f"graphs/graph_data/foodwebs/{graph_str}/"
if graph_str == "little_rock":
    G = nx.read_edgelist(path_str + "edges.csv", delimiter=',',
                         create_using=nx.DiGraph)
    A = nx.to_numpy_array(G).T
elif graph_str == "caribbean":
    A = np.genfromtxt(graph_str, delimiter=",")
else:
    raise ValueError("This graph_str is not an option.")
# A = A - A.T
N = len(A[0])
n = 9
Un, Sn, M = computeTruncatedSVD_more_positive(A, n)
print(computeEffectiveRanks(svdvals(A), graph_str, N))
print(f"\nDimension of the reduced system n = {n} \n")

W = A/Sn[0][0]  # We normalize the network by the largest singular value

plt.matshow(W)
plt.show()

calW = M@W@pinv(M)
print(calW)
plt.matshow(calW)
plt.show()

calW3 = compute_tensor_order_3(M, W)

plt.figure(figsize=(8, 8))
ax1 = plt.subplot(3, 3, 1)
ax1.matshow(calW3[0, :, :])
ax2 = plt.subplot(3, 3, 2)
ax2.matshow(calW3[1, :, :])
ax3 = plt.subplot(3, 3, 3)
ax3.matshow(calW3[2, :, :])
ax4 = plt.subplot(3, 3, 4)
ax4.matshow(calW3[3, :, :])
ax5 = plt.subplot(3, 3, 5)
ax5.matshow(calW3[4, :, :])
ax6 = plt.subplot(3, 3, 6)
ax6.matshow(calW3[5, :, :])
ax7 = plt.subplot(3, 3, 7)
ax7.matshow(calW3[6, :, :])
ax8 = plt.subplot(3, 3, 8)
ax8.matshow(calW3[7, :, :])
ax9 = plt.subplot(3, 3, 9)
ax9.matshow(calW3[8, :, :])
plt.show()

