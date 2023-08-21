# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from graphs.compute_tensors import compute_tensor_order_3
from singular_values.compute_svd import computeTruncatedSVD_more_positive
from scipy.linalg import svdvals , pinv
import matplotlib.pyplot as plt


N = 20
A = np.random.uniform(-1, 1, (N, N))
n = N
Un, Sn, M = computeTruncatedSVD_more_positive(A, n)
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

