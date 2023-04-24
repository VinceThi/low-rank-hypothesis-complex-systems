# -*- coding: utf-8 -*-
# @author: Patrick Desrosiers

import numpy as np
from plots.config_rcparams import *

# Rescaled singular values
N = 2000                  # number of points (singular value indices)
x = np.linspace(1, N, N)  # domain (set of singular value indices)

# Curve 1: supra-linear decreasing function
a = 0.75
b = 0.2
y1 = (1-a*(x-1)/(N-1))**b

# Curve 2: linear decreasing function
a = 0.75
y2 = 1-a*(x-1)/(N-1)

# Curve 3: hypergeometric decreasing function
b = 0.75
c = 2.25
epsilon = np.log(8)/np.log(N)
zeta = N**epsilon
y3 = (1-(x-1)/(N-1))**(c-2)/(1+zeta*(x-1)/(N-1))**b

# Curve 4: hypergeometric decreasing function
b = 0.75
c = 2.25
epsilon = np.log(64)/np.log(N)
zeta = N**epsilon
y3b = (1-(x-1)/(N-1))**(c-2)/(1+zeta*(x-1)/(N-1))**b

# Curve 5: exponentially decreasing function
omega = 0.97
y4 = omega**(x-1)

# Plot -- linear scale
plt.figure(figsize=(8, 4))

ax1 = plt.subplot(121)
ax1.plot(x, y1, linewidth=2)
plt.text(x[len(x)-700], y1[len(x)-700]+0.015,
         "Supra-linear", rotation=-15, fontsize=10, color='C0')
# plt.text(x[len(x)-1800], y1[len(x)-1800]+0.015,
#          "Supra-linear", rotation=-13, fontsize=10, color='C0')
ax1.plot(x, y2, linewidth=2)
plt.text(x[len(x)-900], y2[len(x)-900]+0.013,
         "Linear", rotation=-35, fontsize=10, color='C1')
ax1.plot(x, y3, linewidth=2)
plt.text(x[len(x)-1200], y3[len(x)-1200]+0.023,
         "Hypergeometric", rotation=-21, fontsize=10, color='C2')
ax1.plot(x, y3b, linewidth=2)
plt.text(x[len(x)-1400], y3b[len(x)-1400]+0.018,
         "Hypergeometric", rotation=-7, fontsize=10, color='C3')
ax1.plot(x, y4, linewidth=2)
plt.text(x[len(x)-1800], y4[len(x)-1800]+0.013,
         "Exponential", rotation=0, fontsize=10, color='C4')
# plt.text(x[len(x)-1950], y4[len(x)-1950],
#          "Exponential", rotation=-85, fontsize=10, color='C4')
ax1.set_xlabel('$x$', fontsize=12)
plt.tick_params(which="both", top=False, right=False)
plt.xticks([1, N/2, N], ['1', '$N/2$', '$N$'])
plt.yticks([0.0, 0.5, 1.0], ['0.0', '0.5', '1.0'])
ax1.set_ylabel(f"Singular-value envelope $\psi(x)$")
# plt.savefig('asymptotics.pdf', bbox_inches="tight")
# plt.show()

# Plot -- log scale
ax2 = plt.subplot(122)
ax2.plot(x, y1, linewidth=2)
ax2.plot(x, y2, linewidth=2)
ax2.plot(x, y3, linewidth=2)
ax2.plot(x, y3b, linewidth=2)
ax2.plot(x, y4, linewidth=2)
ax2.set_yscale('log')
ax2.set_ylim(1e-3, 1.5e0)
plt.tick_params(which="both", top=False, right=False)
plt.legend(['$O(N)$', '$O(N)$', '$O(N^{1-\epsilon}),\,\epsilon=0.3$',
            '$O(N^{1-\epsilon}),\,\epsilon=0.6$', '$O(1)$'],
           fontsize=10, loc='lower center')
ax2.set_xlabel('$x$', fontsize=12)
plt.xticks([1, N/2, N], ['1', '$N/2$', '$N$'])
# plt.savefig('asymptotics-log.pdf', bbox_inches="tight")
plt.show()
