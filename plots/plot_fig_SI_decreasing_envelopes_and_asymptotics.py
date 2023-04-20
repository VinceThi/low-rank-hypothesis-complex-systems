# -*- coding: utf-8 -*-
# @author: Patrick Desrosiers

import numpy as np
import matplotlib.pyplot as plt

# Rescaled singular values
N=2000                  # number of points (singular value indices)
x = np.linspace(1,N,N)  # domain (set of singular value indices)

# Curve 1: supra-linear decreasing function
a=0.75
b=0.2
y1 = (1-a*(x-1)/(N-1))**b

# Curve 2: linear decreasing function
a=0.75
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
omega =0.97
y4 =omega**(x-1)

# Plot -- linear scale
plt.figure(figsize=(4,4))
plt.plot(x,y1,linewidth=2.5)
plt.plot(x,y2,linewidth=2.5)
plt.plot(x,y3,linewidth=2.5)
plt.plot(x,y3b,linewidth=2.5)
plt.plot(x,y4,linewidth=2.5)
plt.xlabel('$i$', fontsize = 12)
plt.xticks([1, N/2, N], ['1', '$N/2$', '$N$'], rotation=0)
plt.yticks([0.0,0.5,1.0], ['0.0', '0.5', '1.0'], rotation=0)
#plt.savefig('asymptotics.pdf', bbox_inches="tight")
plt.show()

# Plot -- log scale
plt.figure(figsize=(4,4))
plt.plot(x,y1,linewidth=2.5)
plt.plot(x,y2,linewidth=2.5)
plt.plot(x,y3,linewidth=2.5)
plt.plot(x,y3b,linewidth=2.5)
plt.plot(x,y4,linewidth=2.5)
plt.gca().set_yscale('log')
plt.gca().set_ylim(1e-3,2e0)
plt.legend(['$O(N)$','$O(N)$', '$O(N^{1-\epsilon}),\,\epsilon=0.3$', '$O(N^{1-\epsilon}),\,\epsilon=0.6$', '$O(1)$'],
           fontsize = 11,loc='lower center')
plt.xlabel('$i$', fontsize = 12)
plt.xticks([1, N/2, N], ['1', '$N/2$', '$N$'], rotation=0)
#plt.savefig('asymptotics-log.pdf', bbox_inches="tight")
plt.show()
