# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_networks import *
from plots.config_rcparams import *
import json

path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "simulations/simulations_data/"

""" Microbial """
graph_str = "gut"
A = get_microbiome_weight_matrix(graph_str)
N = len(A[0])
N_arange = np.arange(1, N, 1)

# Max x vs. Px approx
path_upper_bound_xPx = "2022_07_31_10h04min31sec_1000_samples_max_x_Px" \
                       "_upper_bound_RMSE_vector_field_microbial_gut.json"
with open(path_str+f"microbial_data/vector_field_errors/" +
          path_upper_bound_xPx) as json_data:
    error_upper_bound_array_xPx = json.load(json_data)
mean_upper_bound_error_xPx = np.mean(error_upper_bound_array_xPx, axis=0)
mean_log10_upper_bound_error_xPx = np.log10(mean_upper_bound_error_xPx)

# Least-squares
path_upper_bound_lstq = "2022_07_31_20h21min21sec_10_samples_optimization_" \
                        "upper_bound_RMSE_vector_field_microbial_gut.json"
with open(path_str+f"microbial_data/vector_field_errors/" +
          path_upper_bound_lstq) as json_data:
    error_upper_bound_array_lstq = json.load(json_data)
mean_upper_bound_error_lstq = np.mean(error_upper_bound_array_lstq, axis=0)
mean_log10_upper_bound_error_lstq = np.log10(mean_upper_bound_error_lstq)
print(mean_upper_bound_error_lstq)

fig = plt.figure(figsize=(4, 4))
ax = plt.subplot(111)
ax.scatter(np.array([1, 50, 200, 400, 600, 800, 830]),
           mean_upper_bound_error_lstq,
           s=15, color=deep[2], label="Least-squares method")
ax.plot(N_arange, mean_upper_bound_error_xPx,
        color=deep[1], label="$x$ or $Px$ approximation")

plt.xlabel('Dimension $n$')
ticks = ax.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])
plt.xlim([-0.01*N, 880])
plt.ylabel("Average upper bounds on $\mathcal{E}(x)$")
ymin = 10**(-3)
ymax = 100
plt.ylim([ymin, ymax])
ax.legend(loc="lower center", frameon=True, edgecolor="#e0e0e0")
ax.set_yscale('log')
plt.tick_params(axis='y', which='both', left=True,
                right=False, labelbottom=False)
plt.tight_layout()
plt.show()
