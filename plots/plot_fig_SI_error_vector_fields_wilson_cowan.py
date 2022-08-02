# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_networks import *
from plots.config_rcparams import *
import json

path_str = "C:/Users/thivi/Documents/GitHub/" \
           "low-rank-hypothesis-complex-systems/" \
           "simulations/simulations_data/"

""" Wilson-Cowan """
graph_str = "celegans_signed"
A = get_connectome_weight_matrix(graph_str)
N = len(A[0])
N_arange = np.arange(1, N, 1)

# Small 'a' approx
path_upper_bound_smalla = "2022_07_27_13h18min36sec_1000_samples_small_a" \
                          "_approximation_upper_bound_RMSE_vector_field_" \
                          "wilson_cowan_celegans_signed.json"
with open(path_str+f"wilson_cowan_data/vector_field_errors/" +
          path_upper_bound_smalla) as json_data:
    error_upper_bound_array_smalla = json.load(json_data)
mean_upper_bound_error_smalla = np.mean(error_upper_bound_array_smalla, axis=0)
mean_log10_upper_bound_error = np.log10(mean_upper_bound_error_smalla)

# Max x vs. Px approx
# path_upper_bound_xPx = "2022_07_27_11h21min14sec_10_samples_max_x_Px_" \
#                        "approximation_upper_bound_RMSE_vector_field_" \
#                        "wilson_cowan_celegans_signed.json"
path_upper_bound_xPx = "2022_03_25_14h45min47sec_1000_samples_upper_bound" \
                       "_RMSE_vector_field_wilson_cowan_celegans_signed.json"
with open(path_str+f"wilson_cowan_data/vector_field_errors/" +
          path_upper_bound_xPx) as json_data:
    error_upper_bound_array_xPx = json.load(json_data)
mean_upper_bound_error_xPx = np.mean(error_upper_bound_array_xPx, axis=0)
mean_log10_upper_bound_error_xPx = np.log10(mean_upper_bound_error_xPx)

# Least-squares
path_upper_bound_lstq = "2022_07_28_12h48min27sec_10_samples_least_squares" \
                        "_upper_bound_RMSE_vector_field_wilson_cowan" \
                        "_celegans_signed.json"
with open(path_str+f"wilson_cowan_data/vector_field_errors/" +
          path_upper_bound_lstq) as json_data:
    error_upper_bound_array_lstq = json.load(json_data)
mean_upper_bound_error_lstq = np.mean(error_upper_bound_array_lstq, axis=0)
mean_log10_upper_bound_error_lstq = np.log10(mean_upper_bound_error_lstq)

fig = plt.figure(figsize=(4, 4))
ax = plt.subplot(111)
ax.scatter(np.array([1, 5, 9, 50, 100, 150, 200, 250, 280, 296]),
           mean_upper_bound_error_lstq,
           s=15, color=deep[2], label="Least-squares method")
ax.plot(N_arange, mean_upper_bound_error_smalla,
        color=deep[0], label="Small $a$ approximation")
ax.plot(N_arange, mean_upper_bound_error_xPx,
        color=deep[1], label="$x$ or $Px$ approximation")

plt.xlabel('Dimension $n$')
ticks = ax.get_xticks()
ticks[ticks.tolist().index(0)] = 1
plt.xticks(ticks[ticks > 0])
plt.xlim([-0.01*N, 310])
plt.ylabel("Average upper bounds on $\mathcal{E}(x)$")
ymin = 0.5*10**(-5)
ymax = 200
plt.ylim([ymin, ymax])
ax.legend(loc="lower center", frameon=True, edgecolor="#e0e0e0")
ax.set_yscale('log')
plt.tick_params(axis='y', which='both', left=True,
                right=False, labelbottom=False)
plt.tight_layout()
plt.show()
