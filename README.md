# The low-rank hypothesis of complex systems
Code to generate the results of the paper [The low-dimension hypothesis of complex systems](
https://doi.org/10.48550/arXiv.2208.04848). 


### Network datasets

The network datasets used in this projects were downloaded from the [Netzschleuder network catalogue](https://networks.skewed.de) in the `.gt.zst` file format into the `graph_data/netzschleuder/` subdirectory.
We also provide the links for other network datasets in the paper.


### Python scripts

#### Dynamics

The `dynamics/` folder contains the code for the integration of various nonlinear dynamics on graphs and to compute the alignment errors.

- `dynamics.py` contains the vector fields of the high-dimensional nonlinear dynamics on graphs.
- `reduced_dynamics.py` contains two alternative functions for each reduced vector fields: one in tensor form and one that uses the functions of the vector fields defined `dynamics.py`.
- `error_vector_fields.py` contains every functions needed to compute the error upper bound in the paper (e.g., the Jacobian, $x'$, ...).
- `integrate.py` contains integrators for dynamics on graphs: rk4, dopri45, a function that uses `scipy.integrate.ode`. Ultimately, we use `scipy.integrate.solve_ivp` with BDF in `simulation/`.

#### Graphs

The `graphs/` folder contains the code to extract real networks, generate random graphs (e.g, $S^1$ model), among others.

- `get_real_networks.py` allows to extract the networks that are not Netzschleuder with functions such as `get_{type of network}(graph_name)`.
- `generate_random_graphs.py` allows to generate graphs from various random graphs with fixed parameters or random parameters.
- `generate_S1_random_graphs.py` contains the function `s1_model`, a generator of graphs from the random geometric model $S^1$.
- `compute_tensors.py` contains different functions to compute the tensors arising in the dimension reduction of the paper.
- `split_weight_nws`, by Gabriel Eilerstein, contains the function `unpack` that allows to separate layer weights for convolutional and fully connected layers in convolutional neural networks from [nws](https://github.com/gabrieleilertsen/nws).
- `extract_graph_properties.py` and `extract_graph_properties_non_netzschleuder.ipynb` extracts various properties and add them to the file `properties/graph_properties.txt` and `graph_data/graph_properties_augmented.txt` respectively.  For weighted graphs, this is at this stage that the keyword used to include the weights into the adjacency matrix must be chosen if the `edge property` corresponding to weights is not called `weight` in the dataset. The graph will be considered as binary if this last step is omitted.

#### Simulations for the dynamics

The `simulation/` folder contains the code to compute the alignment errors, the trajectories and the equilibrium points for the different dynamics of the paper.

- `errors_qmf_sis.py` allows to generate the data to get Fig.~3a.
- `errors_wilson_cowan.py` allows to generate the data to get Fig.~3b.
- `errors_microbial.py` allows to generate the data to get Fig.~3c.
- `errors_rnn.py` allows to generate the data to get Fig.~3d.
- `bifurcations_qmf_sis.py` allows to generate the data to get Fig.~3e.
- `bifurcations_wilson_cowan.py` allows to generate the data to get Fig.~3f.
- `bifurcations_microbial.py` allows to generate the data to get Fig.~3g.
- `trajectories_rnn.py` allows to generate the data to get Fig.~3h.


#### Singular values, ranks and effective ranks

Once one or more new network datasets have been added to the `graph_data/netzschleuder/` subdirectory, their singular values, rank and effective ranks can computed by executing the following scripts in this specific order.

- `compute_singular_values.py` computes the singular values for every new network datasets.  The singular values are saved into the file `properties/singular_values/<dataset name>_singular_values.txt`.
- `compute_effective_ranks.py` computes the rank as well as various _effective_ ranks and add them into the file `properties/effective_ranks.txt`.


#### Plotting the results
- `plot_fig_1b_drosophila_network.py`, `plot_fig_1d_drosophila_singular_values.py`, `plot_fig_1e_effective_rank_vs_rank_scatterplot.py` and `plot_fig_1fn_effective_rank_to_dimension_ratio_densities.py` generate subfigures for Fig. 1, which is then assembled on Inkscape.
- `plot_fig_3_error_vector_fields.py` generates Fig. 3 with the alignment errors and the bifurcations/trajectories.
- `plot_singular_values.py` contains the functions to generate scree plots or histograms for the singular values of one or many networks. The function `plot_singular_values` gives the scree plots with the effective ranks, the cumulative explained variance and the y axis in log if desired.


#### Unit tests

Unit tests are in the folder `tests/` and are seperated in three: `tests/test_dynamics/`, `tests/test_graphs/`, and `tests/test_singular_values/`.

- The tests in `tests/test_dynamics/` ensure that the complete dynamics and the reduced dynamics coincide in tensor form at $n=N$ (scripts test_{dynamics' name}) and that the $x'$ and Jacobian matrices (found analytically or numerically) to compute the upper bound on the alignment error are correct (scripts test_error_{dynamics' name}).

- The tests in `tests/test_graphs/test_compute_tensors.py` ensure that the tensors arising in the dimension reduction are well computed numerically. Simple speed tests for different methods (einsum, matmul, loop) to compute the tensors are also available.

- The tests `tests/test_singular_values/test_compute_effective_ranks.py` ensure that the effective ranks thrank and shrank under different norms (frobenius, spectral/operator, nuclear) gives the correct value in a simple example of a matrix of rank 20. thrank and shrank with different norms are also compared for the drosophila connectome. The script `tests/test_singular_values/test_compute_svd.py` contains simple tests for functions introduced in  `singular_values/compute_svd.py`. Finally, the tests in `tests/test_singular_values/test_optimal_shrinkage.py` ensure that for a given matrix shrank and thrank under different norms give the same results as the Matlab scripts optimal_shrinkage.m [Gavish, Matan and Donoho, David. (2016). Code Supplement for
"Optimal Shrinkage of Singular Values". Stanford Digital Repository.
Available at: http://purl.stanford.edu/kv623gt2817] and optimal_SVHT_coef.m [Donoho, David and Gavish, Matan. (2014). Code supplement to "The Optimal Hard
Threshold for Singular Values is 4/sqrt(3)". Stanford Digital Repository.
Available at: https://purl.stanford.edu/vg705qn9070].


#### Versions

- Python 3.6
- Numpy 1.15.4
- Scipy 1.5.4
- Matplotlib 2.2.2.
