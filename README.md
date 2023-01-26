# low-dimension-hypothesis
Code to generate the results of the paper [The low-dimension hypothesis of complex systems](
https://doi.org/10.48550/arXiv.2208.04848). 


### Network datasets

The network datasets used in this projects were downloaded from the [Netzschleuder network catalogue](https://networks.skewed.de) in the `.gt.zst` file format into the `graph_data/netzschleuder/` subdirectory.
We also provide the links for other network datasets in the paper.


### Python scripts


#### Analyzing singular values, rank and effective ranks

Once one or more new network datasets have been added to the `graph_data/netzschleuder/` subdirectory, their singular values, rank and effective ranks can computed by executing the following scripts in this specific order.

1. `extract_graph_properties.py` extracts various properties and add them to the file `properties/graph_properties.txt`.  For weighted graphs, this is at this stage that the keyword used to include the weights into the adjacency matrix must be chosen if the `edge property` corresponding to weights is not called `weight` in the dataset. The graph will be considered as binary if this last step is omitted.
2. `compute_singular_values.py` computes the singular values for every new network datasets.  The singular values are saved into the file `properties/singular_values/<dataset name>_singular_values.txt`.
3. `compute_effective_ranks.py` computes the rank as well as various _effective_ ranks and add them into the file `properties/effective_ranks.txt`.


#### Plotting the results

- `plot_singular_values.py` generates a figure for each network dataset in which the singular values and the effective ranks can be analyzed visually.  Figures are saved in both `.pdf` and `.png` formats.
- `plot_effective_rank_to_ratio_vs_rank_to_ratio.py` generates a figure in which the effective rank is compared the the rank of each dataset.  In each case, the rank and the effective rank are divided by the number of vertices in the graph.

#### Versions

- Python 3.6
- Numpy 1.15.4
- Scipy 1.5.4
- Matplotlib 2.2.2.
