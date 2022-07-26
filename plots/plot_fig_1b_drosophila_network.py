# -​*- coding: utf-8 -*​-
# @author: Patrick Desrosiers and Vincent Thibeault

import graph_tool.all as gt
import numpy as np
import pandas as pd
import networkx as nx

# Connectome
df = pd.read_csv('drosophila_exported-traced-adjacencies-v1.1/'
                 'traced-total-connections.csv')
Graphtype = nx.DiGraph()
G_drosophila = nx.from_pandas_edgelist(df,
                                       source='bodyId_pre',
                                       target='bodyId_post',
                                       edge_attr='weight',
                                       create_using=Graphtype)

# Weighted adjacency matrix
A = nx.to_numpy_array(G_drosophila, weight='weight')
N = A.shape[0]  # number of neurons

# Reduced weighted adjacency matrix obtained by
# randomly selecting 5% of the connections
sample_size = 1085
sample = np.random.randint(N, size=sample_size)
sample = list(sample)
adj_5percent = A[np.ix_(sample, sample)]

# Visual representation using graph-tool
g = gt.Graph(directed=False)
g.add_edge_list(np.transpose(adj_5percent.nonzero()))
state = gt.minimize_nested_blockmodel_dl(g)
gt.draw_hierarchy(state, output="drosophilia_nested_mdl_5percent.pdf")
