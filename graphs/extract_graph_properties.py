# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info>
import glob
import graph_tool.all as gt
import os
import pandas as pd
import tabulate


def to_fwf(df, fname, cols=None):
    """Custom method 'to_fwf' for Pandas

    Parameters
    ----------
    fname : str
        The path to the new file in which the hidden parameters will be written
    cols : list or array of strings, optional
        A list or an array containing the name of the columns (as strings) to
        be written in the file (if None all columns will be written).
    """

    if cols is None:
        cols = df.columns

    header = list(cols)
    header[0] = '# ' + header[0]
    content = tabulate.tabulate(df[cols].values.tolist(), header,
                                tablefmt='plain', stralign='right',
                                colalign=('left',))
    open(fname, 'w').write(content)


# Adds the custom method to Pandas.
pd.DataFrame.to_fwf = to_fwf


def analyzeGraph(networkFilename):

    theGraph = gt.load_graph(networkFilename)

    nbVertices = theGraph.num_vertices()

    nbEdges = theGraph.num_edges()

    selfloops = 'noselfloops'
    if any(gt.label_self_loops(theGraph, mark_only=True)) == 1:
        selfloops = 'selfloops'

    multiedges = 'nomultiedges'
    if any(gt.label_parallel_edges(theGraph, mark_only=True)) == 1:
        multiedges = 'multiedges'

    direction = 'undirected'
    density = 2 * nbEdges / (nbVertices * (nbVertices - 1))
    if theGraph.is_directed():
        direction = 'directed'
        density /= 2

    partite = 'unipartite'
    if gt.is_bipartite(theGraph):
        partite = 'bipartite'

    weights = 'unweighted'
    if 'Weighted' in theGraph.gp.tags:
        weights = 'weighted'

    averageDegree = density * (nbVertices - 1)

    invalidWeightTags = ['Weighted', 'Unweighted', 'Metadata', 'Multigraph']
    tags = [tag for tag in theGraph.gp.tags if tag not in invalidWeightTags]
    tags = ','.join(tag for tag in tags)
    tags = tags.replace(' ', '')

    # Other valid weight tags, in order of priority.
    validWeightTags = ['link_counts', 'count', 'value', 'fiber_count_median',
                       'connectivity', 'synapses', 'migration_2015_total',
                       'duration']
    weightTag = 'None'
    if weights == 'weighted':
        if 'weight' in theGraph.ep.keys():
            weightTag = 'weight'
        else:
            weightTags = [tag for tag in theGraph.ep.keys()
                          if tag in validWeightTags]
            if len(weightTags) > 0:
                weightTag = weightTags[0]

    return [direction, weights, partite, selfloops, multiedges, nbVertices,
            nbEdges, density, averageDegree, tags, weightTag]


def extractGraphProperties():

    graphPropFilename = 'properties/graph_properties.txt'
    if not os.path.isfile(graphPropFilename):
        header = ['name', '(un)dir', '(un)weighted', 'uni/bi-partite',
                  'selfloops', 'multiedges', 'nbVertices', 'nbEdges',
                  'density', 'averageDegree', 'tags', 'weightTag']
        graphPropDF = pd.DataFrame(columns=header)
        graphPropList = []
    else:
        header = open(graphPropFilename, 'r').readline().\
            replace('#', ' ').split()
        graphPropDF = pd.read_table(graphPropFilename, names=header,
                                    comment="#", delimiter=r"\s+")
        graphPropList = graphPropDF.values.tolist()

    for networkName in glob.glob('graphs/graph_data/netzschleuder/*.gt.zst'):
        networkName = networkName.split('.')[0].split('/')[-1]
        if not (graphPropDF['name'] == networkName).any():
            print(networkName)
            networkFilename = 'graphs/graph_data/netzschleuder/' + networkName\
                              + '.gt.zst'
            graphPropList.append([networkName] + analyzeGraph(networkFilename))

    graphPropDF = pd.DataFrame(graphPropList, columns=header)
    graphPropDF.sort_values('name', inplace=True)
    graphPropDF.reset_index(drop=True, inplace=True)
    graphPropDF.to_fwf(graphPropFilename)


if __name__ == "__main__":
    extractGraphProperties()
