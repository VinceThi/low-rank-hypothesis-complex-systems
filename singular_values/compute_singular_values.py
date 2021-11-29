# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info>
import glob
import graph_tool.all as gt
import numpy as np
import os
import pandas as pd
import scipy.linalg as la


def computeSingularValues():
    """Computes the singular values of every graphs
     and saves them into a file."""

    graphPropFilename = 'properties/graph_properties.txt'
    header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
    graphPropDF = pd.read_table(graphPropFilename, names=header, comment="#",
                                delimiter=r"\s+")
    graphPropDF.set_index('name', inplace=True)

    for networkName in glob.glob('graph_data/netzschleuder/*.gt.zst'):

        networkName = networkName.split('.')[0].split('/')[-1]
        networkFilename = 'graph_data/netzschleuder/' + networkName + '.gt.zst'
        singularValuesFilename = 'properties/singular_values/' + networkName \
                                 + '_singular_values.txt'

        if not os.path.isfile(singularValuesFilename):
            print(networkName)

            theGraph = gt.load_graph(networkFilename)

            # Checks if the graph is weighted
            #  (and extracts the internal edge property).
            weightTag = graphPropDF.loc[networkName]['weightTag']
            weights = None
            if weightTag != 'None':
                weights = theGraph.edge_properties[weightTag]

            A = gt.adjacency(theGraph, weight=weights)

            numericalZero = 1e-13
            singularValues = la.svdvals(A.toarray())
            singularValues = singularValues[singularValues > numericalZero]

            with open(singularValuesFilename, 'wb') as singularValuesFile:
                singularValuesFile.write('# Singular values\n'.encode('ascii'))
                np.savetxt(singularValuesFile, singularValues)


if __name__ == "__main__":
    computeSingularValues()
