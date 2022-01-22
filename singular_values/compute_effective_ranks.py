# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info>
#          Vincent Thibeault
import glob
import numpy as np
import os
import pandas as pd
import sys
import tabulate

# Effective rank based on the definition
# by Gavish and Donoho (doi:10.1109/TIT.2014.2323359).
sys.path.insert(1, 'src/optht')
import optht  # https://github.com/erichson/optht


def to_fwf(df, fname, cols=None):
    """Custom method 'to_fwf' for Pandas

    Parameters
    ----------
    df : dataframe
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


def computeERank(singularValues):
    """Effective rank based on the definition using spectral entropy
     (https://ieeexplore.ieee.org/document/7098875
      and doi:10.1186/1745-6150-2-2). """
    normalizedSingularValues = singularValues / np.sum(singularValues)
    return np.exp(-np.sum(normalizedSingularValues
                          * np.log(normalizedSingularValues)))


def findElbowPosition(singularValues):
    """Effective rank based on the elbow method."""
    # Coordinates of the diagonal line y = 1 - x  using ax + by + c = 0.
    a, b, c = 1, 1, -1

    # Define normalized axis with first SV at (0,1) and last SV at (1,0).
    x = np.linspace(0, 1, num=len(singularValues))
    y = (singularValues - np.min(singularValues)) /\
        (np.max(singularValues) - np.min(singularValues))

    # See https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # Line_defined_by_an_equation
    distanceToDiagonal = np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

    # Returns the index of the largest distance (rank must be larger than 0).
    return np.argmax(distanceToDiagonal) + 1


def computeEffectiveRankEnergyRatio(singularValues, threshold=0.9):
    """Effective rank based on the energy ratio."""
    normalizedCumulSquaredSingularValues = np.cumsum(np.square(singularValues))
    normalizedCumulSquaredSingularValues /= \
        normalizedCumulSquaredSingularValues[-1]
    return np.argmax(normalizedCumulSquaredSingularValues > threshold) + 1


def computeStableRank(singularValues):
    return np.sum(singularValues*singularValues) / np.max(singularValues)**2


def computeEffectiveRanks(singularValues, matrixName, size):
    """
    Computes the rank and various effective ranks and add them to a dataframe.
    :param singularValues: Singular values of a matrix
    :param matrixName: (str) Name of the matrix for which we have computed the
                             singular values
    :param size: Size of the matrix
    :return Table with the name of the matrix, its size, rank, and effective
            ranks (G-D rank, erank, elbow, energy ratio, stable rank).
    """
    header = ['Name', 'Size', 'Rank', 'G-D rank',
              'erank', 'Elbow', 'Energy ratio', 'Stable Rank']
    properties = [[matrixName, size,
                  len(singularValues[singularValues > 1e-13]),
                  optht.optht(1, sv=singularValues, sigma=None),
                  computeERank(singularValues),
                  findElbowPosition(singularValues),
                  computeEffectiveRankEnergyRatio(singularValues,
                                                  threshold=0.9),
                  computeStableRank(singularValues)]]
    return tabulate.tabulate(properties, headers=header)


def computeEffectiveRanksManyNetworks():
    """ Computes the rank andvarious effective ranks
    and add them to a dataframe for many networks. """

    graphPropFilename = 'properties/graph_properties.txt'
    header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
    graphPropDF = pd.read_table(graphPropFilename, names=header,
                                comment="#", delimiter=r"\s+")
    graphPropDF.set_index('name', inplace=True)

    effectiveRanksFilename = 'properties/effective_ranks.txt'
    if not os.path.isfile(effectiveRanksFilename):
        header = ['name', 'nbVertices', 'rank', 'rankGD',
                  'erank', 'elbow', 'energyRatio', 'stableRank']
        effectiveRanksDF = pd.DataFrame(columns=header)
        effectiveRanksList = []
    else:
        header = open(effectiveRanksFilename, 'r')\
            .readline().replace('#', ' ').split()
        effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                         comment="#", delimiter=r"\s+")
        effectiveRanksList = effectiveRanksDF.values.tolist()

    for networkName in glob.glob('properties/singular_values/'
                                 '*_singular_values.txt'):

        networkName = networkName.split('_singular_values')[0].split('/')[-1]

        if not effectiveRanksDF['name'].str.contains(networkName).any():
            singularValuesFilename = 'properties/singular_values/'\
                                     + networkName + '_singular_values.txt'
            singularValues = np.loadtxt(singularValuesFilename)
            print(networkName)
            effectiveRanks = [networkName,
                              graphPropDF.loc[networkName]['nbVertices'],
                              int(len(singularValues)),
                              optht.optht(1, sv=singularValues, sigma=None),
                              computeERank(singularValues),
                              findElbowPosition(singularValues),
                              computeEffectiveRankEnergyRatio(singularValues,
                                                              threshold=0.9),
                              computeStableRank(singularValues)]
            effectiveRanksList.append(effectiveRanks)

    effectiveRanksDF = pd.DataFrame(effectiveRanksList, columns=header)
    effectiveRanksDF.sort_values('name', inplace=True)
    effectiveRanksDF.reset_index(drop=True, inplace=True)
    effectiveRanksDF.to_fwf(effectiveRanksFilename)


if __name__ == "__main__":
    computeEffectiveRanksManyNetworks()
