# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import scipy.linalg as la
from plots.plot_singular_values import plot_singular_values
from graphs.get_real_networks import get_economic_weight_matrix
from singular_values.compute_effective_ranks import *

networkName = "non_financial_institution04-Jan-2001"
# "AT_2008", "CY_2015", "EE_2010", "PT_2009", "SI_2016",
#  "financial_institution07-Apr-1999", "households_04-Sep-1998",
# "households_09-Jan-2002", "non_financial_institution04-Jan-2001"
singularValuesFilename = 'properties/' + networkName \
                         + '_singular_values.txt'

W = get_economic_weight_matrix(networkName)
N = len(W[0])
singularValues = la.svdvals(W)

with open(singularValuesFilename, 'wb') as singularValuesFile:
    singularValuesFile.write('# Singular values\n'.encode('ascii'))
    np.savetxt(singularValuesFile, singularValues)


print(computeEffectiveRanks(singularValues, networkName, N))

plot_singular_values(singularValues, effective_ranks=0)

