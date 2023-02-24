# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from scipy.stats import pareto


def truncated_pareto(N, kappa_min, kappa_max, gamma):
    kappas = [val for val in kappa_min * pareto.rvs(gamma - 1, size=N)
              if val < kappa_max]
    while len(kappas) < N:
        kappas.extend([val for val
                       in kappa_min*pareto.rvs(gamma-1, size=N-len(kappas))
                       if val < kappa_max])
    return np.array(kappas)
