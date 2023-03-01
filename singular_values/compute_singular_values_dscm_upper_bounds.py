# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def singvals_infinite_sum(a, b, k):
    return np.linalg.norm(a**k)*np.linalg.norm(b**k)


""" Unweighted directed soft configuration model bounds in two regimes """


def upper_bound_singvals_infinite_sum_sparser(i, alpha, beta, tol):
    """ The upper bounds (infinite sum) on the singular values of the
     directed soft configuration model for 0 < <A_ij> < 1/2 with
     gamma in (0,1). """
    ell_k = singvals_infinite_sum(alpha, beta, i)
    upper_bound = ell_k
    k = i + 1
    while ell_k > tol:
        ell_k = singvals_infinite_sum(alpha, beta, k)
        upper_bound += ell_k
        k += 1
    return upper_bound


def upper_bound_singvals_sparser(N, i, gamma):
    """ The upper bounds on the singular values of the directed soft
     configuration model for <A_ij> < gamma/(1 + gamma) with gamma in (0,1)."""
    return N*gamma**i/(1 - gamma)


def upper_bound_singvals_infinite_sum_denser(i, alpha, beta, tol):
    """ The upper bounds (infinite sum) on the singular values of the
     directed soft configuration model for 1/2 < <A_ij> < 1.
     "i" is an index between 1 and len(alpha). """
    m_k = singvals_infinite_sum(1/alpha, 1/beta, i)
    upper_bound = len(alpha)*(i == 1) + m_k
    k = i + 1
    while m_k > tol:
        m_k = singvals_infinite_sum(1/alpha, 1/beta, k)
        upper_bound += m_k
        k += 1
    return upper_bound


def upper_bound_singvals_denser(N, i, omega):
    """ The upper bounds on the singular values of the directed soft
     configuration model for <A_ij> > omega/(1 + omega) with omega > 1. """
    return N*(i == 1) + N*omega**(1-i)/(omega - 1)


""" Weighted directed soft configuration model"""


def upper_bound_singvals_infinite_sum_weighted(i, y, z, tol):
    """ The upper bounds (infinite sum) on the singular values of the weighted
     directed soft configuration model. """
    n_k = singvals_infinite_sum(y, z, i)
    upper_bound = n_k
    k = i + 1
    while n_k > tol:
        n_k = singvals_infinite_sum(y, z, k)
        upper_bound += n_k
        k += 1
    return upper_bound


def upper_bound_singvals_exponential_weighted(N, i, tau):
    """ The upper bounds on the singular values of the weighted directed soft
     configuration model for <W_ij> < tau/(1 - tau) with tau in (0,1)."""
    return N*tau**i/(1 - tau)
