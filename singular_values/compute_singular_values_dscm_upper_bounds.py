# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


def upper_bound_singvals_sparser(N, i, gamma):
    """ The upper bounds on the singular values of the directed soft
     configuration model for <A_ij> < gamma/(1 + gamma) with gamma in (0,1)."""
    return N*gamma**i/(1 - gamma)


def upper_bound_singvals_denser(N, i, omega):
    """ The upper bounds on the singular values of the directed soft
     configuration model for <A_ij> > omega/(1 + omega) with omega > 1. """
    return N*(i == 1) + N*omega**(1-i)/(omega - 1)
