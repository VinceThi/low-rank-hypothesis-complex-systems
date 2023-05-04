# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from scipy.special import gamma
from plots.config_rcparams import *

""" --------- This scripts is about the asymptotics of the erank ---------- """


""" Exponential decrease of the singular values """


def plot_erank_asymptotics_expo():
    alpha = 0.45
    omega = 0.5
    x = np.arange(1, 100, 1)

    plt.figure(figsize=(4, 4))
    plt.plot(x, x*alpha**x, label=f"$N\\alpha^N$ ($\\alpha = {alpha}$)")
    plt.plot(x, omega**x, label=f"$\\omega^N$ ($\\omega = {omega}$)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc=1)
    plt.xlabel("$N$")
    plt.ylim([10**(-14), 1])
    plt.tick_params(which="both", top=False, right=False)
    plt.show()


""" Hypergeometric decrease of the singular values """


def g(b, c, d):
    return gamma(1 - b)*gamma(c - 1)*(d**b)/gamma(c - b)


def expo(N, epsilon, d, bt, bb, ct, cb, delta):
    bd = (1 - delta)*bb
    cd = (1 - delta)*cb + 2*delta
    gt = g(bt, ct, d)
    gb = g(bd, cd, d)
    return np.exp((gb*N**(epsilon*(bt - bd)))/(gt*delta) - 1/delta)


def t0(N, epsilon, d, bt, bb, ct, cb, delta):
    e = expo(N, epsilon, d, bt, bb, ct, cb, delta)
    return N**(1 - (bb + 1)*epsilon)*e


def t1(N, epsilon, d, bt, bb, ct, cb, delta):
    e = expo(N, epsilon, d, bt, bb, ct, cb, delta)
    bd = (1 - delta)*bb
    return N**(epsilon*(2*bt - bd - 1))*e


def t2(N, epsilon, d, bt, bb, ct, cb, delta):
    e = expo(N, epsilon, d, bt, bb, ct, cb, delta)
    db = bt - bb
    return N**(1 - (1 - 2*db)*epsilon + bb*delta*epsilon)*e


def t3(N, epsilon, d, bt, bb, ct, cb, delta):
    e = expo(N, epsilon, d, bt, bb, ct, cb, delta)
    db = bt - bb
    return N**(1 - 2*(1 - db)*epsilon + bb*delta*epsilon)*e


def plot_erank_asymptotics_hypergeo():
    epsilon = 0.1
    d = 5
    bt, bb = 0.55, 0.45  # b^*, b_*
    delta = 0.1  # between 0 and 1 + (1 − 2b∗)/b∗
    print(f"b^* = {bt} must be between 0.5 and 1")
    assert bt > 1/2
    assert bt < 1
    print(f"delta must be lower than 1 + (1 - 2b^*)/b_* = {1 + (1 - 2*bt)/bb}")
    assert delta < 1 + (1 - 2*bt)/bb
    print(f"b_* must be above 2b^* - 1 = {2*bt - 1}")
    assert bb > 2*bt - 1
    print(f"2b^* - (1-delta)b_* - 1 = {2*bt - (1-delta)*bb - 1} "
          f"must be smaller than 0")
    assert 2*bt - (1-delta)*bb - 1 < 0

    ct, cb = 2.2, 2.1
    x = np.arange(10, 10000010, 10)

    plt.figure(figsize=(5, 5))
    plt.title(f"$\\epsilon = {epsilon},\,\, \\delta = {delta},\,\,"
              f" b_* = {bb},\,\, b^* = {bt}$", fontsize=10)
    plt.plot(x, x, label="$N$", linewidth=3, color=dark_grey)
    plt.plot(x, expo(x, epsilon, d, bt, bb, ct, cb, delta),
             label="$e^{y(N)}$")
    # \\epsilon(b^* - (1-\\delta)b_{*})
    plt.plot(x, t0(x, epsilon, d, bt, bb, ct, cb, delta),
             label="$N^{1-(b_*+1)\\epsilon}e^{y(N)}$")
    plt.plot(x, t1(x, epsilon, d, bt, bb, ct, cb, delta),
             label="$N^{(2b^* - b_{\\delta} - 1)\\epsilon}e^{y(N)}$")
    plt.plot(x, t2(x, epsilon, d, bt, bb, ct, cb, delta),
             label="$N^{1 - (1 - 2\\Delta b)\\epsilon + "
                   "b_{*}\\delta\\epsilon}e^{y(N)}$")
    plt.plot(x, t3(x, epsilon, d, bt, bb, ct, cb, delta),
             label="$N^{1 - 2(1 - \\Delta b)\\epsilon + "
                   "b_{*}\\delta\\epsilon}e^{y(N)}$")

    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc=2)
    plt.xlabel("$N$")
    plt.ylim([10**(-6), 10**9])
    plt.tick_params(which="both", top=False, right=False)
    plt.show()


if __name__ == "__main__":
    # plot_erank_asymptotics_expo()
    plot_erank_asymptotics_hypergeo()
