# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.compute_tensors import *
import pytest


def test_compute_tensor_order_3():
    N = 3
    n = 2
    W = np.random.random((N, N))
    M = np.random.random((n, N))

    calW_test = compute_tensor_order_3(M, W)

    # A slower but equivalent way of computing the tensor
    calW_true = np.zeros((n, n, n))
    Mp = np.linalg.pinv(M)
    WMp = W @ Mp
    for mu in range(n):
        for nu in range(n):
            for tau in range(n):
                element_munutau = 0
                for i in range(N):
                    element_munutau += M[mu, i] * Mp[i, nu] * WMp[i, tau]
                calW_true[mu, nu, tau] += element_munutau

    assert np.allclose(calW_test, calW_true)


def test_product_tensor_vector():
    n = 2
    X = np.random.random(n)
    calW3 = np.random.random((n, n, n))

    product_test = np.einsum("uvw,v,w->u", calW3, X, X)

    product_true = np.zeros(n)
    for mu in range(n):
        element_mu = 0
        for nu in range(n):
            for tau in range(n):
                element_mu += calW3[mu, nu, tau] * X[nu] * X[tau]
        product_true[mu] = element_mu

    assert np.allclose(product_test, product_true)


if __name__ == "__main__":
    pytest.main()
