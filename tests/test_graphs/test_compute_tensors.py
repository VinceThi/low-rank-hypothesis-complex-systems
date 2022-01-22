# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.compute_tensors import *
from time import time
import pytest


def test_compute_tensor_order_3():
    N = 30
    n = 3
    W = np.random.random((N, N))
    M = np.random.random((n, N))
    calW_tensordot = compute_tensor_order_3_tensordot(M, W)
    calW_loop = compute_tensor_order_3(M, W)
    assert np.allclose(calW_tensordot, calW_loop)


def test_speed_compute_tensor_order_3():
    """ Speed comparison between compute_tensor_order_3,
     compute_tensor_order_3_nojit and compute_tensor_order_3_tensordot.

    Approach with nested tensordots and einsum:
    See https://stackoverflow.com/questions/35838037/efficient-reduction-of-multiple-tensors-in-python
    it's faster to vectorize with tensordot before taking the einsum. However,
    this approach takes too much memory and won't even compute for large
    n and N.

    Approach with nested for loops:
    We decorate the functions with Numba @jit(nopython=True, parallel=True)
    and prange instead of range which might decrease the speed for small n, N
    but will significantly improve the speed for the high n, N as shown in
    this test.
    """
    N = 200
    n = 5

    W = np.random.random((N, N))
    M = np.random.random((n, N))

    start_time_tensordot = time()
    compute_tensor_order_3_tensordot(M, W)
    end_time_tensordot = time()
    time_tensordot = end_time_tensordot - start_time_tensordot
    print("\n\nSpeed comparison compute_tensor_order_3")
    print(f"\ntime tensordot = {time_tensordot}")

    start_time_loop_nojit = time()
    compute_tensor_order_3_nojit(M, W)
    end_time_loop_nojit = time()
    time_loop_nojit = end_time_loop_nojit - start_time_loop_nojit
    print(f"time loop nojit = {time_loop_nojit}")

    start_time_loop_jit = time()
    compute_tensor_order_3(M, W)
    end_time_loop_jit = time()
    time_loop_jit = end_time_loop_jit - start_time_loop_jit
    print(f"time loop jit = {time_loop_jit}")

    assert time_loop_jit < time_loop_nojit < time_tensordot


# def test_compute_tensor_order_3_parameters():
#     N = 3
#     n = 2
#     D = np.random.random((N, N))
#     M = np.random.random((n, N))
#     Mp = pinv(M)
#     calT = np.tensordot(np.tensordot((M@D).T, Mp, axes=0), Mp, axes=0)
#     calD_tensordot = np.einsum("iuiviw->uvw", calT)
#     calD_loop = compute_tensor_order_3_parameters(M, D)
#     assert np.allclose(calD_tensordot, calD_loop)


def test_compute_tensor_order_4():
    N = 3
    n = 2
    W = np.random.random((N, N))
    M = np.random.random((n, N))
    Mp = pinv(M)
    calT = np.tensordot(np.tensordot(M.T, np.tensordot(Mp, Mp, axes=0),
                                     axes=0), W@Mp, axes=0)
    calW_tensordot = np.einsum("iuiviwix->uvwx", calT)
    calW_loop = compute_tensor_order_4(M, W)

    assert np.allclose(calW_loop, calW_tensordot)


def test_product_tensor_vector():
    n = 100
    X = np.random.random(n)
    calW3 = np.random.random((n, n, n))

    start_time_einsum = time()
    product_einsum = np.einsum("uvw,v,w->u", calW3, X, X)
    end_time_einsum = time()
    print("\n\nSpeed comparison in test_product_tensor_vector")
    print("\ncomputation time einsum = ", end_time_einsum-start_time_einsum)

    start_time_loop = time()
    product_loop = np.zeros(n)
    for mu in range(n):
        element_mu = 0
        for nu in range(n):
            for tau in range(n):
                element_mu += calW3[mu, nu, tau] * X[nu] * X[tau]
        product_loop[mu] = element_mu
    end_time_loop = time()
    print("computation time loop = ", end_time_loop - start_time_loop)

    assert np.allclose(product_einsum, product_loop)


if __name__ == "__main__":
    pytest.main()
