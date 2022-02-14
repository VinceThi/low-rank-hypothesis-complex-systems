# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.compute_tensors import *
from time import time
import pytest

test_validity = True
test_speed = False


if test_validity:

    def test_compute_tensor_order_3():
        N = 30
        n = 3
        W = np.random.random((N, N))
        M = np.random.random((n, N))
        calW_tensordot = compute_tensor_order_3_tensordot(M, W)
        calW_loop = compute_tensor_order_3(M, W)
        assert np.allclose(calW_tensordot, calW_loop)


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

elif test_speed:
    def test_einsum_vs_dot_vs_matmul():
        """
        About the speed of einsum, see
        https://stackoverflow.com/questions/18365073/
        why-is-numpys-einsum-faster-than-numpys-built-in-functions
        See the particular case of np.dot vs. einsum, where einsum is slower.
        In this test we compare einsum, dot, matmul, and @.
        """

        arr_2D = np.arange(500**2, dtype=np.double).reshape(500, 500)

        start_time_einsum = time()
        product_einsum = np.einsum('ij,jk', arr_2D, arr_2D)
        end_time_einsum = time()
        print("\n\nSpeed comparison np.einsum, np.dot, np.matmul, and @")
        print("\ncomputation time einsum= ", end_time_einsum-start_time_einsum)

        start_time_dot = time()
        product_dot = np.dot(arr_2D, arr_2D)
        end_time_dot = time()
        print("computation time dot = ", end_time_dot - start_time_dot)

        start_time_matmul = time()
        product_matmul = np.matmul(arr_2D, arr_2D)
        end_time_mathmul = time()
        print("computation time matmul= ", end_time_mathmul-start_time_matmul)

        start_time_at = time()
        product_at = arr_2D@arr_2D
        end_time_at = time()
        print("computation time at = ", end_time_at - start_time_at)

        assert np.allclose(product_einsum, product_dot) and \
            np.allclose(product_einsum, product_matmul) and \
            np.allclose(product_einsum, product_at)


    def test_speed_compute_tensor_order_3():
        """ Speed comparison between compute_tensor_order_3,
         compute_tensor_order_3_nojit and compute_tensor_order_3_tensordot.

        Approach with nested tensordots and einsum:
        See https://stackoverflow.com/questions/35838037/
        efficient-reduction-of-multiple-tensors-in-python
       it's faster to vectorize with tensordot before taking the einsum. Yet,
        this approach takes too much memory and won't even compute for large
        n and N.

        Approach with nested for loops:
        We decorate the functions with Numba @jit(nopython=True, parallel=True)
        and prange instead of range which might decrease the speed for
        small n, N but will significantly improve the speed for the high n, N
        as shown in this test.
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


    def test_speed_product_tensor_vector():
        n = 100

        mean_speed_einsum = 0
        mean_speed_matmul = 0
        mean_speed_at = 0
        mean_speed_loop = 0

        nb_sample = 50
        for i in range(nb_sample):

            X = np.random.uniform(-5, 5, n)
            calW3 = np.random.uniform(-1, 1, (n, n, n))

            start_time_einsum = time()
            np.einsum("uvw,v,w->u", calW3, X, X)
            end_time_einsum = time()
            mean_speed_einsum += (end_time_einsum-start_time_einsum)/nb_sample

            start_time_matmul = time()
            np.matmul(np.matmul(calW3, X), X)
            end_time_matmul = time()
            mean_speed_matmul += (end_time_matmul-start_time_matmul)/nb_sample

            start_time_at = time()
            (calW3@X)@X
            end_time_at = time()
            mean_speed_at += (end_time_at - start_time_at)/nb_sample

            start_time_loop = time()
            product_loop = np.zeros(n)
            for mu in range(n):
                element_mu = 0
                for nu in range(n):
                    for tau in range(n):
                        element_mu += calW3[mu, nu, tau]*X[nu]*X[tau]
                product_loop[mu] = element_mu
            end_time_loop = time()
            mean_speed_loop += (end_time_loop - start_time_loop)/nb_sample
            # print("computation time loop = ",end_time_loop - start_time_loop)

        print("\n\nSpeed comparison einsum, matmul, @, loop")
        print(f"\nmean time einsum = "
              f"{mean_speed_einsum:.5E}")
        print(f"mean time matmul = "
              f"{mean_speed_matmul:.5E}")
        print(f"mean time @ = "
              f"{mean_speed_at:.5E}")
        print(f"mean time loop = "
              f"{mean_speed_loop:.5E}")
        assert 1
        # np.allclose(product_einsum, product_loop) and \
        # np.allclose(product_einsum, product_matmul) and \
        # np.allclose(product_einsum, product_at)

else:
    print("Nothing to be tested.")

if __name__ == "__main__":
    pytest.main()
