# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import pytest
from dynamics.dynamics import kuramoto_sakaguchi, complex_kuramoto_sakaguchi
from dynamics.reduced_dynamics import reduced_kuramoto_sakaguchi,\
    reduced_kuramoto_sakaguchi_vector_field
from graphs.compute_tensors import compute_tensor_order_4
from time import time


def test_real_vs_complex_kuramoto_sakaguchi_full():
    """ Compares the real and complex Kuramoto-Sakaguchi dynamics. """
    t = 10
    N = 5
    alpha = np.pi/4
    theta = np.random.uniform(0, 2*np.pi, N)
    W = np.random.uniform(-1, 1, (N, N))
    coupling = 1
    D = np.diag(np.random.random(N))
    z = np.exp(1j*theta)
    dotz_real = 1j*z*kuramoto_sakaguchi(t, theta, W, coupling, D, alpha)
    dotz_cplx = complex_kuramoto_sakaguchi(t, z, W, coupling, D, alpha)
    assert np.allclose(dotz_real, dotz_cplx)


def test_complete_vs_reduced_kuramoto_sakaguchi_full():
    """ Compares the complete Kuramoto-Sakaguchi dynamics with the reduced one
     in the tensor form for n = N and heterogeneous parameters. """
    t = 10
    N = 5
    alpha = np.pi/4
    theta = np.random.uniform(0, 2*np.pi, N)
    z = np.exp(1j*theta)
    W = np.random.uniform(-1, 1, (N, N))
    coupling = 1
    D = np.diag(np.random.random(N))
    U, s, M = np.linalg.svd(W)
    calD = M@D@np.linalg.pinv(M)
    calW = M@W@np.linalg.pinv(M)
    calW_tensor4 = compute_tensor_order_4(M, W)
    Mdotz = M@(1j*z*kuramoto_sakaguchi(t, theta, W, coupling, D, alpha))
    dotZ = reduced_kuramoto_sakaguchi(t, M@z, calW, calW_tensor4,
                                      coupling, calD, alpha)
    assert np.allclose(Mdotz, dotZ)


def test_speed_compute_reduced_kuramoto_sakaguchi():
    """
    Speed comparison between reduced_kuramoto_sakaguchi
     and reduced_kuramoto_sakaguchi_vector_field.

    This test might file for some n and N. It is expected if, for instance,
     N is low or n is high.
    """
    t = 10
    N = 500
    n = 1
    mean_speed_complete_dynamics = 0
    mean_speed_reduced_dynamics_unsimplified = 0
    mean_speed_reduced_dynamics_tensor = 0
    nb_sample = 500
    for i in range(nb_sample):
        alpha = np.random.uniform(0, 2*np.pi, 1)
        theta = np.random.uniform(0, 2*np.pi, N)
        z = np.exp(1j * theta)
        W = np.random.uniform(-1, 1, (N, N))
        coupling = 1
        D = np.diag(np.random.random(N))
        U, s, Vh = np.linalg.svd(W)
        M = Vh[:n, :]
        Mp = np.linalg.pinv(M)
        calD = M@D@np.linalg.pinv(M)
        calW = M@W@np.linalg.pinv(M)
        calW_tensor3 = compute_tensor_order_4(M, W)

        start_time_dotMx = time()
        M@complex_kuramoto_sakaguchi(t, z, W, coupling, D, alpha)
        end_time_dotMx = time()
        mean_speed_complete_dynamics\
            += (end_time_dotMx - start_time_dotMx)/nb_sample

        start_time_dotX_vec = time()
        reduced_kuramoto_sakaguchi_vector_field(t, M@z, W, coupling,
                                                M, Mp, D, alpha)
        end_time_dotX_vec = time()
        mean_speed_reduced_dynamics_unsimplified \
            += (end_time_dotX_vec - start_time_dotX_vec)/nb_sample

        start_time_dotX = time()
        reduced_kuramoto_sakaguchi(t, M@z, calW, calW_tensor3,
                                   coupling, calD, alpha)
        end_time_dotX = time()
        mean_speed_reduced_dynamics_tensor +=\
            (end_time_dotX - start_time_dotX)/nb_sample

    print("\n\nSpeed comparison reduced_kuramoto_sakaguchi")
    print(f"\nmean time dotMx complete = "
          f"{mean_speed_complete_dynamics:.10E}")
    print(f"mean time dotX unsimplified = "
          f"{mean_speed_reduced_dynamics_unsimplified:.10E}")
    print(f"mean time dotX tensor = "
          f"{mean_speed_reduced_dynamics_tensor:.10E}")

    assert 1

# def test_complete_vs_reduced_kuramoto_sakaguchi_rank_identical_param():
#    """ Compares the complete Kuramoto-Sakaguchi dynamics with the reduced one
#      for n = rank(W), the rank of the weight matrix W which is a star of
#      4 nodes. """
#     t = 10
#     N = 4
#     alpha = np.pi/4
#     theta = np.random.uniform(0, 2*np.pi, N)
#     z = np.exp(1j*theta)
#     W = np.array([[0, 1, 1, 1],
#                   [1, 0, 0, 0],
#                   [1, 0, 0, 0],
#                   [1, 0, 0, 0]])
#     rankW = 2
#     coupling = 1
#     D = np.diag([0.5, 0.5, 0.5, 0.5])
#     U, s, Vh = np.linalg.svd(W)
#     M = Vh[:rankW, :]
#     calD = M@D@np.linalg.pinv(M)
#     calW = M@W@np.linalg.pinv(M)
#     calW_tensor4 = compute_tensor_order_4(M, W)
#     Mdotz = M@(1j*z*kuramoto_sakaguchi(t, theta, W, coupling, D, alpha))
#     dotZ = reduced_kuramoto_sakaguchi(t, M@z, calW, calW_tensor4,
#                                       coupling, calD, alpha)
#     print(Mdotz, dotZ)
#     print(np.linalg.norm(M@W@(np.eye(N)-np.linalg.pinv(M)@M)))
#     print(np.linalg.norm(M@D@(np.eye(N)-np.linalg.pinv(M)@M)))
#     print(np.linalg.norm(W @ (np.eye(N) - np.linalg.pinv(M) @ M)))
#     print(np.linalg.norm(D @ (np.eye(N) - np.linalg.pinv(M) @ M)))
#     print(np.linalg.norm(Mdotz - dotZ))
#
#     assert np.allclose(Mdotz, dotZ)


if __name__ == "__main__":
    pytest.main()
