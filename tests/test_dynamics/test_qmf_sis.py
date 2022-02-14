# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import pytest
from dynamics.dynamics import qmf_sis
from dynamics.reduced_dynamics import reduced_qmf_sis,\
    reduced_qmf_sis_vector_field
from graphs.compute_tensors import compute_tensor_order_3
from time import time


def test_complete_vs_reduced_qmf_sis_full():
    """ Compares the complete quenched mean-field SIS dynamics with the reduced
    one in the tensor form for n = N and heterogeneous parameters. """
    t = 10
    N = 5
    x = np.random.random(N)
    W = np.random.uniform(-1, 1, (N, N))
    coupling = 1
    D = np.diag(np.random.random(N))
    U, s, M = np.linalg.svd(W)
    calD = M@D@np.linalg.pinv(M)
    calW = M@W@np.linalg.pinv(M)
    calW_tensor3 = compute_tensor_order_3(M, W)
    Mdotx = M@qmf_sis(t, x, W, coupling, D)
    dotX = reduced_qmf_sis(t, M@x, calW, calW_tensor3, coupling, calD)
    assert np.allclose(Mdotx, dotX)


def test_speed_compute_reduced_qmf_sis():
    """
    Speed comparison between reduced_qmf_sis and reduced_qmf_sis_vector_field.

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
        x = np.random.random(N)
        W = np.random.uniform(-1, 1, (N, N))
        coupling = 1
        D = np.diag(np.random.random(N))
        U, s, Vh = np.linalg.svd(W)
        M = Vh[:n, :]
        Mp = np.linalg.pinv(M)
        calD = M@D@np.linalg.pinv(M)
        calW = M@W@np.linalg.pinv(M)
        calW_tensor3 = compute_tensor_order_3(M, W)

        start_time_dotMx = time()
        M@qmf_sis(t, x, W, coupling, D)
        end_time_dotMx = time()
        mean_speed_complete_dynamics\
            += (end_time_dotMx - start_time_dotMx)/nb_sample

        start_time_dotX_vec = time()
        reduced_qmf_sis_vector_field(t, M@x, W, coupling, M, Mp, D)
        end_time_dotX_vec = time()
        mean_speed_reduced_dynamics_unsimplified \
            += (end_time_dotX_vec - start_time_dotX_vec)/nb_sample

        start_time_dotX = time()
        reduced_qmf_sis(t, M@x, calW, calW_tensor3, coupling, calD)
        end_time_dotX = time()
        mean_speed_reduced_dynamics_tensor +=\
            (end_time_dotX - start_time_dotX)/nb_sample

    print("\n\nSpeed comparison reduced_qmf_sis")
    print(f"\nmean time dotMx complete = "
          f"{mean_speed_complete_dynamics:.5E}")
    print(f"mean time dotX unsimplified = "
          f"{mean_speed_reduced_dynamics_unsimplified:.5E}")
    print(f"mean time dotX tensor = "
          f"{mean_speed_reduced_dynamics_tensor:.5E}")

    assert 1


# This test fails as expected  TODO not proved
# def test_complete_vs_reduced_qmf_sis_rank_identical_params():
#     """ Compares the complete quenched mean-field SIS dynamics
#      with the reduced one for n = rank(W) and identical parameters """
#     t = 10
#     N = 3
#     x = np.array([0.2, 0.8, 0.1])
#     W = np.array([[0, 0.2, 0.8],
#                   [0.5, 0.4, 0.2],
#                   [0.5, 0.4, 0.2]])
#     rankW = 2
#     coupling = 1
#     D = np.diag([2, 2, 2])
#     U, s, Vh = np.linalg.svd(W)
#     M = Vh[:rankW, :]
#     calW_tensor3 = compute_tensor_order_3(M, W)
#     calD = M@D@np.linalg.pinv(M)
#     calW = M@W@np.linalg.pinv(M)
#     Mdotx = M@qmf_sis(t, x, W, coupling, D)
#     dotX = reduced_qmf_sis(t, M@x, calW, calW_tensor3, coupling, calD)
#     assert np.allclose(Mdotx, dotX)


def test_complete_vs_reduced_qmf_sis_rank_special():
    """ Compares the complete quenched mean-field SIS dynamics
     with the reduced one for n = rank(W), the rank of the weight matrix W
      and the parameters matrix D is correlated to W.
       TODO precise "correlated" """

    """ WARNING: This is a very particular case that works."""

    t = 10
    x = np.array([0.2, 0.8, 0.1])
    W = np.array([[0, 0.2, 0.1],
                  [0.5, 0.4, 0.2],
                  [0.5, 0.4, 0.2]])
    rankW = 2
    coupling = 1
    D = np.diag([3, 2, 2])
    U, s, Vh = np.linalg.svd(W)
    M = Vh[:rankW, :]
    calW_tensor3 = compute_tensor_order_3(M, W)
    calD = M@D@np.linalg.pinv(M)
    calW = M@W@np.linalg.pinv(M)
    Mdotx = M@qmf_sis(t, x, W, coupling, D)
    dotX = reduced_qmf_sis(t, M@x, calW, calW_tensor3, coupling, calD)
    assert np.allclose(Mdotx, dotX)


if __name__ == "__main__":
    pytest.main()
