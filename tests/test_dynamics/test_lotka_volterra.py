# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import pytest
from dynamics.dynamics import lotka_volterra
from dynamics.reduced_dynamics import reduced_lotka_volterra,\
    reduced_lotka_volterra_vector_field
from graphs.compute_tensors import compute_tensor_order_3
from time import time


def test_complete_vs_reduced_lotka_volterra_full():
    """ Compares the complete Lotka-Volterra dynamics with the reduced one
     in the tensor form for n = N and heterogeneous parameters. """
    t = 10
    N = 5
    x = np.random.random(N)
    W = np.random.random((N, N))
    coupling = 1
    D = np.diag(np.random.random(N))
    U, s, M = np.linalg.svd(W)
    calD = M@D@np.linalg.pinv(M)
    calWD_tensor3 = compute_tensor_order_3(M, W-D)
    Mdotx = M@lotka_volterra(t, x, W, coupling, D)
    dotX = reduced_lotka_volterra(t, M@x, calWD_tensor3, coupling, calD)
    assert np.allclose(Mdotx, dotX)


def test_speed_compute_reduced_lotka_volterra():
    """
    Speed comparison between reduced_lotka_volterra and
    reduced_lotka_volterra_vector_field.

    This test might file for some n and N. It is expected if, for instance,
     N is low or n is high.
    """
    t = 10
    N = 500
    n = 10
    mean_speed_complete_dynamics = 0
    mean_speed_reduced_dynamics_unsimplified = 0
    mean_speed_reduced_dynamics_tensor = 0
    nb_sample = 500
    for i in range(nb_sample):
        x = np.random.random(N)
        W = np.random.uniform(-1, 1, (N, N))
        coupling = 0.5
        D = np.diag(np.random.random(N))
        U, s, Vh = np.linalg.svd(W)
        M = Vh[:n, :]
        Mp = np.linalg.pinv(M)
        calD = M@D@np.linalg.pinv(M)
        WD = W-D/coupling
        calWD_tensor3 = compute_tensor_order_3(M, WD)

        start_time_dotMx = time()
        M@lotka_volterra(t, x, W, coupling, D)
        end_time_dotMx = time()
        mean_speed_complete_dynamics\
            += (end_time_dotMx - start_time_dotMx)/nb_sample

        start_time_dotX_vec = time()
        reduced_lotka_volterra_vector_field(t, M@x, W, coupling, M, Mp, D)
        end_time_dotX_vec = time()
        mean_speed_reduced_dynamics_unsimplified \
            += (end_time_dotX_vec - start_time_dotX_vec)/nb_sample

        start_time_dotX = time()
        reduced_lotka_volterra(t, M@x, calWD_tensor3, coupling, calD)
        end_time_dotX = time()
        mean_speed_reduced_dynamics_tensor +=\
            (end_time_dotX - start_time_dotX)/nb_sample

    print("\n\nSpeed comparison reduced_lotka_volterra")
    print(f"\nmean time dotMx complete = "
          f"{mean_speed_complete_dynamics:.10E}")
    print(f"mean time dotX unsimplified = "
          f"{mean_speed_reduced_dynamics_unsimplified:.10E}")
    print(f"mean time dotX tensor = "
          f"{mean_speed_reduced_dynamics_tensor:.10E}")

    assert mean_speed_reduced_dynamics_tensor\
        < mean_speed_reduced_dynamics_unsimplified\
        < mean_speed_complete_dynamics


if __name__ == "__main__":
    pytest.main()
