# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import pytest
from dynamics.dynamics import microbial
from dynamics.reduced_dynamics import reduced_microbial
from graphs.compute_tensors import compute_tensor_order_3,\
    compute_tensor_order_4


def test_complete_vs_reduced_microbial_full():
    """ Compares the complete microbial dynamics with the reduced
    one in the tensor form for n = N and heterogeneous parameters. """
    t = 10
    N = 50
    x = np.random.random(N)
    W = np.random.uniform(-1, 1, (N, N))
    coupling = 3
    a, b, c = 5, 13, 10/3
    D = 30*np.diag(np.random.random(N))
    U, s, M = np.linalg.svd(W)
    calD = M@D@np.linalg.pinv(M)
    calM = np.sum(M, axis=1)
    calW_tensor3 = compute_tensor_order_3(M, W)
    calM_tensor3 = compute_tensor_order_3(M, np.eye(N))
    calM_tensor4 = compute_tensor_order_4(M, np.eye(N))
    Mdotx = M@microbial(t, x, W, coupling, D, a, b, c)
    dotX = reduced_microbial(t, M@x, calW_tensor3, coupling,
                             calD, calM, calM_tensor3, calM_tensor4, a, b, c)
    assert np.allclose(Mdotx, dotX)


if __name__ == "__main__":
    pytest.main()
