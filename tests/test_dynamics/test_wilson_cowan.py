import numpy as np
import pytest
from dynamics.dynamics import wilson_cowan
from dynamics.reduced_dynamics import reduced_wilson_cowan

t = 10
N = 5
x = np.random.random(N)
W = np.random.random((N, N))
coupling = 1
a, b, c, d = 1, 0, 1, 3
U, s, Vh = np.linalg.svd(W)
S = np.diag(s)
M = Vh
L = U @ S


def test_wilson_cowan_vs_reduced_wilson_cowan():
    """ Compares the complete Wilson-Cowan dynamics with the reduced one
     for n = N. """
    Mdotx = M@wilson_cowan(t, x, W, coupling, a, b, c, d)
    dotX = reduced_wilson_cowan(t, M@x, L, M, coupling, a, b, c, d)
    assert np.allclose(Mdotx, dotX)


if __name__ == "__main__":
    pytest.main()
