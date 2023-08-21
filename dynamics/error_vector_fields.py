# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numpy.linalg import solve, norm
from scipy.optimize import least_squares
from scipy.spatial import KDTree


def mse(a, b):
    return np.mean((a - b)**2)


def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))


def rmse_nearest_neighbors(a, b):
    """
    :param a: N points in D dimensions reference array, shape (N, D)
    :param b: X points in D dimensions array, shape (X, 3)
    :return: root mean square error between two (X, D) arrays
    """
    nearest_neighbors = np.zeros(np.shape(a))
    for i, pt in enumerate(a):
        nearest_neighbors[i, :] = b[KDTree(b).query(pt)[1]]
    return rmse(a, nearest_neighbors)


def relative_error_vector_fields(M, vfield_true, vfield_approximate,
                                 args_vfield_true, args_vfield_approximate):
    """

    :param M: (nxN array) Reduction matrix
    :param vfield_true: (N-dim array) Complete vector field
    :param vfield_approximate: (n-dim array) Reduced vector field
    :param args_vfield_true: (tuple) Arguments of the complete vector field
    :param args_vfield_approximate: (tuple) Arguments of the reduced v. field

    :return: Relative error between the reduced and the complete vector fields
    """
    n, N = np.shape(M)
    diff = mse(M@vfield_true(*args_vfield_true),
               vfield_approximate(*args_vfield_approximate))
    normalization = mse(M@vfield_true(*args_vfield_true), np.zeros(n))
    numerical_zero = 1e-13
    if normalization < numerical_zero:
        if diff < numerical_zero:
            return 0
        else:
            raise ValueError("The relative error is undefined. The denominator"
                             "is 0 but not the numerator.")
    else:
        return diff/normalization


def error_vector_fields_upper_bound(x, Jx, Jy, s, M, P):
    """
    :param x: (N-dim array) point in R^N
    :param Jx: (NxN array) jacobian matrix with derivatives in x
    :param Jy: (NxN array) jacobian matrix with derivatives in y = Wx
    :param s: (N-dim array) Singular values of W
    :param M: (nxN array) Reduction matrix = V_n^T = n-truncated, transposed,
                          right singular vector of the weight matrix W
    :param P: (NxN array) Projector M^+ M = np.linalg.pinv(M)@M

    :return: (float)
    Upper bound on the error between the vector field of a N-dimensional
    dynamics on network and its least-square optimal reduction.

    The upper bound involves the triangle inequality, the induced spectral
    norm, and the submultiplicativity.
    """
    n, N = np.shape(M)
    chi = (np.eye(N) - P)@x
    # print(s[n]*norm(M@Jy, ord=2)*norm(x), s[n], norm(M@Jy, ord=2), norm(x))
    return (norm(M@Jx@chi) + s[n]*norm(M@Jy, ord=2)*norm(x))/np.sqrt(n)


def error_vector_fields_upper_bound_triangle(x, Jx, Jy, W, M, P):
    """
    :param x: (N-dim array) point in R^N
    :param Jx: (NxN array) jacobian matrix with derivatives in x
    :param Jy: (NxN array) jacobian matrix with derivatives in y = Wx
    :param W: (NxN array) Weight matrix
    :param M: (nxN array )Reduction matrix
    :param P: (NxN array) Projector M^+ M = np.linalg.pinv(M)@M

    :return: (float)
    Upper bound on the error between the vector field of a N-dimensional
    dynamics on network and its least-square optimal reduction.
    The upper bound involves only the triangle inequality.
    """
    n, N = np.shape(M)
    chi = (np.eye(N) - P)@x
    return (norm(M@Jx@chi) + norm(M@Jy@W@chi))/np.sqrt(n)


def error_vector_fields_upper_bound_induced_norm(x, Jx, Jy, W, M, P):
    """
    :param x: (N-dim array) point in R^N
    :param Jx: (NxN array) jacobian matrix with derivatives in x
    :param Jy: (NxN array) jacobian matrix with derivatives in y = Wx
    :param W: (NxN array) Weight matrix
    :param M: (nxN array )Reduction matrix
    :param P: (NxN array) Projector M^+ M = np.linalg.pinv(M)@M

    :return: (float)
    Upper bound on the error between the vector field of a N-dimensional
    dynamics on network and its least-square optimal reduction.
    The upper bound involves the triangle inequality and the induced spectral
    norm for the second term.
    """
    n, N = np.shape(M)
    chi = (np.eye(N) - P)@x
    return (norm(M@Jx@chi) +
            norm(M@Jy@W@(np.eye(N)-P), ord=2)*norm(x))/np.sqrt(n)


# -------------------------------- Errors SIS ---------------------------------
def jacobian_x_SIS(x, W, coupling, D):
    return -D - coupling*np.diag(W@x)


def jacobian_y_SIS(x, coupling):
    return coupling*(np.eye(len(x)) - np.diag(x))


def x_prime_SIS(x, W, P):
    chi = (np.eye(len(x)) - P)@x
    A = np.diag(chi)@W + np.diag(W@chi)
    b = x*(W@x) - (P@x)*(W@P@x)
    return solve(A, b)


# ------------------------------ Errors RNN -----------------------------------
def sigmoid(x):
    return 1/(1+np.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def jacobian_y_rnn(dsigmoid, coupling):
    return 4*coupling*np.diag(dsigmoid)


def derivative_sigmoid_prime_rnn(x, W, P, coupling):
    chi = (np.eye(len(x)) - P)@x
    Wchi = W@chi
    if np.allclose(Wchi, np.zeros(len(Wchi))):
        """ In this case, we can set d to 0 (or any other finite value) because
        the (n = r = rank) (r+1)-th singular value (norm of Wchi) is 0 too. """
        d = np.zeros(len(Wchi))
    else:
        d = Wchi**(-1)*(sigmoid(2*coupling*W@x)
                        - sigmoid(2*coupling*W@P@x)) / (2*coupling)
    return d


# ------------------------- Errors Wilson-Cowan -------------------------------
def jacobian_x_wilson_cowan(x, W, coupling, D, a, b, c):
    return -D - a*np.diag(sigmoid(b*(coupling*W@x-c)))


def jacobian_y_wilson_cowan(x, W, coupling, a, b, c):
    return b*coupling*(np.eye(len(x)) - a*np.diag(x)) * \
           derivative_sigmoid(b*(coupling*W@x-c))


def objective_function_wilson_cowan(x_prime, x, W, P, coupling, a, b, c):
    chi = (np.eye(len(x)) - P) @ x
    q = b*(coupling*W@x-c)
    q_tilde = b*(coupling*W@P@x-c)
    q_prime = b*(coupling*W@x_prime-c)
    delta = (1 - a*x)*sigmoid(q) - (1 - a*P@x)*sigmoid(q_tilde)
    beta = -a*chi + b*coupling*(W@chi)*(1 - a*x_prime)
    alpha = b*coupling*(W@chi)*(1 - a*x_prime)
    return alpha*sigmoid(q_prime)**2 - beta*sigmoid(q_prime) + delta


def x_prime_wilson_cowan(position, W, P, coupling, a, b, c):
    N = len(position)
    x0 = np.random.uniform(0, 1, N)
    return least_squares(objective_function_wilson_cowan, x0=x0,
                         bounds=(np.zeros(N), np.ones(N)),
                         ftol=1e-5, xtol=1e-5, gtol=1e-5,
                         args=(position, W, P, coupling, a, b, c)).x


def jacobian_y_approx_wilson_cowan(dsigmoid, coupling, b):
    """ This is only valid for a small enough constant 'a'. See
     test_dsig_prime_Jx_Jy_wilson_cowan in
     tests/test_dynamics/test_error_vector_field """
    return b*coupling*np.diag(dsigmoid)


def derivative_sigmoid_prime_wilson_cowan(x, W, P, coupling, b, c):
    """ This is only valid for a small enough constant 'a'. See
     test_dsig_prime_Jx_Jy_wilson_cowan in
     tests/test_dynamics/test_error_vector_field """
    chi = (np.eye(len(x)) - P)@x
    Wchi = W@chi + 1e-9
    # print(Wchi)
    if np.allclose(Wchi, np.zeros(len(Wchi))):
        """ In this case, we can set d to 0 (or any other finite value) because
        the (n = r = rank) (r+1)-th singular value (norm of Wchi) is 0 too. """
        d = np.zeros(len(Wchi))
    else:
        d = Wchi**(-1)*(sigmoid(b*(coupling*W@x-c))
                        - sigmoid(b*(coupling*W@P@x-c)))/(b*coupling)
    return d


# -------------------------- Errors microbial ---------------------------------
def jacobian_x_microbial(x, W, coupling, D, b, c):
    return -D + np.diag(2*b*x - 3*c*x**2 + coupling*(W@x))


def jacobian_y_microbial(x, coupling):
    return coupling*np.diag(x)


def A_microbial(x, p, c):
    return -3*c*np.diag(x - p)


def B_microbial(x, p, W, b, coupling):
    chi = x - p
    return 2*b*np.diag(chi) + coupling*np.diag(chi)@W + coupling*np.diag(W@chi)


def C_microbial(x, p, W, b, c, coupling):
    return b*(p**2 - x**2) - c*(p**3 - x**3) + coupling*(p*(W@p) - x*(W@x))


def x_prime_microbial_neglect_linear_term(A, C):
    """ We neglect the linear term Bx' in the objective function to
     get an approximation of x'. """
    CA = -C/np.diag(A)
    # Warning
    # We set the following to ensure that we are in the possible range of x_i',
    # i.e., between 0 and 1. The parameters of the dynamics must be set to
    # ensure that the trajectories are between 0 and 1, otherwise, the
    # following lines can hardly be justified.
    CA[CA < 0] = 0.01  # or 0.5
    CA[CA > 1] = 0.99  # or 0.5
    return np.sqrt(CA)


def x_prime_microbial_neglect_coupling(x, W, P, coupling, b, c):
    """ We approximate the coupling term coupling*np.diag(chi)@W
    of the objective function by coupling*np.diag(W@chi)
     to get an approximation of x'. """
    p = P@x
    chi = x - p
    Wchi = W@chi
    A = -3*c*chi
    B_approx = 2*b*chi + coupling*Wchi
    B = 2*b*np.diag(chi) + coupling*np.diag(chi)@W + coupling*np.diag(W@chi)
    C = b*(p**2 - x**2) - c*(p**3 - x**3) + coupling*(p*(W@p) - x*(W@x))
    BAC = B_approx**2 - 4*A*C
    BAC[BAC < 0] = 0.01
    BAC[BAC > 1] = 0.99
    xpp = (-B_approx + np.sqrt(BAC))/(2*A)
    xpm = (-B_approx - np.sqrt(BAC))/(2*A)

    ofp = objective_function_microbial(xpp, A, B, C)
    ofm = objective_function_microbial(xpm, A, B, C)

    xp_return = xpp
    if np.mean(ofp**2) > np.mean(ofm**2):
        xp_return = xpm

    return xp_return


def objective_function_microbial(xp, A, B, C):
    """ The coupled quadratic equations to find one zero in x'(variable xp)
    and evaluate the upper bound on the alignment error. """
    return A@(xp**2) + B@xp + C


def x_prime_microbial_optimize(A, B, C, nb_inits=1, max_nfev=1000,
                               tol=1e-4, max_x=1):
    """ The least-squares work well for small random matrices,
     but it is less accurate for the gut microbiome network """
    N = len(A[0])
    args_f = (A, B, C)
    xp_return = max_x/2*np.ones(N)
    fvec_return = objective_function_microbial(xp_return, *args_f)
    for i in range(nb_inits):
        x0 = np.random.uniform(0, 1, N)
        xp = least_squares(objective_function_microbial, x0=x0,
                           bounds=(np.zeros(N), max_x*np.ones(N)),
                           ftol=1e-5, xtol=1e-8, gtol=1e-8, max_nfev=max_nfev,
                           args=args_f).x
        fvec = objective_function_microbial(xp, *args_f)
        MSE_fvec = np.mean(fvec**2)
        print(f"MSE[objective_function_microbial(x', *args)] = {MSE_fvec}")
        if np.all(np.abs(fvec) < tol):
            # print("All the elements of the evaluated objective"
            #       " function is smaller than the tolerance.")
            xp_return = xp
            break
        elif MSE_fvec < np.mean(fvec_return**2):
            # np.any(np.abs(fvec_return) > tol):
            # print("At least one element of the evaluated objective"
            #       " function is greater than the minimum.")
            xp_return = xp
            fvec_return = fvec

            continue

    return xp_return
