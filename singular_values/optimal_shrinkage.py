# -*- coding: utf-8 -*-
# @author: Vincent Thibeault
"""
This is a Python traduction of the Matlab script optimal_shrinkage.m from
Gavish, Matan and Donoho, David. (2016). Code Supplement for
"Optimal Shrinkage of Singular Values". Stanford Digital Repository.
Available at: http://purl.stanford.edu/kv623gt2817

We also merge the code of Ben Erichson at https://github.com/erichson/optht,
which is a Python traduction of the Matlab script optimal_SVHT_coef.m from

Donoho, David and Gavish, Matan. (2014). Code supplement to "The Optimal Hard
Threshold for Singular Values is 4/sqrt(3)". Stanford Digital Repository.
Available at: https://purl.stanford.edu/vg705qn9070

See also the thesis of P.O. Perry : https://arxiv.org/pdf/0909.3052.pdf

Finally, we correct the error for the optimal singular value shrinkage for
operator norm loss with the Theorem 3.1 of

W. Leeb, "Optimal singular value shrinkage for operator norm loss:
 Extending to non-square matrices", Stat. Probab. Lett., 186, 2022.

We modified the code in our style and made few modifications.
"""

import numpy as np
import logging
import warnings
from singular_values.marchenko_pastur_pdf import median_marcenko_pastur

# Create logger
log = logging.getLogger(__name__)


def optimal_SVHT_coef_sigma_known(beta):
    """ Implement Equation (11) of Gavish, Donoho (2014).
    Ref: https://github.com/erichson/optht """
    return np.sqrt(2*(beta + 1) + (8*beta) /
                   (beta + 1 + np.sqrt(beta**2 + 14*beta + 1)))


def optimal_SVHT_coef_sigma_unknown(beta):
    """ Implement Equation (5) of Gavish, Donoho (2014).
    Ref: https://github.com/erichson/optht  """
    return 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43


def optimal_threshold(singvals, beta, sigma=None, target_rank=False):
    """
    Compute optimal hard threshold for singular values.
    Off-the-shelf method for determining the optimal singular value truncation
    (hard threshold) for matrix denoising.
    The method gives the optimal location both in the case of the known or
    unknown noise level.

    :param singvals: array_like
    The singular values for the given input matrix.
    :param beta: scalar or array_like
        Scalar determining the aspect ratio of a matrix, i.e., ``beta = m/n``,
        where ``m >= n``.  Instead the input matrix can be provided and the
        aspect ratio is determined automatically.
    :param sigma:  real, optional
        Noise level if known.
    :param target_rank: bool
        If False (default), the function return the optimal threshold. Else,
         it returns the optimal target rank.
    :return: cutoff or k : float and int respectively
        Optimal threshold or optimal target rank.

    Usage
    -----
    See https://github.com/erichson/optht (note that we inverse the first args)

    Notes
    -----
    Code is adapted from Matan Gavish and David Donoho, see [1] and
    Ben Erichson, see [2].

    References
    ----------
    [1] Gavish, Matan, and David L. Donoho.
        "The optimal hard threshold for singular values is 4/sqrt(3)"
        IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.
        http://arxiv.org/abs/1305.5870
    [2] Ben Erichson, https://github.com/erichson/optht
    """
    # Compute aspect ratio of the input matrix
    if isinstance(beta, np.ndarray):
        m = min(beta.shape)
        n = max(beta.shape)
        beta = m / n

    if beta < 0 or beta > 1:
        raise ValueError('Parameter `beta` must be in (0,1].')

    if sigma is None:
        log.info('Sigma unknown.')
        # Approximate ``w(beta)``
        coef_approx = optimal_SVHT_coef_sigma_unknown(beta)
        log.info(f'Approximated `w(beta)` value: {coef_approx}')
        # Compute the optimal ``w(beta)``
        coef = optimal_SVHT_coef_sigma_known(beta) / \
            np.sqrt(median_marcenko_pastur(beta))
        cutoff = coef * np.median(singvals)
        if np.allclose(cutoff, 0) and np.allclose(np.median(singvals), 0):
            warnings.warn("The noise was predicted to be zero, because"
                          " the median of the singular values is 0."
                          " The rank of the original matrix is returned.")
            return np.nan  # len(singvals[singvals > 1e-13])
    else:
        log.info('Sigma known.')
        # Compute optimal ``w(beta)``
        coef = optimal_SVHT_coef_sigma_known(beta)
        cutoff = coef * np.sqrt(len(singvals)) * sigma
    log.info(f'`w(beta)` value: {coef}')
    log.info(f'Cutoff value: {cutoff}')
    if target_rank:
        # Compute and return rank
        greater_than_cutoff = np.where(singvals > cutoff)
        if greater_than_cutoff[0].size > 0:
            k = np.max(greater_than_cutoff) + 1
        else:
            k = 0
        log.info(f'Target rank: {k}')
        return k
    else:
        return cutoff


def inverse_asymptotic_singvals(y, beta):
    """ Equation (8) in Optimal Shrinkage of Singular Values of Gavish and
    Donoho 2017. It is the inverse of Equation (15) which would yield four
    solutions. This is the case where the positive square roots are considered.
    """
    inverse = np.emath.sqrt(
        0.5*((y**2-beta-1) + np.emath.sqrt((y**2-beta-1)**2 - 4*beta)))
    return np.real(inverse*(y >= 1+np.sqrt(beta)))


def optimal_shrinker_frobenius(y, beta):
    """ Equation (7) in Optimal Shrinkage of Singular Values of Gavish and
        Donoho 2017 """
    return np.sqrt(np.maximum(((y**2-beta-1)**2 - 4*beta), np.zeros(len(y))))/y


def optimal_shrinker_operator(y, beta):
    """ Equation (9) in Optimal Shrinkage of Singular Values of Gavish and
        Donoho 2017 for square matrices and the correction of William Leeb,
        "Optimal singular value shrinkage for operator norm loss:
        Extending to non-square matrices", 2022. """
    t = inverse_asymptotic_singvals(y, beta)
    eta = t*np.sqrt((t**2 + np.min(1, beta))/(t**2 + np.max(1, beta)))
    return np.maximum(eta, np.zeros(len(y)))


def optimal_shrinker_nuclear(y, beta):
    """ Equation (10) in Optimal Shrinkage of Singular Values of Gavish and
        Donoho 2017 """
    with np.errstate(divide='ignore'):
        f = (inverse_asymptotic_singvals(y, beta)**4
             - np.sqrt(beta)*inverse_asymptotic_singvals(y, beta)*y - beta) / \
            ((inverse_asymptotic_singvals(y, beta)**2)*y)
    return np.maximum(np.zeros(len(y)), f)


def optimal_shrinkage(singvals, beta, loss, sigma=None):
    """
    Original documentation with the following modifications:
    - format, typo, notation adjustments
    - Python traduction of the example

    Perform optimal shrinkage (w.r.t one of a few possible losses) on data
    singular values, when the noise is assumed white, and the noise level is
    known or unknown.

    :param singvals: vector of data singular values, obtained by running svd
                on the data matrix
    :param beta: aspect ratio m/n of the m-by-n matrix whose singular values
                are given
    :param loss: loss function for which the shrinkage should be optimal
                presently implemented: 'frobenius' (Frobenius or square
                                                    Frobenius norm loss = MSE)
                                       'nuclear' (nuclear norm loss)
                                       'operator'  (operator norm loss)
    :param sigma: (optional) noise standard deviation (of each entry of the
                noise matrix) if this argument is not provided, the noise
                level is estimated from the data.
    :return:
    shrinked_singvals: vector of singular values after
                       performing optimal shrinkage

    Usage:
      Given an m-by-n matrix Y known to be low rank and observed in white noise
      with zero mean, form a denoised matrix Xhat by:

      U, S, Vh = np.linalg.svd(Y)
      shrink_s = optimal_shrinkage(S, m/n, 'operator')
      Xhat = U@np.diag(shrink_s)@Vh

      where you can replace 'op' with one of the other losses.
      if the noise level sigma is known, in the third line use instead
          y = optimal_shrinkage(s, m/n, 'operator', sigma);

    ---------------------------------------------------------------------------
    Authors: Matan Gavish and David Donoho <lastname>@stanford.edu, 2013
             Vincent Thibeault, vincent.thibeault.1@ulaval.ca, 2022

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along
    with this program.  If not, see <http://www.gnu.org/licenses/>.
    ---------------------------------------------------------------------------
    """
    assert beta <= 1
    assert beta > 0

    if sigma is None:
        sigma = np.median(singvals)/np.sqrt(median_marcenko_pastur(beta))

    if np.allclose(sigma, 0) and np.allclose(np.median(singvals), 0):
        warnings.warn("The noise was predicted to be zero, because"
                      " the median of the singular values is 0. The original"
                      " singular values are returned.")
        return np.nan   # singvals
    else:
        y = singvals/sigma
        x = inverse_asymptotic_singvals(y, beta)

        print(sigma)

        if loss == 'frobenius':
            shrinked_singvals = sigma*optimal_shrinker_frobenius(y, beta)
        elif loss == 'nuclear':
            shrinked_singvals = sigma*optimal_shrinker_nuclear(y, beta)
            shrinked_singvals[
                np.where((x**4 - np.sqrt(beta)*x*y - beta) <= 0)] = 0
            # See Eq. (10) of the paper    ^
        elif loss == 'operator':  # with the correction of W. Leeb, 2022
            shrinked_singvals = sigma*optimal_shrinker_operator(y, beta)
        else:
            raise ValueError("Unknown loss."
                             " The Frobenius norm/loss 'frobenius',"
                             " the nuclear norm/loss 'nuclear' and"
                             " the operator (spectral) norm/loss 'operator' "
                             " can be chosen.")

        return shrinked_singvals
