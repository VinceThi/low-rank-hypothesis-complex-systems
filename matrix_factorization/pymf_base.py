# Authors: Christian Thurau and Vincent Thibeault
# License: BSD 3 Clause
"""
PyMF base class used in (almost) all matrix factorization methods.
V. Thibeault: I simplified the script for my own usage.
"""
import numpy as np
import logging
import logging.config
import scipy.sparse


_EPS = np.finfo(float).eps

# # I tried to jit but it did not work, because
# # TypeError: cannot subclass from a jitclass (SNMF inheritates from PyMFBase)
# from numba import jitclass, int64, float64
# spec = [('_EPS', float64),
#         ('data', float64[:]),
#         ('_num_bases', float64[:]),
#         ('W_init', float64[:]),
#         ('H_init', float64[:]),
#         ('_data_dimension)', int64),
#         ('_num_samples', int64)]
# @jitclass(spec)


class PyMFBase:
    """
    PyMF Base Class. Does nothing useful apart from providing
    some basic methods.
    """

    _EPS = _EPS  # some small value

    def __init__(self, data, W_init=None, H_init=None, num_bases=4):
        """
        """

        def setup_logging():
            # create logger
            self._logger = logging.getLogger("pymf")

            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s")

                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()

        # set variables
        self.data = data
        self._num_bases = num_bases

        self.W_init = W_init
        self.H_init = H_init

        self._data_dimension, self._num_samples = self.data.shape

    def residual(self):
        """ Returns the residual in % of the total amount of data
        Returns
        -------
        residual : float
        """
        res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
        total = 100.0 * res / np.sum(np.abs(self.data))
        return total

    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank
        approximation given by WH. Minimizing the Fnorm is the most common
        optimization criterion for matrix factorization methods.
        Returns:
        -------
        frobenius norm: F = ||data - WH||
        """
        # check if W and H exist
        if hasattr(self, 'H') and hasattr(self, 'W'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:, :] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt(
                    np.sum((self.data[:, :] - np.dot(self.W, self.H)) ** 2))
        else:
            err = None

        return err

    def _init_w(self):
        """ Initalize W to random values in [0,1] if W_init is None.
            Else it initializes to the given W_init matrix.
        """
        if self.W_init is None:
            # add a small value, otherwise nmf and related methods get
            # into trouble as
            # they have difficulties recovering from zero.
            self.W = np.random.random(
                (self._data_dimension, self._num_bases)) + 10 ** -4
        else:
            self.W = self.W_init

    def _init_h(self):
        """ Initalize H to random values in [0,1] if H_init is None.
            Else it initializes to the given H_init matrix.
        """
        if self.H_init is None:
            # add a small value, otherwise nmf and related methods get
            # into trouble as
            # they have difficulties recovering from zero.
            self.H = np.random.random(
                (self._num_bases, self._num_samples)) + 10**(-4)
        else:
            self.H = self.H_init

    def _update_h(self):
        """ Overwrite for updating H.
        """
        pass

    def _update_w(self):
        """ Overwrite for updating W.
        """
        pass

    def _converged(self, i):
        """
        If the optimization of the approximation is below the machine precision
        return True.
        Parameters
        ----------
            i   : index of the update step
        Returns
        -------
            converged : boolean
        """
        derr = np.abs(self.ferr[i] - self.ferr[i - 1]) / self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100,  # show_progress=False,
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data

        Parameters
        ----------
        niter : int
                number of iterations.
        # show_progress : bool
        #         print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k].

        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """

        # if show_progress:
        #     self._logger.setLevel(logging.INFO)
        # else:
        #     self._logger.setLevel(logging.ERROR)

        # create W and H if they don't already exist
        # -> any custom initialization to W, H should be done before
        if not hasattr(self, 'W') and compute_w:
            self._init_w()

        if not hasattr(self, 'H') and compute_h:
            self._init_h()

        # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(niter)

        for i in range(niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()

            if compute_err:
                self.ferr[i] = self.frobenius_norm()
                # self._logger.info(
                #     'FN: %s (%s/%s)' % (self.ferr[i], i + 1, niter))
            # else:
            #     # self._logger.info('Iteration: (%s/%s)' % (i + 1, niter))

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
