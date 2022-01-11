import os
import time
import tempfile

import numpy as np
from matplotlib import pyplot as plt

from pypower import setup_logging
from pypower.fft_window import get_correlation_function_tophat_derivative


def test_deriv(plot=False):
    kedges = np.linspace(0.001, 1., 4)

    for ell in [0, 2, 4]:

        fana = get_correlation_function_tophat_derivative(kedges, ell=ell)
        assert len(fana) == len(kedges) - 1
        sep = np.logspace(-3, 3, 1000)
        fnum = get_correlation_function_tophat_derivative(kedges, k=1./sep[::-1], ell=ell)
        assert len(fnum) == len(kedges) - 1

        ax = plt.gca()
        ax.plot(sep, fana[1](sep), label='analytic')
        ax.plot(sep, fnum[1](sep), label='numerical')
        ax.legend()
        ax.set_xscale('log')
        plt.show()


if __name__ == '__main__':

    setup_logging()
    #test_deriv()
