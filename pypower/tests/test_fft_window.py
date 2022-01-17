import os
import time
import tempfile

import numpy as np
from matplotlib import pyplot as plt
from mockfactory import Catalog

from pypower import CatalogFFTPower, MeshFFTWindow, CatalogFFTWindow, PowerSpectrumWindowMatrix, ParticleMesh, mpi, setup_logging
from pypower.fft_window import get_correlation_function_tophat_derivative

from test_fft_power import data_fn, randoms_fn


import time

class MemoryMonitor(object):
    """
    Class that monitors memory usage and clock, useful to check for memory leaks.

    >>> with MemoryMonitor() as mem:
            '''do something'''
            mem()
            '''do something else'''
    """
    def __init__(self, pid=None):
        """
        Initalize :class:`MemoryMonitor` and register current memory usage.

        Parameters
        ----------
        pid : int, default=None
            Process identifier. If ``None``, use the identifier of the current process.
        """
        import psutil
        self.proc = psutil.Process(os.getpid() if pid is None else pid)
        self.mem = self.proc.memory_info().rss / 1e6
        self.time = time.time()
        msg = 'using {:.3f} [Mb]'.format(self.mem)
        print(msg, flush=True)

    def __enter__(self):
        """Enter context."""
        return self

    def __call__(self, log=None):
        """Update memory usage."""
        mem = self.proc.memory_info().rss / 1e6
        t = time.time()
        msg = 'using {:.3f} [Mb] (increase of {:.3f} [Mb]) after {:.3f} [s]'.format(mem,mem-self.mem,t-self.time)
        if log:
            msg = '[{}] {}'.format(log, msg)
        print(msg, flush=True)
        self.mem = mem
        self.time = t

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        self()


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


def test_fft():
    from pmesh.pm import ParticleMesh, RealField
    boxsize, nmesh = [1000.]*3, [64]*3
    pm = ParticleMesh(BoxSize=boxsize, Nmesh=nmesh, dtype='c16', comm=mpi.COMM_WORLD)
    rfield = RealField(pm)
    shape = rfield.value.shape
    #rfield[...] = 1.
    rfield[...] = np.random.uniform(0., 1., size=shape)
    cfield = rfield.r2c().value
    #print(cfield[0,0,0])
    from numpy import fft
    ref = fft.fftn(rfield.value)/np.prod(shape)
    assert np.allclose(cfield, ref)

    a = np.arange(10)
    b = 2 + np.arange(10)[::-1]
    a = np.concatenate([a, np.zeros_like(a)], axis=0)
    b = np.concatenate([b, np.zeros_like(b)], axis=0)
    n = a.size
    c = np.zeros_like(a)
    for ii in range(len(c)):
        for ib, bb in enumerate(b):
            wii = ii if ii <= n // 2 else ii - n
            wii += ib
            if 0 <= wii < n: c[ii] += bb * a[wii]

    test = fft.irfft(fft.rfft(a) * fft.rfft(b).conj()).conj()
    assert np.allclose(test, c)

    with MemoryMonitor() as mem:
        pm = ParticleMesh(BoxSize=boxsize, Nmesh=nmesh, dtype='c16', comm=mpi.COMM_WORLD)
        rfield = RealField(pm)


def test_fft_window():

    boxsize = 2000.
    nmesh = 64
    kedges = np.linspace(0., 0.1, 7)
    ells = (0, 2)
    resampler = 'tsc'
    interlacing = 2
    dtype = 'f8'
    cdtype = 'c16'
    boxcenter = np.array([1e6,0.,0.])[None,:]
    data = Catalog.load_fits(data_fn)
    randoms = Catalog.load_fits(randoms_fn)

    for catalog in [data, randoms]:
        catalog['Position'] += boxcenter
        catalog['Weight'] = catalog.ones()


    for los in ['x', 'firstpoint', 'endpoint']:

        power = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'],
                                boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=kedges, position_type='pos', dtype=dtype)

        edgesin = np.linspace(0.1, 0.11, 3)
        window = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power, position_type='pos', dtype=dtype)

        with tempfile.TemporaryDirectory() as tmp_dir:
            fn = data.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            window.save(fn)
            window = CatalogFFTWindow.load(fn)

        windowc = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power, position_type='pos', dtype=cdtype)
        #print(windowc.poles.value/window.poles.value)
        assert np.allclose(windowc.poles.value, window.poles.value, rtol=0.5)
        if window.los_type == 'global':
            #print(windowc.wedges.value/window.wedges.value)
            assert np.allclose(windowc.wedges.value, window.wedges.value, rtol=0.5)

        window1 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin[:2], projsin=window.poles.projsin, ells=(0,), power_ref=power, position_type='pos', dtype=dtype)
        window2 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin[1:], projsin=window.poles.projsin, ells=(0,), power_ref=power, position_type='pos', dtype=dtype)
        windowc = window1.concatenate_x(window1, window2)
        assert np.allclose(windowc.poles.value, window.poles.value[:,:len(window.poles.xout[0])])

        window1 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, projsin=window.poles.projsin[:1], power_ref=power.poles, position_type='pos', dtype=dtype)
        window2 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, projsin=window.poles.projsin[1:], power_ref=power.poles, position_type='pos', dtype=dtype)
        windowc = window1.concatenate_proj(window1, window2)
        assert np.allclose(windowc.poles.value, window.poles.value)

        window.poles.resum_input_odd_wide_angle()
        if window.los_type == 'global':
            window.wedges.resum_input_odd_wide_angle()
            #window_global_poles = window.poles

        window.poles.rebin_x(factorout=2)
        assert len(window.poles.xout[0]) == (len(kedges) - 1)//2

        if window.los_type == 'global':
            window = MeshFFTWindow(edgesin=edgesin, power_ref=power, periodic=True, dtype=dtype)


if __name__ == '__main__':

    setup_logging()
    #test_deriv()
    #test_fft()
    test_fft_window()
