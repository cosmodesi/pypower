import os
import time
import tempfile

import numpy as np
from matplotlib import pyplot as plt
from mockfactory import Catalog

from pypower import CatalogFFTPower, MeshFFTWindow, CatalogFFTWindow, PowerSpectrumFFTWindowMatrix, ParticleMesh, mpi, setup_logging
from pypower.fft_window import get_correlation_function_tophat_derivative

from test_fft_power import data_fn, randoms_fn



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
    kedges = np.array([0.1, 0.12, 0.14])

    for ell in [0, 1, 2, 3, 4, 5]:

        fana = get_correlation_function_tophat_derivative(kedges, ell=ell)
        assert len(fana) == len(kedges) - 1
        sep = np.logspace(-3, 3, 1000)
        fnum = get_correlation_function_tophat_derivative(kedges, k=1./sep[::-1], ell=ell)
        assert len(fnum) == len(kedges) - 1

        if plot:
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
        for ia, aa in enumerate(a):
            wii = ii if ii <= n // 2 else ii - n
            wii += ia
            if 0 <= wii < n: c[ii] += aa * b[wii]

    test = fft.irfft(fft.rfft(a).conj() * fft.rfft(b))
    assert np.allclose(test, c)

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

    for los in ['x', 'firstpoint', 'endpoint']:

        data = Catalog.load_fits(data_fn)
        randoms = Catalog.load_fits(randoms_fn)

        for catalog in [data, randoms]:
            catalog['Position'] += boxcenter
            catalog['Weight'] = catalog.ones()

        edges = kedges
        if los in ['x']:
            edges = (kedges, np.linspace(-1., 1., 4))

        power = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'],
                                boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=edges, position_type='pos', dtype=dtype)

        with tempfile.TemporaryDirectory() as tmp_dir:
            fn = data.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            power.save(fn)
            power = CatalogFFTPower.load(fn)

        edgesin = np.linspace(0.1, 0.11, 3)
        window = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power, position_type='pos', dtype=dtype)

        with tempfile.TemporaryDirectory() as tmp_dir:
            fn = data.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            window.save(fn)
            window = CatalogFFTWindow.load(fn)
            window.save(fn)

        power_f4 = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'],
                                boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=kedges, position_type='pos', dtype='f4').poles
        window_f4 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power_f4, position_type='pos')
        assert window_f4.dtype.itemsize == 4

        windowc = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power, position_type='pos', dtype=cdtype)
        #print(windowc.poles.value/window.poles.value)
        #assert np.allclose(windowc.poles.value, window.poles.value, rtol=0.5)
        #if window.attrs['los_type'] == 'global':
        #    print(windowc.wedges.value/window.wedges.value)
        #    assert np.allclose(windowc.wedges.value, window.wedges.value, rtol=0.5)

        window1 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin[:2], projsin=window.poles.projsin, ells=(0,), power_ref=power, position_type='pos', dtype=dtype)
        window2 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin[1:], projsin=window.poles.projsin, ells=(0,), power_ref=power, position_type='pos', dtype=dtype)
        windowc = window1.concatenate_x(window1, window2)
        assert np.allclose(windowc.poles.value, window.poles.value[:,:len(window.poles.xout[0])])

        randoms['Position'][0] += boxsize
        window1 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, projsin=window.poles.projsin[:1], power_ref=power.poles, wrap=True, position_type='pos', dtype=dtype)
        window2 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, projsin=window.poles.projsin[1:], power_ref=power.poles, wrap=True, position_type='pos', dtype=dtype)
        windowc = window1.concatenate_proj(window1, window2)
        assert np.allclose(windowc.poles.value, window.poles.value)

        randoms['Position'][0] -= boxsize
        projsin = [(ell, 0) for ell in ells]
        if los in ['firstpoint', 'endpoint']: projsin += [(ell, 1) for ell in range(1, max(ells)+2, 2)]
        alpha = data.sum('Weight')/randoms.sum('Weight')
        window_noref = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, projsin=projsin, edges=edges, ells=ells, los=los,
                                        boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos', wnorm=power.poles.wnorm/alpha**2, dtype=dtype)
        assert np.allclose(window_noref.poles.value, window.poles.value)

        randoms['Position'] = mpi.gather_array(randoms['Position'], root=0, mpicomm=catalog.mpicomm)
        randoms['Weight'] = mpi.gather_array(randoms['Weight'], root=0, mpicomm=catalog.mpicomm)
        window_root = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power, position_type='pos', dtype=dtype, mpiroot=0)
        assert np.allclose(window_root.poles.value, window.poles.value)

        if randoms.mpicomm.rank == 0:
            window_root = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power, position_type='pos', dtype=dtype, mpicomm=mpi.COMM_SELF)
            assert np.allclose(window_root.poles.value, window.poles.value)

        window.poles.resum_input_odd_wide_angle()
        if window.attrs['los_type'] == 'global':
            window.wedges.resum_input_odd_wide_angle()
            #window_global_poles = window.poles

        window.poles.rebin_x(factorout=2)
        assert len(window.poles.xout[0]) == (len(kedges) - 1)//2

        if window.attrs['los_type'] == 'global':
            window = MeshFFTWindow(edgesin=(0.03, 0.04), power_ref=power, periodic=True)
            assert not np.allclose(window.poles.value, 0.)


if __name__ == '__main__':

    setup_logging()
    #test_deriv()
    #test_fft()
    test_fft_window()
