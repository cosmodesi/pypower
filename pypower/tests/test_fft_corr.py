import os
import tempfile

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo.fiducial import DESI
from mockfactory import EulerianLinearMock
from mockfactory.make_survey import RandomBoxCatalog

from pypower import MeshFFTCorr, CatalogFFTCorr, CorrelationFunctionStatistics, setup_logging


def kaiser(power, bias=1., f=1.):

    def model(s, ells):
        beta = f / bias
        toret = {}
        toret[0] = bias**2 * (1. + 2. / 3. * beta + 1. / 5. * beta**2) * power.to_xi(fftlog_kwargs={'ell': 0})(s)
        toret[2] = bias**2 * (4. / 3. * beta + 4. / 7. * beta**2) * power.to_xi(fftlog_kwargs={'ell': 2})(s)
        toret[4] = bias**2 * 8. / 35 * beta**2 * power.to_xi(fftlog_kwargs={'ell': 4})(s)
        ells = (0, 2, 4)
        return np.array([toret[ell] if ell in toret else np.zeros_like(s) for ell in ells])

    return model


def plot_correlation_function(poles, theory=None):
    ax = plt.gca()
    if theory is not None:
        theory = theory(poles.s, poles.ells)
    mask = (poles.s < 200.)
    for ill, ell in enumerate(poles.ells):
        ax.plot(poles.s[mask], poles.s[mask]**2 * poles(ell=ell)[mask], color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
        ax.plot(poles.s[mask], poles.s[mask]**2 * theory[ill][mask], linestyle='--', color='C{:d}'.format(ill))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
    if poles.mpicomm.rank == 0:
        plt.show()


def test_corr_statistic():

    for dtype in ['f8', 'c16']:

        edges = np.linspace(0., 0.2, 11)
        modes = (edges[:-1] + edges[1:]) / 2.
        nmodes = np.arange(modes.size)
        ells = (0, 2, 4)
        corr = [ill * np.arange(nmodes.size, dtype='f8') + 0.1j * (np.arange(nmodes.size, dtype='f8') - 5) for ill in ells]
        corr = CorrelationFunctionStatistics(edges, modes, corr, nmodes, ells, statistic='multipole')
        corr_ref = corr.copy()
        corr.rebin(factor=2)
        assert corr.corr.shape[1] == corr_ref.corr.shape[1] // 2  # poles are first dimension
        k = (corr_ref.modes[0][::2] * corr_ref.nmodes[::2] + corr_ref.modes[0][1::2] * corr_ref.nmodes[1::2]) / (corr_ref.nmodes[::2] + corr_ref.nmodes[1::2])
        assert np.allclose(corr.s, k)
        assert np.allclose(corr.sedges, np.linspace(0., 0.2, 6))
        assert corr.shape == (modes.size // 2,)
        assert np.allclose(corr_ref[::2].corr_nonorm, corr.corr_nonorm)
        corr2 = corr_ref.copy()
        corr2.select((0., 0.1))
        assert np.all(corr2.modes[0] <= 0.1)
        wedges = corr_ref.to_wedges(muedges=np.linspace(-1., 1., 11))
        assert wedges.shape == corr_ref.shape + (10,)
        wedges.get_corr()

        def mid(edges):
            return (edges[:-1] + edges[1:]) / 2.

        for axis in range(corr.ndim): assert np.allclose(corr.modeavg(axis=axis, method='mid'), mid(corr.edges[axis]))

        corr2 = corr_ref + corr_ref
        assert np.allclose(corr2.corr, corr_ref.corr, equal_nan=True)
        assert np.allclose(corr2.wnorm, 2. * corr_ref.wnorm, equal_nan=True)

        corr = corr_ref.copy()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # tmp_dir = '_tests'
            # corr.mpicomm = mpicomm # to get a Barrier (otherwise the directory on root=0 may be deleted before other ranks access it)
            # fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            fn = os.path.join(tmp_dir, 'tmp.npy')
            corr.save(fn)
            test = CorrelationFunctionStatistics.load(fn)
            assert np.allclose(test.corr, corr.corr, equal_nan=True)
            fn = os.path.join(tmp_dir, 'tmp.npy')
            test.save(fn)

        corr2 = corr.copy()
        corr2.modes[0] = 1
        assert np.all(corr.modes[0] == test.modes[0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = '_tests'
            fn = os.path.join(tmp_dir, 'tmp_poles.txt')
            corr.save_txt(fn, complex=False)
            test = np.loadtxt(fn, unpack=True)
            assert np.allclose(test, [corr.nmodes, corr.modeavg(method='mid'), corr.s] + list(corr.corr.real), equal_nan=True)
            corr.save_txt(fn, complex=True)
            test = np.loadtxt(fn, unpack=True, dtype=np.complex_)
            assert np.allclose(test, [corr.nmodes, corr.modeavg(method='mid'), corr.s] + list(corr.corr), equal_nan=True)

        for complex in [False, True]:
            assert np.allclose(corr(complex=complex, return_s=True)[1], corr.get_corr(complex=complex), equal_nan=True)
            assert np.allclose(corr(complex=complex), corr.get_corr(complex=complex), equal_nan=True)
            assert np.isnan(corr(s=-1., ell=0, complex=complex))
            assert not np.isnan(corr(s=modes, complex=complex)).any()
            assert np.allclose(corr(s=[0.1, 0.2], ell=corr.ells), corr(s=[0.1, 0.2]))
            assert corr(s=[0.1, 0.2], complex=complex).shape == (len(corr.ells), 2)
            assert corr(s=[0.1, 0.2], ell=0, complex=complex).shape == (2,)
            assert corr(s=0.1, ell=0, complex=complex).shape == ()
            assert corr(s=0.1, ell=(0, 2), complex=complex).shape == (2,)
            assert np.allclose(corr(s=[0.2, 0.1], complex=complex), corr(s=[0.1, 0.2], complex=complex)[..., ::-1], atol=0)
            assert np.allclose(corr(s=[0.2, 0.1], ell=(0, 2), complex=complex), corr(s=[0.1, 0.2], ell=(2, 0), complex=complex)[::-1, ::-1], atol=0)

        edges = (np.linspace(0., 0.2, 11), np.linspace(-1., 1., 21))
        modes = np.meshgrid(*((e[:-1] + e[1:]) / 2 for e in edges), indexing='ij')
        nmodes = np.arange(modes[0].size, dtype='i8').reshape(modes[0].shape)
        corr = np.arange(nmodes.size, dtype='f8').reshape(nmodes.shape)
        corr = corr + 0.1j * (corr - 5)
        corr = CorrelationFunctionStatistics(edges, modes, corr, nmodes, statistic='wedge')
        corr_ref = corr.copy()
        corr.rebin(factor=(2, 2))
        assert corr.corr.shape[0] == corr_ref.corr.shape[0] // 2
        assert corr.modes[0].shape == (5, 10)
        assert not np.isnan(corr(0., 0.))
        assert np.isnan(corr(-1., 0.))
        corr.rebin(factor=(1, 10))
        assert corr.corr_nonorm.shape == (5, 1)
        assert np.allclose(corr_ref[::2, ::20].corr_nonorm, corr.corr_nonorm, atol=0)
        assert corr_ref[1:7:2].shape[0] == 3
        corr2 = corr_ref.copy()
        corr2.select(None, (0., 0.5))
        assert np.all(corr2.modes[1] <= 0.5)
        for axis in range(corr.ndim): assert np.allclose(corr.modeavg(axis=axis, method='mid'), mid(corr.edges[axis]))

        corr2 = corr_ref + corr_ref
        assert np.allclose(corr2.corr, corr_ref.corr)
        assert np.allclose(corr2.wnorm, 2. * corr_ref.wnorm)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # tmp_dir = '_tests'
            # corr.mpicomm = mpicomm # to get a Barrier (otherwise the directory on root=0 may be deleted before other ranks access it)
            # fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            fn = os.path.join(tmp_dir, 'tmp.npy')
            corr.save(fn)
            test = CorrelationFunctionStatistics.load(fn)
            assert np.all(test.corr == corr.corr)
            fn = os.path.join(tmp_dir, 'tmp.npy')
            test.save(fn)

        corr = corr_ref.copy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # tmp_dir = '_tests'
            fn = os.path.join(tmp_dir, 'tmp_wedges.txt')
            corr.save_txt(fn, complex=False)
            test = np.loadtxt(fn, unpack=True)
            mids = np.meshgrid(*(corr.modeavg(axis=axis, method='mid') for axis in range(corr.ndim)), indexing='ij')
            assert np.allclose([tt.reshape(corr.shape) for tt in test], [corr.nmodes, mids[0], corr.modes[0], mids[1], corr.modes[1], corr.corr.real], equal_nan=True)
            corr.save_txt(fn, complex=True)
            test = np.loadtxt(fn, unpack=True, dtype=np.complex_)
            assert np.allclose([tt.reshape(corr.shape) for tt in test], [corr.nmodes, mids[0], corr.modes[0], mids[1], corr.modes[1], corr.corr], equal_nan=True)

        for muedges in [np.linspace(-1., 1., 21), np.linspace(-1., 1., 2)]:
            edges = (np.linspace(0., 0.2, 11), muedges)
            modes = np.meshgrid(*((e[:-1] + e[1:]) / 2 for e in edges), indexing='ij')
            nmodes = np.ones(tuple(len(e) - 1 for e in edges), dtype='i8')
            corr = np.arange(nmodes.size, dtype='f8').reshape(nmodes.shape)
            corr = corr + 0.1j * (corr - 5)
            corr = CorrelationFunctionStatistics(edges, modes, corr, nmodes, statistic='wedge')
            for complex in [False, True]:
                assert np.allclose(corr(complex=complex, return_s=True, return_mu=True)[2], corr.get_corr(complex=complex), equal_nan=True)
                assert np.allclose(corr(complex=complex, return_s=True)[1], corr.get_corr(complex=complex), equal_nan=True)
                assert np.allclose(corr(complex=complex), corr.get_corr(complex=complex), equal_nan=True)
                assert not np.isnan(corr(0., 0., complex=complex))
                assert np.isnan(corr([-1.] * 5, 0., complex=complex)).all()
                assert np.isnan(corr(-1., [0.] * 5, complex=complex)).all()
                assert corr(s=[0.1, 0.2], complex=complex).shape == (2, corr.shape[1])
                assert corr(s=[0.1, 0.2], mu=[0.3], complex=complex).shape == (2, 1)
                assert corr(s=[[0.1, 0.2]] * 3, mu=[[0.3]] * 2, complex=complex).shape == (3, 2, 2, 1)
                assert corr(s=[0.1, 0.2], mu=0., complex=complex).shape == (2,)
                assert corr(s=0.1, mu=0., complex=complex).shape == ()
                assert corr(s=0.1, mu=[0., 0.1], complex=complex).shape == (2,)
                assert np.allclose(corr(s=[0.2, 0.1], mu=[0.2, 0.1], complex=complex), corr(s=[0.1, 0.2], mu=[0.1, 0.2], complex=complex)[::-1, ::-1], atol=0)


def test_global():
    z = 1.
    bias, f, boxsize, nmesh, boxcenter, nbar, los, ells, seed = 1., 0.8, 500., 256, 0., 1e-2, 'x', (0, 2, 4), 42
    power = DESI().get_fourier().pk_interpolator().to_1d(z=z)

    mock = EulerianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los)
    mock.set_analytic_selection_function(nbar=nbar)

    result = MeshFFTCorr(mock.mesh_delta_r + 1., los=los, ells=ells, edges={'step': 5.})
    plot_correlation_function(result.poles, theory=kaiser(power=power, bias=bias, f=f))

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=44)
    randoms = RandomBoxCatalog(nbar=4. * nbar, boxsize=boxsize, boxcenter=boxcenter, seed=45)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='cic', compensate=True) + 1.

    result = CatalogFFTCorr(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], ells=ells, los=los,
                            edges={'step': 5.}, boxsize=boxsize, nmesh=nmesh, resampler='tsc', interlacing=2, position_type='pos', mpicomm=data.mpicomm)
    plot_correlation_function(result.poles, theory=kaiser(power=power, bias=bias, f=f))
    result.wedges.plot(show=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = '_tests'
        fn = os.path.join(tmp_dir, 'tmp_poles.txt')
        result.poles.save_txt(fn, complex=False)
        fn = os.path.join(tmp_dir, 'tmp_wedges.txt')
        result.wedges.save_txt(fn, complex=False)


def test_local():
    z = 1.
    bias, f, boxsize, nmesh, boxcenter, nbar, los, ells, seed = 1., 0.8, 500., 256, (10000., 0., 0.), 1e-2, None, (0, 2, 4), 42
    power = DESI().get_fourier().pk_interpolator().to_1d(z=z)

    mock = EulerianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los)
    mock.set_analytic_selection_function(nbar=nbar)

    result = MeshFFTCorr(mock.mesh_delta_r + 1., los=los, ells=ells, edges={'step': 5.}, boxcenter=boxcenter)
    plot_correlation_function(result.poles, theory=kaiser(power=power, bias=bias, f=f))

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=44)
    randoms = RandomBoxCatalog(nbar=4. * nbar, boxsize=boxsize, boxcenter=boxcenter, seed=45)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='cic', compensate=True) + 1.

    result = CatalogFFTCorr(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], ells=ells, los=los,
                            edges={'step': 6.}, boxsize=boxsize, nmesh=nmesh, resampler='tsc', interlacing=2, position_type='pos', mpicomm=data.mpicomm)
    plot_correlation_function(result.poles, theory=kaiser(power=power, bias=bias, f=f))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = '_tests'
        fn = os.path.join(tmp_dir, 'tmp_poles.txt')
        result.poles.save_txt(fn, complex=False)


def test_cutsky():

    z = 1.
    bias, f, boxsize, nmesh, boxcenter, nbar, los, ells, seed = 1., 0.8, 500., 256, (0., 0., 0.), 1e-3, 'x', (0, 2, 4), 42
    power = DESI().get_fourier().pk_interpolator().to_1d(z=z)

    mock = EulerianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los if los in 'xyz' else None)
    mock.set_analytic_selection_function(nbar=nbar)

    # result = MeshFFTCorr(mock.mesh_delta_r + 1., los=los, ells=ells, edges={'step': 5.}, boxcenter=boxcenter)
    # plot_correlation_function(result.poles, theory=kaiser(power=power, bias=bias, f=f))

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=44)
    randoms = RandomBoxCatalog(nbar=4. * nbar, boxsize=boxsize, boxcenter=boxcenter, seed=45)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='cic', compensate=True) + 1.

    from pycorr import TwoPointCorrelationFunction
    edges = (np.linspace(0., 100., 26), np.linspace(-1., 1., 101))
    pc = TwoPointCorrelationFunction('smu', edges=edges, data_positions1=data['Position'], data_weights1=data['Weight'], los=los,
                                     position_type='pos', boxsize=boxsize, mpicomm=data.mpicomm)
    fft = CatalogFFTCorr(data_positions1=data['Position'], data_weights1=data['Weight'], ells=ells, los=los,
                         edges=edges[0], boxsize=boxsize, nmesh=nmesh, resampler='tsc', interlacing=2, position_type='pos',
                         mpicomm=data.mpicomm).poles

    ax = plt.gca()
    mask = fft.s > 20
    for ill, ell in enumerate(fft.ells):
        ax.plot(fft.s[mask], fft.s[mask]**2 * fft(ell=ell)[mask], label=r'$\ell = {:d}$'.format(ell), color='C{:d}'.format(ill), linestyle='-')
        s, xi = pc(ell=ell, return_sep=True)
        ax.plot(s[mask], s[mask]**2 * xi[mask], label=r'$\ell = {:d}$'.format(ell), color='C{:d}'.format(ill), linestyle='--')
    plt.show()


def test_cutsky():

    z = 1.
    bias, f, boxsize, nmesh, boxcenter, nbar, los, ells, seed = 1., 0.8, 500., 256, (10000., 0., 0.), 1e-3, 'x', (0,), 42
    power = DESI().get_fourier().pk_interpolator().to_1d(z=z)

    mock = EulerianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los if los in 'xyz' else None)
    mock.set_analytic_selection_function(nbar=nbar)

    # result = MeshFFTCorr(mock.mesh_delta_r + 1., los=los, ells=ells, edges={'step': 5.}, boxcenter=boxcenter)
    # plot_correlation_function(result.poles, theory=kaiser(power=power, bias=bias, f=f))

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=44)
    randoms = RandomBoxCatalog(nbar=4. * nbar, boxsize=boxsize, boxcenter=boxcenter, seed=45)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='cic', compensate=True) + 1.

    from pycorr import TwoPointCorrelationFunction
    edges = (np.linspace(0., 100., 26), np.linspace(-1., 1., 101))
    pc = TwoPointCorrelationFunction('smu', edges=edges, data_positions1=data['Position'], data_weights1=data['Weight'],
                                     randoms_positions1=randoms['Position'], los=los, position_type='pos', boxsize=boxsize,
                                     mpicomm=data.mpicomm)
    boxsize2 = boxsize
    fft = CatalogFFTCorr(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], ells=ells, los=los,
                         edges=edges[0], boxsize=boxsize2, nmesh=nmesh, resampler='tsc', interlacing=2, position_type='pos',
                         mpicomm=data.mpicomm).poles
    fftr = CatalogFFTCorr(data_positions1=randoms['Position'], ells=ells, los=los,
                          edges=edges[0], boxsize=boxsize2, nmesh=nmesh, resampler='tsc', interlacing=2, position_type='pos',
                          mpicomm=data.mpicomm).poles
    fftr.wnorm = fft.wnorm * (randoms.csize / data.csize) ** 2
    rr = fftr(ell=0, null_zero_mode=False)

    ax = plt.gca()
    mask = fft.s > 20
    print(rr[mask])
    for ill, ell in enumerate(fft.ells):
        ax.plot(fft.s[mask], fft.s[mask]**2 * fft(ell=ell)[mask], label=r'$\ell = {:d}$'.format(ell), color='C{:d}'.format(ill), linestyle=':')
        ax.plot(fft.s[mask], fft.s[mask]**2 * fft(ell=ell)[mask] / rr[mask], label=r'$\ell = {:d}$'.format(ell), color='C{:d}'.format(ill), linestyle='-')
        s, xi = pc(ell=ell, return_sep=True)
        ax.plot(s[mask], s[mask]**2 * xi[mask], label=r'$\ell = {:d}$'.format(ell), color='C{:d}'.format(ill), linestyle='--')
    plt.show()


if __name__ == '__main__':

    setup_logging()
    test_corr_statistic()
    test_global()
    test_local()
    # test_cutsky()
