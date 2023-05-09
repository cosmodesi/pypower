import os
import tempfile
import pytest

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo import Cosmology
from mockfactory import Catalog

from pypower import (CorrelationFunctionSmoothWindow, PowerSpectrumSmoothWindow, Projection,
                     BaseMatrix, CorrelationFunctionSmoothWindowMatrix, PowerSpectrumSmoothWindowMatrix,
                     CorrelationFunctionOddWideAngleMatrix, PowerSpectrumOddWideAngleMatrix, CatalogFFTPower, CatalogSmoothWindow,
                     mpi, setup_logging)
from pypower.smooth_window import wigner3j_square

from test_fft_power import data_fn, randoms_fn


def test_wigner():
    assert wigner3j_square(ellout=3, ellin=4, prefactor=True)[0] == [1, 3, 5, 7]


def test_power_spectrum_window_matrix():
    ellsin = (0, 1, 2, 3, 4)
    wa_orders = [0, 1]
    ellsout = ellsin
    projsin = [Projection(ell=ell, wa_order=wa_order) for ell in ellsin for wa_order in wa_orders]
    projsout = [Projection(ell=ell, wa_order=wa_order) for ell in ellsout for wa_order in wa_orders]

    kout = np.linspace(0.1, 0.4, 2)

    swin = np.linspace(1e-4, 1e3, 1000)
    win = np.exp(-(swin / 10.)**2)
    dwindow = {}
    for n in range(3):
        dwindow[n] = {}
        for ell in range(5):
            dwindow[n][ell] = win.copy()
            if ell > 0: dwindow[n][ell] *= np.random.uniform()
            # if (ell % 2 == 1) and (n == 0): dwindow[n][ell][...] = 0.
            # if (ell % 2 == 0) and (n == 1): dwindow[n][ell][...] = 0.
            # if (ell % 2 == 1): dwindow[n][ell][...] = 0.
            if n > 1: dwindow[n][ell][...] = 0.

    from scipy.interpolate import interp1d

    def window(proj, sep, **kwargs):
        dwin = dwindow[proj.wa_order]
        if proj.ell <= 4: dwin = dwin[proj.ell]
        else: dwin = 0. * swin
        return interp1d(swin, dwin, kind='linear', fill_value=((1. if ell == 0 else 0.), 0.), bounds_error=False)(sep)

    sep = np.geomspace(swin[0], swin[-1], 1024 * 16 * 2)
    kin = 1. / sep[::-1] / (sep[1] / sep[0])

    wm = PowerSpectrumSmoothWindowMatrix(kout, projsin, projsout=projsout, weightsout=2 * np.ones_like(kout), window=window, k=kin, kin_rebin=2, kin_lim=None, sep=sep, sum_wa=False)
    kin = wm.xin[0]
    mask = (kin > 0.001) & (kin < 1.)
    test = wm.value.T

    from create_Wll import create_W
    ref = create_W(kout, swin, dwindow)

    from matplotlib import pyplot as plt
    from pypower.smooth_window import weights_trapz

    for wa_order in wa_orders:
        fig, lax = plt.subplots(len(ellsout), len(ellsin), figsize=(12, 10))
        for illout, ellout in enumerate(ellsout):
            for illin, ellin in enumerate(ellsin):
                iprojout = projsout.index(Projection(ell=ellout, wa_order=wa_order))
                iprojin = projsout.index(Projection(ell=ellin, wa_order=wa_order))
                testi = test[iprojout * len(kout), iprojin * len(kin):(iprojin + 1) * len(kin)] / (weights_trapz(kin**3) / 3.)

                if wa_order % 2 == 0 and ellin % 2 == 1:
                    testi = 0. * testi  # convention of create_Wll (no theory odd multipoles at 0th order)
                if wa_order % 2 == 1 and ellin % 2 == 0:
                    testi = 0. * testi  # convention of create_Wll (no theory even multipoles at 1th order)

                refi = ref[(wa_order, ellout, ellin)][0]
                lax[illout][illin].plot(kin[mask], testi[mask], label='test ({:d},{:d})'.format(ellout, ellin))
                lax[illout][illin].plot(kin[mask], refi[mask], label='ref')
                lax[illout][illin].legend()
            # print(np.max(test_-ref_))
            # assert np.allclose(test_,ref_,rtol=1e-1,atol=1e-3)

    plt.show()


def test_window():

    edges = np.linspace(0., 10, 1001)
    k = (edges[:-1] + edges[1:]) / 2.
    win = np.exp(-(k * 10)**2)
    y, projs = [], []
    for wa_order in range(2):
        for ell in range(9):
            y_ = win.copy()
            if ell > 0: y_ *= np.random.uniform() / 10.
            y.append(y_)
            projs.append(Projection(ell=ell, wa_order=wa_order))
    nmodes = np.ones_like(k, dtype='i4')
    boxsize = np.array([1000.] * 3, dtype='f8')
    window = PowerSpectrumSmoothWindow(edges, k, y, nmodes, projs, attrs={'boxsize': boxsize})
    window2 = PowerSpectrumSmoothWindow(edges, k, [yy / 2. for yy in y], 2. * nmodes, projs, attrs={'boxsize': boxsize})
    window = PowerSpectrumSmoothWindow.concatenate_x(window, window2)
    assert np.allclose(window.power_nonorm[0], y[0] / 2.)

    window2 = PowerSpectrumSmoothWindow.concatenate_proj(window, window)
    window2.select(projs=projs[:2])
    assert window2.projs == projs[:2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'tmp.npy')
        window.save(fn)
        test = PowerSpectrumSmoothWindow.load(fn)
        fn = os.path.join(tmp_dir, 'tmp.npy')
        test.save(fn)

    assert np.allclose(test(projs[0], k), window.power_nonorm[0])
    assert np.allclose(test(projs[0], 10.), 0.)
    assert test(projs[0], 10.).shape == ()
    assert test(projs[0], [10.] * 3).shape == (3, )
    assert test(k=[9.] * 3).shape == (len(projs), 3)
    assert np.allclose(test(k=[1., 2.]), test(k=[2., 1.])[..., ::-1], atol=0)

    window2 = PowerSpectrumSmoothWindow(edges, k, y, nmodes, projs, power_zero_nonorm=[10.] + [0.] * (len(projs) - 1), attrs={'boxsize': boxsize})
    assert np.allclose(window2(proj=0, k=0., null_zero_mode=False), win[0])
    assert np.allclose(window2(proj=0, k=0., null_zero_mode=True), win[0] - 10.)

    y = np.array(y)
    y2 = y.copy()
    y2[:, 1::2] = np.nan
    window2 = PowerSpectrumSmoothWindow(edges, k[::2], y[:, ::2], nmodes[::2], projs, attrs={'boxsize': boxsize})
    window2_nan = PowerSpectrumSmoothWindow(edges, k, y2, nmodes, projs, attrs={'boxsize': boxsize})
    assert np.allclose(window2_nan(projs[0], k), window2(projs[0], k))

    with pytest.raises(IndexError):
        window((18, 2))
    assert np.allclose(window((18, 2), default_zero=True), 0.)

    window_real = window.to_real()
    assert np.allclose(window.to_real(sep=1. / window.k[window.k > 0][::-1]).corr, window_real.corr)
    assert np.allclose(window.to_real(k=window.k, smooth=0.).corr, window.to_real(k=window.k).corr)
    assert np.allclose(window.to_real(smooth=0.5).corr, window.to_real(smooth=np.exp(-(0.5 * window.k)**2)).corr)
    assert not np.isnan(window2_nan.to_real().corr).any()

    with pytest.raises(IndexError):
        window_real((18, 2))
    assert np.allclose(window_real((18, 2), default_zero=True), 0.)
    assert window_real(projs[0], 10.).shape == ()
    assert window_real(projs[0], [10.] * 3).shape == (3, )
    assert window_real(sep=[9.] * 3).shape == (len(projs), 3)
    assert np.allclose(window_real(sep=[1., 2.]), window_real(sep=[2., 1.])[..., ::-1], atol=0)

    window.power_zero_nonorm[0] = 10.
    window_real2 = window.to_real()
    assert np.allclose(window_real2.corr, window_real.corr)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'tmp.npy')
        window_real.save(fn)
        test = CorrelationFunctionSmoothWindow.load(fn)
        test.save(fn)

    assert np.allclose(test(projs[0], 1. / k[::-1]), window_real.corr[0])
    test.select(rp=2.5)


def test_fft_window():

    boxsize = 2000.
    nmesh = 64
    kedges = np.linspace(0., 0.1, 6)
    ells = (0, 2)
    resampler = 'tsc'
    interlacing = 2
    dtype = 'f8'
    cdtype = 'c16'
    boxcenter = np.array([3000., 0., 0.])[None, :]

    for los in ['x', 'firstpoint', 'endpoint']:

        data = Catalog.read(data_fn)
        randoms = Catalog.read(randoms_fn)

        for catalog in [data, randoms]:
            catalog['Position'] += boxcenter
            catalog['Weight'] = catalog.ones()

        poles = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'],
                                boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=kedges, position_type='pos', dtype=dtype).poles

        edges = {'step': 0.01}

        window1 = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], power_ref=poles, edges=edges, position_type='pos').poles
        with tempfile.TemporaryDirectory() as tmp_dir:
            # tmp_dir = '_tests'
            fn = data.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            fn_txt = data.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.txt'), root=0)
            window1.save(fn)
            window1.save_txt(fn_txt, complex=False)
            window1.mpicomm.Barrier()
            window = PowerSpectrumSmoothWindow.load(fn)
            fn = os.path.join(tmp_dir, 'tmp.npy')
            window.save(fn)
            assert np.allclose(window.power[0], window1.power[0], equal_nan=True)

        poles_f4 = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'],
                                   boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=kedges, position_type='pos', dtype='f4').poles
        window_f4 = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], power_ref=poles_f4, edges=edges, position_type='pos')
        assert window_f4.dtype.itemsize == 4

        windowc = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], power_ref=poles, edges=edges, position_type='pos', dtype=cdtype).poles

        for iproj, proj in enumerate(windowc.projs):
            atol = 20. if proj.ell % 2 == 0 else 1e-5
            assert np.allclose(windowc.power[iproj].imag, window.power[iproj].imag, atol=atol, rtol=5e-2)
            atol = 2e-1 if proj.ell % 2 else 1e-5
            assert np.allclose(windowc.power[iproj].real, window.power[iproj].real, atol=atol, rtol=5e-2)

        randoms['Position'][0] += boxsize
        windowp1 = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], power_ref=poles, wrap=True, edges=edges, projs=window.projs[:2], position_type='pos', dtype=dtype).poles
        windowp2 = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], power_ref=poles, wrap=True, edges=edges, projs=window.projs[2:], position_type='pos', dtype=dtype).poles
        windowc = PowerSpectrumSmoothWindow.concatenate_proj(windowp1, windowp2)
        assert np.allclose(windowc.power, window1.power)

        assert not np.allclose(windowp1.wnorm_ref, windowp1.wnorm)
        bak = windowc.power_nonorm.copy()
        windowc += windowc
        assert np.allclose(windowc.power, window1.power)
        assert np.allclose(windowc.power_nonorm, 2 * bak)

        window2 = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], randoms_weights2=0.5 * randoms['Weight'], power_ref=poles, wrap=True, edges=edges, projs=window.projs[:2], position_type='pos', dtype=dtype).poles
        assert np.allclose(window2.power_nonorm, 0.5 * windowp1.power_nonorm)
        assert np.allclose(window2.power, windowp1.power)

        window1 = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], power_ref=poles, wrap=True, edges={'step': 0.0005}, position_type='pos').poles
        window2 = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], power_ref=poles, wrap=True, edges={'step': 0.0005}, position_type='pos', boxsize=1000., dtype=dtype).poles
        windowc = PowerSpectrumSmoothWindow.concatenate_x(window1, window2)
        assert windowc.k[-1] > window1.k[-1]
        for name in ['power', 'k', 'nmodes']:
            assert np.allclose(getattr(windowc, name)[..., :len(window1.k)], getattr(window1, name), equal_nan=True)
            assert np.allclose(getattr(windowc, name)[..., len(window1.k):], getattr(window2, name)[..., len(windowc.k) - len(window1.k):], equal_nan=True)
        windowc2 = PowerSpectrumSmoothWindow.concatenate_x(window2, window1)
        mask = np.flatnonzero(window2.nmodes[:len(window1.nmodes)] != window1.nmodes)
        assert np.allclose(windowc2.power[:, mask], windowc.power[:, mask], equal_nan=True)
        assert np.allclose(windowc2.modes[0], windowc.modes[0], equal_nan=True)

        frac_nyq = 0.8
        windowc = PowerSpectrumSmoothWindow.concatenate_x(window1, window2, frac_nyq=frac_nyq)
        assert windowc.k[-1] > window1.k[-1]
        knyq1 = np.pi * np.min(window1.attrs['nmesh'] / window1.attrs['boxsize'])
        knyq2 = np.pi * np.min(window2.attrs['nmesh'] / window2.attrs['boxsize'])
        for name in ['power', 'k', 'nmodes']:
            assert np.allclose(getattr(windowc, name)[..., windowc.kedges[1:] <= frac_nyq * knyq1], getattr(window1, name)[..., window1.kedges[1:] <= frac_nyq * knyq1], equal_nan=True)
            assert np.allclose(getattr(windowc, name)[..., windowc.kedges[1:] > frac_nyq * knyq1], getattr(window2, name)[..., (window2.kedges[1:] > frac_nyq * knyq1) & (window2.kedges[1:] <= frac_nyq * knyq2)], equal_nan=True)

        assert windowc[1:5:2].shape[0] == 2

        windowc = PowerSpectrumSmoothWindow.concatenate_x(window1, window2, frac_nyq=(frac_nyq, ))
        assert windowc.k[-1] > window1.k[-1]
        knyq1 = np.pi * np.min(window1.attrs['nmesh'] / window1.attrs['boxsize'])
        for name in ['power', 'k', 'nmodes']:
            assert np.allclose(getattr(windowc, name)[..., windowc.kedges[1:] <= frac_nyq * knyq1], getattr(window1, name)[..., window1.kedges[1:] <= frac_nyq * knyq1], equal_nan=True)
            assert np.allclose(getattr(windowc, name)[..., windowc.kedges[1:] > frac_nyq * knyq1], getattr(window2, name)[..., (window2.kedges[1:] > frac_nyq * knyq1)], equal_nan=True)

        if window1.mpicomm.rank == 0:
            # Let us compute the wide-angle and window function matrix
            kout = poles.k  # output k-bins
            ellsout = poles.ells  # output multipoles
            ellsin = poles.ells  # input (theory) multipoles
            wa_orders = 1  # wide-angle order
            sep = np.geomspace(1e-4, 4e3, 1024)  # configuration space separation for FFTlog
            kin_rebin = 4  # rebin input theory to save memory
            kin_lim = (0, 2e1)  # pre-cut input (theory) ks to save some memory
            # Input projections for window function matrix:
            # theory multipoles at wa_order = 0, and wide-angle terms at wa_order = 1
            projsin = list(ellsin) + PowerSpectrumOddWideAngleMatrix.propose_out(ellsin, wa_orders=wa_orders)
            # Window matrix
            wm = PowerSpectrumSmoothWindowMatrix(kout, projsin=projsin, projsout=ellsout, window=window, sep=sep, kin_rebin=kin_rebin, kin_lim=kin_lim, default_zero=True)
            assert np.allclose(wm.weight, poles.wnorm)
            bak = wm.value.copy()
            wm2 = wm + wm
            assert np.allclose(wm2.value, wm.value)
            assert np.allclose(wm2.value, bak)

        randoms['Position'][0] -= boxsize
        projsin = [(ell, 0) for ell in range(0, 2 * max(ells) + 1, 2)]
        if los in ['firstpoint', 'endpoint']:
            projsin += [(ell, 1) for ell in range(1, 2 * max(ells) + 2, 2)]
        alpha = data['Weight'].csum() / randoms['Weight'].csum()
        window_noref = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edges=edges, projs=projsin, los=los,
                                           boxsize=boxsize, boxcenter=poles.attrs['boxcenter'], nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos',
                                           wnorm=poles.wnorm / alpha**2, dtype=dtype).poles

        assert np.allclose(window_noref.power, window.power)

        positions = mpi.gather(randoms['Position'], mpiroot=0, mpicomm=catalog.mpicomm)
        weights = mpi.gather(randoms['Weight'], mpiroot=0, mpicomm=catalog.mpicomm)
        window_root = CatalogSmoothWindow(randoms_positions1=positions, randoms_weights1=weights, power_ref=poles, edges=edges, position_type='pos', mpiroot=0).poles
        assert np.allclose(window_root.power, window.power)

        if randoms.mpicomm.rank == 0:
            window_root = CatalogSmoothWindow(randoms_positions1=positions, randoms_weights1=weights, power_ref=poles, edges=edges, position_type='pos', mpicomm=mpi.COMM_SELF).poles
            assert np.allclose(window_root.power, window.power)

        nk = len(windowc.k)
        windowc.rebin(2)
        assert len(windowc.k) == nk // 2

        if los in ['firstpoint', 'endpoint']:
            window = CatalogSmoothWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], power_ref=poles, edges=edges, position_type='pos', direct_selection_attrs={'rp': (0., 2.5)}, direct_edges={'step': 0.2, 'max': 10.}).poles
            #assert np.allclose(window.get_power(add_direct=False), window_root.power)
            window.to_real()


def get_correlation_function_window():
    sep = np.linspace(1e-4, 1e3, 1000)
    win = np.exp(-(sep / 100.)**2)

    y, projs = [], []
    for wa_order in range(2):
        for ell in range(10):
            y_ = win.copy()
            if ell > 0: y_ *= np.random.uniform() / 10.
            y.append(y_)
            projs.append(Projection(ell=ell, wa_order=wa_order))
    return CorrelationFunctionSmoothWindow(sep, y, projs)


def test_correlation_function_window_matrix():

    window = get_correlation_function_window()
    ells = [0, 2, 4]
    projsin = ells + PowerSpectrumOddWideAngleMatrix.propose_out(ells, wa_orders=1)
    wm = CorrelationFunctionSmoothWindowMatrix(np.linspace(0., 1., 10), projsin, projsout=ells, window=window)
    kwargs = {'wa_orders': 1, 'los': 'firstpoint'}
    wa = CorrelationFunctionOddWideAngleMatrix(wm.xin[0], ells, projsout=wm.projsin, **kwargs)
    matrix = BaseMatrix.join(wa, wm)
    wm.resum_input_odd_wide_angle(**kwargs)
    assert np.allclose(matrix.value, wm.value)
    assert len(matrix.xin) == len(matrix.projsin) == len(ells)
    assert len(matrix.xout) == len(matrix.projsout) == len(ells)


def test_window_convolution():
    window = get_correlation_function_window()
    ells = [0, 2, 4]
    sep = np.geomspace(window.sep[0], window.sep[-1], 1024)
    kin_lim = (1e-3, 1e1)
    kout = np.linspace(0., 0.3, 60)
    projsin = ells + PowerSpectrumOddWideAngleMatrix.propose_out(ells, wa_orders=1)
    wm = PowerSpectrumSmoothWindowMatrix(kout, projsin, projsout=ells, window=window, sep=sep, kin_lim=kin_lim)
    kwargs = {'d': 1000., 'wa_orders': 1, 'los': 'firstpoint'}
    wa = PowerSpectrumOddWideAngleMatrix(wm.xin[0], ells, projsout=wm.projsin, **kwargs)
    matrix = BaseMatrix.join(wa, wm)
    wm.resum_input_odd_wide_angle(**kwargs)
    assert np.allclose(matrix.value, wm.value)
    assert len(matrix.xin) == len(matrix.projsin) == len(ells)
    assert len(matrix.xout) == len(matrix.projsout) == len(ells)
    matrix.rebin_x(factorout=2)
    matrix.rebin_x(factorin=5)

    kin = matrix.xin[0]
    kout = matrix.xout[0]
    pklin = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()(kin)

    def kaiser(f=0.8, bias=1.4):
        beta = f / bias
        toret = []
        toret.append((1. + 2. / 3. * beta + 1. / 5. * beta**2) * pklin)
        toret.append((4. / 3. * beta + 4. / 7. * beta**2) * pklin)
        toret.append(8. / 35 * beta**2 * pklin)
        return toret

    pk = kaiser()
    pkconv = matrix.dot(pk, unpack=True)
    ax = plt.gca()
    for ill in range(len(ells)):
        ax.plot(kin, kin * pk[ill], color='C{:d}'.format(ill), linestyle='--')
        ax.plot(kout, kout * pkconv[ill], color='C{:d}'.format(ill), linestyle='-')
    ax.set_xlim(0., 0.3)
    plt.show()


if __name__ == '__main__':

    setup_logging()

    test_wigner()
    test_window()
    test_fft_window()
    test_power_spectrum_window_matrix()
    test_correlation_function_window_matrix()
    test_window_convolution()
