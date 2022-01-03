import os

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo import Cosmology
from mockfactory import Catalog

from pypower import CorrelationFunctionWindow, PowerSpectrumWindow, Projection,\
                    BaseMatrix, PowerSpectrumWindowMatrix, PowerSpectrumOddWideAngleMatrix,\
                    CatalogFFTPower, CatalogFFTWindow, setup_logging

from test_fft_power import data_fn, randoms_fn


base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, '_plots')
window_fn = os.path.join(plot_dir, 'window_function.npy')


def test_deriv():
    n = 5
    m = np.zeros((n,n), dtype='f8')
    m += np.diag(np.ones(n-1), k=1) - np.diag(np.ones(n-1), k=-1)
    m[0,0] = -0.5
    m[0,1] = 0.5
    m[-1,-1] = 0.5
    m[-1,-2] = -0.5

    ref = np.zeros((n,n), dtype='f8')
    for index1 in range(n):
        index2 = index1

        if index2 > 0 and index2 < n-1:
            pre_factor = 1.
        else:
            pre_factor = 0.5

        if index2 > 0:
            ref[index1, index2-1] = -pre_factor
        else:
            ref[index1, index2] += -pre_factor

        if index2 < n-1:
            ref[index1, index2+1] = pre_factor
        else:
            ref[index1, index2] += pre_factor

    assert np.allclose(m, ref)


def test_projection():

    tu = (2, 1)
    proj = Projection(tu)
    assert proj.ell == tu[0] and proj.wa_order == tu[1]
    proj = Projection(*tu)
    assert proj.ell == tu[0] and proj.wa_order == tu[1]
    proj = Projection(ell=tu[0], wa_order=tu[1])
    assert proj.ell == tu[0] and proj.wa_order == tu[1]
    assert proj.latex() == '(\\ell, n) = (2, 1)'
    assert proj.latex(inline=True) == '$(\\ell, n) = (2, 1)$'


def test_wigner():
    from pypower.approx_window import wigner3j_square
    assert wigner3j_square(ellout=3, ellin=4, prefactor=True)[0] == [1, 3, 5, 7]


def test_power_spectrum_odd_wideangle():
    ells = [0, 2, 4]
    kmin, kmax = 0., 0.2
    nk = 10
    dk = (kmax - kmin)/nk
    k = np.array([i*dk + dk/2. for i in range(nk)])
    d = 1.
    projsin = [Projection(ell=ell, wa_order=0) for ell in ells]
    projsout = [Projection(ell=ell, wa_order=ell % 2) for ell in range(ells[-1]+1)]
    wa = PowerSpectrumOddWideAngleMatrix(k, projsin, projsout=projsout, d=1., wa_orders=1, los='firstpoint')

    from wide_angle_tools import get_end_point_LOS_M
    ref = get_end_point_LOS_M(d, Nkth=nk, kmin=kmin, kmax=kmax)

    assert np.allclose(wa.matrix, ref)

    assert wa.projsout != wa.projsin
    wa.select_projs(projsout=projsin)
    assert wa.projsout == wa.projsin

    shape = wa.matrix.shape
    wa.rebin_x(factorout=2)
    assert wa.matrix.shape == (shape[0]//2, shape[1])
    assert np.allclose(wa.xout[0], k[::2])

    klim = (0., 0.15)
    mask = (k >= klim[0]) & (k <= klim[1])
    assert not np.all(mask)
    wa.select_x(xinlim=klim)
    assert np.allclose(wa.xin[0], k[mask])


def test_window_matrix():
    ellsin = (0, 1, 2, 3, 4)
    wa_orders = [0, 1]
    ellsout = ellsin
    projsin = [Projection(ell=ell, wa_order=wa_order) for ell in ellsin for wa_order in wa_orders]
    projsout = [Projection(ell=ell, wa_order=wa_order) for ell in ellsout for wa_order in wa_orders]

    kout = np.linspace(0.1, 0.4, 2)

    swin = np.linspace(1e-4, 1e3, 1000)
    win = np.exp(-(swin/10.)**2)
    dwindow = {}
    for n in range(3):
        dwindow[n] = {}
        for ell in range(5):
            dwindow[n][ell] = win.copy()
            if ell > 0: dwindow[n][ell] *= np.random.uniform()
            #if (ell % 2 == 1) and (n == 0): dwindow[n][ell][...] = 0.
            #if (ell % 2 == 0) and (n == 1): dwindow[n][ell][...] = 0.
            #if (ell % 2 == 1): dwindow[n][ell][...] = 0.
            if n > 1: dwindow[n][ell][...] = 0.

    from scipy.interpolate import interp1d

    def window(proj, sep, **kwargs):
        dwin = dwindow[proj.wa_order]
        if proj.ell <= 4: dwin = dwin[proj.ell]
        else: dwin = 0.*swin
        return interp1d(swin, dwin, kind='linear', fill_value=((1. if ell == 0 else 0.), 0.), bounds_error=False)(sep)

    sep = np.geomspace(swin[0], swin[-1], 1024*16)
    kin = 1./sep[::-1]/(sep[1]/sep[0])

    wm = PowerSpectrumWindowMatrix(kout, projsin, projsout=projsout, window=window, k=kin, kinlim=None, sep=sep, sum_wa=False)
    kin = wm.xin[0]
    mask = (kin > 0.001) & (kin < 1.)
    test = wm.matrix

    from create_Wll import create_W
    ref = create_W(kout, swin, dwindow)

    from matplotlib import pyplot as plt
    from pypower.approx_window import weights_trapz

    for wa_order in wa_orders:
        fig,lax = plt.subplots(len(ellsout), len(ellsin), figsize=(10,8))
        for illout,ellout in enumerate(ellsout):
            for illin,ellin in enumerate(ellsin):
                iprojout = projsout.index(Projection(ell=ellout, wa_order=wa_order))
                iprojin = projsout.index(Projection(ell=ellin, wa_order=wa_order))
                test_ = test[iprojout*len(kout), iprojin*len(kin):(iprojin+1)*len(kin)] / (weights_trapz(kin**3) / 3.)

                if wa_order % 2 == 0 and ellin % 2 == 1:
                    test_ = 0.*test_ # convention of create_Wll (no theory odd multipoles at 0th order)
                if wa_order % 2 == 1 and ellin % 2 == 0:
                    test_ = 0.*test_ # convention of create_Wll (no theory even multipoles at 1th order)
                if ellin % 2 == 1:
                    test_ *= -1 # convention for input odd power spectra (we provide the imaginary part, wide_angle_tools.py provides - the inmaginary part)
                if ellout % 2 == 1:
                    test_ *= -1 # same as above

                ref_ = ref[(wa_order,ellout,ellin)][0]
                lax[illout][illin].plot(kin[mask], test_[mask], label='test ({:d},{:d})'.format(ellout,ellin))
                lax[illout][illin].plot(kin[mask], ref_[mask], label='ref')
                lax[illout][illin].legend()
            #print(np.max(test_-ref_))
            #assert np.allclose(test_,ref_,rtol=1e-1,atol=1e-3)

    plt.show()


def test_fft_window():
    boxsize = 2000.
    nmesh = 64
    kedges = np.linspace(0., 0.1, 6)
    ells = (0, 2, 4)
    resampler = 'tsc'
    interlacing = 2
    dtype = 'f4'
    los = None
    boxcenter = np.array([3000.,0.,0.])[None,:]
    data = Catalog.load_fits(data_fn)
    randoms = Catalog.load_fits(randoms_fn)

    for catalog in [data, randoms]:
        catalog['Position'] += boxcenter
        catalog['Weight'] = catalog.ones()

    poles = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'],
                            boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=kedges, position_type='pos', dtype=dtype).poles
    window = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], power_ref=poles, position_type='pos', dtype=dtype).poles
    window.save(window_fn)
    test = PowerSpectrumWindow.load(window_fn)
    assert np.allclose(test(test.projs[0], test.k), window.power[0])


def test_window():

    edges = np.linspace(1e-4, 10, 1001)
    k = (edges[:-1] + edges[1:])/2.
    win = np.exp(-(k*10)**2)
    y, projs = [], []
    for wa_order in range(2):
        for ell in range(9):
            y_ = win.copy()
            if ell > 0: y_ *= np.random.uniform()/10.
            y.append(y_)
            projs.append(Projection(ell=ell, wa_order=wa_order))
    nmodes = np.ones_like(k, dtype='i4')
    boxsize = np.array([1000.]*3, dtype='f8')
    window = PowerSpectrumWindow(edges, k, y, nmodes, projs, attrs={'boxsize':boxsize})
    window.save(window_fn)
    test = PowerSpectrumWindow.load(window_fn)
    assert np.allclose(test(projs[0], k), window.power_nonorm[0])
    window_real = window.to_real()
    window_real.save(window_fn)
    test = CorrelationFunctionWindow.load(window_fn)
    assert np.allclose(test(projs[0], 1./k[::-1]), window_real.corr[0])


def get_window_matrix(projsin, projsout=(0, 2, 4)):
    sep = np.linspace(1e-4, 1e3, 1000)
    win = np.exp(-(sep/100.)**2)

    y, projs = [], []
    for wa_order in range(2):
        for ell in range(9):
            y_ = win.copy()
            if ell > 0: y_ *= np.random.uniform()/10.
            y.append(y_)
            projs.append(Projection(ell=ell, wa_order=wa_order))
    window = CorrelationFunctionWindow(sep, y, projs)

    sep = np.geomspace(sep[0], sep[-1], 1024)
    kinlim = (1e-3, 1e1)
    kout = np.linspace(0., 0.3, 60)
    return PowerSpectrumWindowMatrix(kout, projsin, projsout=projsout, window=window, sep=sep, kinlim=kinlim)


def test_window_convolution():

    ells = [0, 2, 4]
    wm = get_window_matrix(ells + PowerSpectrumOddWideAngleMatrix.propose_out(ells, wa_orders=1))
    wa = PowerSpectrumOddWideAngleMatrix(wm.xin[0], ells, projsout=wm.projsin, d=1000., wa_orders=1, los='firstpoint')
    matrix = BaseMatrix.join(wa, wm)
    assert len(matrix.xin) == len(matrix.projsin) == len(ells)
    assert len(matrix.xout) == len(matrix.projsout) == len(ells)
    matrix.rebin_x(factorout=2)
    matrix.rebin_x(factorin=5)

    kin = matrix.xin[0]
    kout = matrix.xout[0]
    pklin = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()(kin)

    def kaiser(f=0.8, bias=1.4):
        beta = f/bias
        toret = []
        toret.append((1. + 2./3.*beta + 1./5.*beta**2)*pklin)
        toret.append((4./3.*beta + 4./7.*beta**2)*pklin)
        toret.append(8./35*beta**2*pklin)
        return toret

    pk = kaiser()
    pkconv = matrix.dot(pk, unpack=True)
    ax = plt.gca()
    for ill in range(len(ells)):
        ax.plot(kin, kin*pk[ill], color='C{:d}'.format(ill), linestyle='--')
        ax.plot(kout, kout*pkconv[ill], color='C{:d}'.format(ill), linestyle='-')
    ax.set_xlim(0., 0.3)
    plt.show()


if __name__ == '__main__':

    setup_logging()

    test_wigner()
    test_deriv()
    test_projection()
    test_power_spectrum_odd_wideangle()
    test_window_matrix()
    test_fft_window()
    test_window()
    test_window_convolution()
