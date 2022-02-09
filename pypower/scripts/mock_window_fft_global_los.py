"""
This script is dedicated to testing the window matrix for cutsky mocks, in the flat sky approximation.
First generate Gaussian mocks::

    (mpiexec -np 4) python mock_window_fft_global_los.py --todo mock --irun 0 20 # start - end of mock ids

Then compute window matrix::

    # you can split in as many parts as you want
    (mpiexec -np 4) python mock_window_fft_global_los.py --todo window --irun 0 3 # icut - ncuts
    (mpiexec -np 4) python mock_window_fft_global_los.py --todo window --irun 1 3
    (mpiexec -np 4) python mock_window_fft_global_los.py --todo window --irun 2 3

Then plot::

    python mock_window_fft_global_los.py --todo plot

Results are saved in "_results" (see below to change).
"""

import os
import logging
import glob
import argparse

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo.fiducial import DESI
from mockfactory import EulerianLinearMock
from mockfactory.make_survey import RandomBoxCatalog

from pypower import CatalogFFTPower, CatalogFFTWindow, PowerSpectrumFFTWindowMatrix, utils, setup_logging


logger = logging.getLogger('FFTWindowGlobalLOS')


cosmo = DESI()
z = 1.
pklin = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
f = cosmo.get_fourier().sigma8_z(z=z, of='theta_cb')/cosmo.get_fourier().sigma8_z(z=z, of='delta_cb')
bias = 1.5
boxsize = 1000.
boxcenter = np.array([600., 0., 0.])
nbar = 5e-4
#edgesin = np.linspace(1e-5, 0.4, 201)
edgesin = np.linspace(0.2, 0.25, 10)

# Change paths here if you wish
base_dir = '_results'
plot_dir = '_plots'
mock_fn = os.path.join(base_dir, 'mock_fft_global_los_{}.npy')
window_fn = os.path.join(base_dir, 'window_fft_global_los_{}.npy')
plot_window_matrix_fn = os.path.join(plot_dir, 'window_fft_global_los_matrix_poles.png')
plot_window_integ_fn = os.path.join(plot_dir, 'window_fft_global_los_integ_poles.png')
plot_poles_fn = os.path.join(plot_dir, 'power_window_fft_global_los_poles.png')
plot_wedges_fn = os.path.join(plot_dir, 'power_window_fft_global_los_wedges.png')


def run_mock(imock=0):
    seed = (imock + 1) * 42
    nmesh = 512; los = 'z'

    mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los)

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    randoms = RandomBoxCatalog(nbar=10*nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='cic', compensate=True) + 1.

    ells = (0, 2);  edges = (np.linspace(0., 0.4, 81), np.linspace(-1., 1., 7))
    power = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], ells=ells, los=los, edges=edges,
                            boxsize=2000., boxcenter=boxcenter, nmesh=256, resampler='tsc', interlacing=3, position_type='pos')
    power.save(mock_fn.format(imock))


def run_window(icut=0, ncuts=1):
    power = CatalogFFTPower.load(mock_fn.format(0))
    randoms = RandomBoxCatalog(nbar=10*nbar, boxsize=boxsize, boxcenter=boxcenter, seed=42)
    start, stop = icut*(len(edgesin) - 1)//ncuts, (icut + 1)*(len(edgesin) - 1)//ncuts + 1
    window = CatalogFFTWindow(randoms_positions1=randoms['Position'], edgesin=edgesin[start:stop], projsin=[0], power_ref=power, position_type='pos')
    window.save(window_fn.format(icut))


def kaiser_model_poles(k, ell=0):
    pk = bias**2*pklin(k)
    beta = f/bias
    if ell == 0: return (1. + 2./3.*beta + 1./5.*beta**2)*pk + 1./nbar
    if ell == 2: return (4./3.*beta + 4./7.*beta**2)*pk
    if ell == 4: return 8./35*beta**2*pk


def kaiser_model_wedges(k, wedge):
    from scipy import special
    ells = (0, 2, 4)
    pk = 0.
    for ell in ells:
        poly = special.legendre(ell).integ()(wedge)
        pk += kaiser_model_poles(k, ell) * (poly[1] - poly[0]) / (wedge[1] - wedge[0])
    return pk


def mock_mean(name='poles'):
    powers = []
    for fn in glob.glob(mock_fn.format('*')):
        powers.append(getattr(CatalogFFTPower.load(fn), name)(complex=False))
    return np.mean(powers, axis=0), np.std(powers, axis=0, ddof=1)/len(powers)**0.5


def plot_window():
    utils.mkdir(plot_dir)
    window = CatalogFFTWindow.load(window_fn.format('all')).poles
    unpacked = window.unpacked()
    fig, lax = plt.subplots(len(window.projsout), len(window.projsin), figsize=(15, 10), squeeze=False)
    for iprojin, projin in enumerate(window.projsin):
        for iprojout, projout in enumerate(window.projsout):
            # Indices in approximative window matrix
            print(projout, unpacked[iprojin][iprojout][0,:])
            lax[iprojout][iprojin].plot(window.xout[iprojout], unpacked[iprojin][iprojout][0,:])
            lax[iprojout][iprojin].set_title('${}$ x ${}$'.format(projin.latex(), projout.latex()))
    logger.info('Saving figure to {}.'.format(plot_window_matrix_fn))
    fig.savefig(plot_window_matrix_fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(fig)


def plot_window_integ():
    utils.mkdir(plot_dir)
    window = CatalogFFTWindow.load(window_fn.format('all')).poles
    integ = np.array_split(window.value.sum(axis=0), np.cumsum(window.nx[-1]))
    ax = plt.gca()
    for iproj, proj in enumerate(window.projsout):
        ax.plot(window.xout[iproj], integ[iproj], label=proj.latex(inline=True))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('$k$ [$h/\mathrm{Mpc}$]')
    logger.info('Saving figure to {}.'.format(plot_window_integ_fn))
    fig = plt.gcf()
    fig.savefig(plot_window_integ_fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(fig)


def plot_poles():
    utils.mkdir(plot_dir)
    window = CatalogFFTWindow.load(window_fn.format('all')).poles
    kin = window.xin[0]
    kout = window.xout[0]
    ellsin = [proj.ell for proj in window.projsin]
    ells = [proj.ell for proj in window.projsout]
    model_theory = np.array([kaiser_model_poles(kin, ell=ell) for ell in ellsin])
    model_conv = window.dot(model_theory, unpack=True)
    model_theory[ellsin.index(0)] -= 1./nbar
    model_conv[ells.index(0)] -= 1./nbar
    mean, std = mock_mean('poles')
    height_ratios = [max(len(ells), 3)] + [1]*len(ells)
    figsize = (6, 1.5*sum(height_ratios))
    fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios':height_ratios}, figsize=figsize, squeeze=True)
    fig.subplots_adjust(hspace=0)
    for ill, ell in enumerate(ellsin):
        lax[0].plot(kin, kin*model_theory[ill], linestyle=':', color='C{:d}'.format(ill), label='theory' if ill == 0 else None)
    for ill, ell in enumerate(ells):
        lax[0].fill_between(kout, kout*(mean[ill] - std[ill]), kout*(mean[ill] + std[ill]), alpha=0.5, facecolor='C{:d}'.format(ill), linewidth=0, label='mocks' if ill == 0 else None)
        lax[0].plot(kout, kout*model_conv[ill], linestyle='-', color='C{:d}'.format(ill), label='theory * conv' if ill == 0 else None)
    for ill, ell in enumerate(ells):
        lax[ill+1].plot(kout, (model_conv[ill] - mean[ill])/std[ill], linestyle='-', color='C{:d}'.format(ill))
        lax[ill+1].set_ylim(-4, 4)
        for offset in [-2., 2.]: lax[ill+1].axhline(offset, color='k', linestyle='--')
        lax[ill+1].set_ylabel(r'$\Delta P_{{{0:d}}} / \sigma_{{ P_{{{0:d}}} }}$'.format(ell))
    for ax in lax: ax.grid(True)
    lax[0].legend()
    lax[0].grid(True)
    lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    lax[-1].set_xlabel('$k$ [$h/\mathrm{Mpc}$]')
    logger.info('Saving figure to {}.'.format(plot_poles_fn))
    fig.savefig(plot_poles_fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(fig)


def plot_wedges():
    utils.mkdir(plot_dir)
    window = CatalogFFTWindow.load(window_fn.format('all')).wedges
    kin = window.xin[0]
    muedges = window.attrs['edges'][1]
    mask_positive = muedges >= 0.
    wedges = muedges[mask_positive]
    wedges = list(zip(wedges[:-1], wedges[1:]))
    masks = [slice(start, len(window.xout[0]), len(muedges) - 1) for start in range(np.sum(~mask_positive), len(muedges) - 1)]
    kout = [window.xout[0][mask, 0] for mask in masks]
    muout = [np.nanmean(window.xout[0][mask, 1]) for mask in masks]
    ellsin = [proj.ell for proj in window.projsin]
    model_theory = np.array([kaiser_model_poles(kin, ell=ell) for ell in ellsin])
    model_conv = window.dot(model_theory, unpack=False)
    model_conv = [model_conv[mask] - 1./nbar for mask in masks]
    model_theory = [kaiser_model_wedges(kin, wedge=wedge) - 1./nbar for wedge in wedges]
    mean, std = mock_mean('wedges')
    mean, std = mean.T[mask_positive[:-1]], std.T[mask_positive[:-1]]
    height_ratios = [max(len(wedges), 3)] + [1]*len(wedges)
    figsize = (6, 1.5*sum(height_ratios))
    fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios':height_ratios}, figsize=figsize, squeeze=True)
    fig.subplots_adjust(hspace=0)
    for imu, mu in enumerate(muout):
        lax[0].plot(kin, kin*model_theory[imu], linestyle=':', color='C{:d}'.format(imu), label='theory' if imu == 0 else None)
    for imu, mu in enumerate(muout):
        lax[0].fill_between(kout[imu], kout[imu]*(mean[imu] - std[imu]), kout[imu]*(mean[imu] + std[imu]), alpha=0.5, facecolor='C{:d}'.format(imu), linewidth=0, label='mocks' if imu == 0 else None)
        lax[0].plot(kout[imu], kout[imu]*model_conv[imu], linestyle='-', color='C{:d}'.format(imu), label='theory * conv' if imu == 0 else None)
    for imu, mu in enumerate(muout):
        lax[imu+1].plot(kout[imu], (model_conv[imu] - mean[imu])/std[imu], linestyle='-', color='C{:d}'.format(imu))
        lax[imu+1].set_ylim(-4, 4)
        for offset in [-2., 2.]: lax[imu+1].axhline(offset, color='k', linestyle='--')
        lax[imu+1].set_ylabel(r'$\Delta P / \sigma_{P}$')
    for ax in lax: ax.grid(True)
    lax[0].legend()
    lax[0].grid(True)
    lax[0].set_ylabel(r'$k P(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    lax[-1].set_xlabel('$k$ [$h/\mathrm{Mpc}$]')
    logger.info('Saving figure to {}.'.format(plot_wedges_fn))
    fig.savefig(plot_wedges_fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(fig)


def main(args=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--todo', type=str, help='what shoud I do? (typically: "mock", then "window", then "plot")', choices=['mock', 'window', 'plot'])
    parser.add_argument('--irun', nargs='*', type=int, default=[0, 1], help='ranges of mocks to run if "todo" is "mock" or icut (between 0 and ncuts) and ncuts if "todo" is "window"')

    opt = parser.parse_args(args=args)
    setup_logging()

    if opt.todo == 'mock':
        if len(opt.irun) == 2:
            opt.irun = range(opt.irun[0], opt.irun[1])
        for imock in opt.irun:
            run_mock(imock=imock)

    if opt.todo == 'window':
        run_window(opt.irun[0], ncuts=opt.irun[1])
        window = CatalogFFTWindow.concatenate_x(*(CatalogFFTWindow.load(window_fn.format(icut)) for icut in range(opt.irun[0] + 1)))
        window.save(window_fn.format('all'))

    if opt.todo == 'plot':
        plot_window()
        plot_window_integ()
        plot_poles()
        plot_wedges()


if __name__ == '__main__':

    main()
