"""
This script is dedicated to testing the approximate window matrix for cutsky mocks, in the flat sky approximation.
First generate Gaussian mocks::

    (mpiexec -np 4) python mock_window_smooth_global_los.py --todo mock --irun 0 20 # start - end of mock ids

Then compute window matrix::

    (mpiexec -np 4) python mock_window_smooth_global_los.py --todo window

Then plot::

    python mock_window_smooth_global_los.py --todo plot

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

from pypower import CatalogFFTPower, CatalogSmoothWindow, PowerSpectrumOddWideAngleMatrix, PowerSpectrumSmoothWindowMatrix, setup_logging


logger = logging.getLogger('SmoothWindowGlobalLOS')


cosmo = DESI()
z = 1.
pklin = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
f = cosmo.get_fourier().sigma8_z(z=z, of='theta_cb')/cosmo.get_fourier().sigma8_z(z=z, of='delta_cb')
bias = 1.5
boxsize = 1000.
boxcenter = np.array([0., 0., 0.])
nbar = 1e-3

# Change paths here if you wish
base_dir = '_results'
mock_fn = os.path.join(base_dir, 'mock_smooth_global_los_{}.npy')
window_poles_fn = os.path.join(base_dir, 'window_smooth_global_los_poles.npy')
window_fn = os.path.join(base_dir, 'window_smooth_global_los_all.npy')
plot_window_fn = os.path.join(base_dir, 'window_smooth_global_los_poles.png')
plot_poles_fn = os.path.join(base_dir, 'power_smooth_global_los_poles.png')


def run_mock(imock=0):
    seed = (imock + 1) * 42
    nmesh = 512; los = 'x'

    mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los)

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    randoms = RandomBoxCatalog(nbar=10*nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='tsc', compensate=True) + 1.

    ells = (0, 2, 4); edges = np.linspace(0., 0.4, 81)
    power = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], ells=ells, los=los, edges=edges,
                            boxsize=2000., boxcenter=boxcenter, nmesh=256, resampler='tsc', interlacing=3, position_type='pos')
    power.save(mock_fn.format(imock))


def run_window():
    power = CatalogFFTPower.load(mock_fn.format(0)).poles
    randoms = RandomBoxCatalog(nbar=10*nbar, boxsize=boxsize, boxcenter=boxcenter, seed=42)
    edges = {'step': 0.0005}
    windows = []
    boxes = (100000, 10000, 2000)
    for box in boxes:
        windows.append(CatalogSmoothWindow(randoms_positions1=randoms['Position'], power_ref=power, edges=edges, boxsize=box, position_type='pos'))
    window = CatalogSmoothWindow.concatenate(*windows, frac_nyq=0.8).poles
    window.save(window_poles_fn)
    sep = np.geomspace(1e-4, 4e3, 1024*16) # configuration space separation for FFTlog
    wm = PowerSpectrumSmoothWindowMatrix(power.k, projsin=power.ells, projsout=power.ells, window=window, sep=sep, kin_rebin=4, kin_lim=(0., 0.5))
    wm.save(window_fn)


def kaiser_model(k, ell=0):
    pk = bias**2*pklin(k)
    beta = f/bias
    if ell == 0: return (1. + 2./3.*beta + 1./5.*beta**2)*pk + 1./nbar
    if ell == 2: return (4./3.*beta + 4./7.*beta**2)*pk
    if ell == 4: return 8./35*beta**2*pk


def mock_mean(name='poles'):
    powers = []
    for fn in glob.glob(mock_fn.format('*')):
        powers.append(getattr(CatalogFFTPower.load(fn), name)(complex=False))
    return np.mean(powers, axis=0), np.std(powers, axis=0, ddof=1)/len(powers)**0.5


def plot_window():
    window = PowerSpectrumSmoothWindow.load(window_poles_fn)
    window_real = window.to_real(sep=np.geomspace(1e-2, 4e3, 2048))
    ax = plt.gca()
    for iproj, proj in enumerate(window_real.projs):
        ax.plot(window_real.sep, window_real(proj=proj), label=proj.latex(inline=True))
    ax.legend()
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_xlabel('$s$ [$\mathrm{Mpc}/h$]')
    ax.set_ylabel(r'$W_{\ell}^{(n)}(s)$')
    plt.gcf().savefig(plot_window_fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(fig)


def plot_poles():
    window = PowerSpectrumSmoothWindowMatrix.load(window_fn)
    kin = window.xin[0]
    kout = window.xout[0]
    ellsin = [proj.ell for proj in window.projsin]
    ells = [proj.ell for proj in window.projsout]
    model_theory = np.array([kaiser_model(kin, ell=ell) for ell in ellsin])
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
        lax[0].plot(kout, kout*model_conv[ill], linestyle='--', color='C{:d}'.format(ill), label='theory * conv' if ill == 0 else None)
    for ill, ell in enumerate(ells):
        lax[ill+1].plot(kout, (model_conv[ill] - mean[ill])/std[ill], linestyle='--', color='C{:d}'.format(ill))
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


def main(args=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--todo', type=str, help='what shoud I do? (typically: "mock", then "window", then "plot")', choices=['mock', 'window', 'plot'])
    parser.add_argument('--irun', nargs='*', type=int, default=[0, 1], help='ranges of mocks to run if "todo" is "mock"')

    opt = parser.parse_args(args=args)
    setup_logging()

    if opt.todo == 'mock':
        if len(opt.irun) == 2:
            opt.irun = range(opt.irun[0], opt.irun[1])
        for imock in opt.irun:
            run_mock(imock=imock)

    if opt.todo == 'window':
        run_window()

    if opt.todo == 'plot':
        plot_poles()


if __name__ == '__main__':

    main()
