"""
This script is dedicated to testing the interlacing correction.
First generate Gaussian mocks::

    (mpiexec -np 4) python mock_interlacing_periodic.py --todo mock --irun 0 20 # start - end of mock ids

Then compute window matrix::

    # you can split in as many parts as you want
    (mpiexec -np 4) python mock_interlacing_periodic.py --todo window --irun 0 3 # icut - ncuts
    (mpiexec -np 4) python mock_interlacing_periodic.py --todo window --irun 1 3
    (mpiexec -np 4) python mock_interlacing_periodic.py --todo window --irun 2 3

Then plot::

    python mock_interlacing_periodic.py --todo plot

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

from pypower import CatalogFFTPower, MeshFFTWindow, setup_logging


logger = logging.getLogger('InterlacingPeriodicWindow')


cosmo = DESI()
z = 1.
pklin = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
f = cosmo.get_fourier().sigma8_z(z=z, of='theta_cb') / cosmo.get_fourier().sigma8_z(z=z, of='delta_cb')
bias = 1.5
nbar = 1e-3
edgesin = np.linspace(0., 1., 201)

# Change paths here if you wish
base_dir = '_results'
plot_dir = '_plots'
mock_fn = os.path.join(base_dir, 'mock_interlacing_periodic_resampler_{resampler:}_interlacing_{interlacing:}_{imock:}.npy')
window_fn = os.path.join(base_dir, 'window_interlacing_periodic_{}.npy')
plot_poles_fn = os.path.join(plot_dir, 'power_window_interlacing_periodic_{}_{}_poles.png')
plot_wedges_fn = os.path.join(plot_dir, 'power_window_interlacing_periodic_{}_{}_wedges.png')

list_options = []
for resampler in ['cic', 'tsc', 'pcs']:
    for interlacing in [0, 2, 3, 4]:
        list_options.append({'resampler': resampler, 'interlacing': interlacing})


def run_mock(imock=0):
    seed = (imock + 1) * 42
    boxsize = 1000.; boxcenter = 0.; los = 'x'

    ells = (0, 2, 4); edges = ({'step': 0.005}, np.linspace(-1., 1., 7))

    mock = EulerianLinearMock(pklin, nmesh=700, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los)

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='tsc', compensate=True) + 1.

    for options in list_options:
        power = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], ells=ells, los=los, edges=edges,
                                boxsize=boxsize, boxcenter=boxcenter, nmesh=256, wrap=True, position_type='pos', **options)
        power.save(mock_fn.format(**options, imock=imock))


def run_window(icut=0, ncuts=1):
    power = CatalogFFTPower.load(mock_fn.format(**list_options[0], imock=0))
    start, stop = icut * (len(edgesin) - 1) // ncuts, (icut + 1) * (len(edgesin) - 1) // ncuts + 1
    window = MeshFFTWindow(edgesin=edgesin[start:stop], power_ref=power, periodic=True)
    window.save(window_fn.format(icut))


def kaiser_model_poles(k, ell=0):
    pk = bias**2 * pklin(k)
    beta = f / bias
    if ell == 0: return (1. + 2. / 3. * beta + 1. / 5. * beta**2) * pk + 1. / nbar
    if ell == 2: return (4. / 3. * beta + 4. / 7. * beta**2) * pk
    if ell == 4: return 8. / 35 * beta**2 * pk


def kaiser_model_wedges(k, wedge):
    from scipy import special
    ells = (0, 2, 4)
    pk = 0.
    for ell in ells:
        poly = special.legendre(ell).integ()(wedge)
        pk += kaiser_model_poles(k, ell) * (poly[1] - poly[0]) / (wedge[1] - wedge[0])
    return pk


def mock_mean(name='poles', **options):
    powers = []
    for fn in glob.glob(mock_fn.format(**options, imock='*')):
        powers.append(getattr(CatalogFFTPower.load(fn), name)(complex=False)[-1])
    return np.mean(powers, axis=0), np.std(powers, axis=0, ddof=1) / len(powers)**0.5


def plot_poles(sort_by='resampler'):
    window = MeshFFTWindow.load(window_fn.format('all')).poles
    kin = window.xin[0]
    kout = window.xout[0]
    ellsin = [proj.ell for proj in window.projsin]
    ells = [proj.ell for proj in window.projsout]
    model_theory = np.array([kaiser_model_poles(kin, ell=ell) for ell in ellsin])
    model_conv = window.dot(model_theory, unpack=True)
    model_conv[ells.index(0)] -= 1. / nbar
    height_ratios = [max(len(ells), 2)] + [1] * len(ells)
    figsize = (6, 1.5 * sum(height_ratios))
    mask = kout > 0.01

    options1 = []
    for options in list_options:
        if options[sort_by] not in options1: options1.append(options[sort_by])

    for option1 in options1:  # option1 is the value for sort_by, e.g. resampler
        loptions2 = []
        for options in list_options:
            if options[sort_by] == option1:
                loptions2.append({key: val for key, val in options.items() if key != sort_by})  # rest of the options, e.g. interlacing
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0)
        for ill, ell in enumerate(ells):
            lax[0].plot(kout[mask], kout[mask] * model_conv[ill][mask], linestyle='-', color='k', label='theory * window' if ill == 0 else None)
        for ioptions, options2 in enumerate(loptions2):
            mean, std = mock_mean(name='poles', **{sort_by: option1}, **options2)
            for ill, ell in enumerate(ells):
                lax[0].plot(kout[mask], kout[mask] * mean[ill][mask], linestyle='-', color='C{:d}'.format(ioptions), label=', '.join('{} = {}'.format(key, val) for key, val in options2.items()) if ill == 0 else None)
            for ill, ell in enumerate(ells):
                lax[ill + 1].plot(kout[mask], (mean[ill][mask] - model_conv[ill][mask]) / model_conv[ells.index(0)][mask], linestyle='-', color='C{:d}'.format(ioptions))
                lax[ill + 1].set_ylim(-0.05, 0.05)
                for offset in [-0.01, 0.01]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
                lax[ill + 1].set_ylabel(r'$\Delta P_{{{0:d}}} / P_{{0}}$'.format(ell))
        for ax in lax: ax.grid(True)
        lax[0].legend()
        lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        fn = plot_poles_fn.format(sort_by, option1)
        logger.info('Saving figure to {}.'.format(fn))
        fig.savefig(fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
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
        window = MeshFFTWindow.concatenate_x(*(MeshFFTWindow.load(window_fn.format(icut)) for icut in range(opt.irun[0] + 1)))
        window.save(window_fn.format('all'))

    if opt.todo == 'plot':
        plot_poles(sort_by='resampler')
        plot_poles(sort_by='interlacing')


if __name__ == '__main__':

    main()
