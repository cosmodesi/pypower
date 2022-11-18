import os
import tempfile

import numpy as np
from matplotlib import pyplot as plt
from mockfactory import Catalog

from pypower import CatalogFFTPower, CatalogMesh, MeshFFTWindow, CatalogFFTWindow, mpi, setup_logging
from pypower.fft_window import get_correlation_function_tophat_derivative

from test_fft_power import data_fn, randoms_fn


def test_deriv(plot=False):
    kedges = np.array([0.1, 0.12, 0.14])

    for ell in [0, 1, 2, 3, 4, 5]:

        fana = get_correlation_function_tophat_derivative(kedges, ell=ell)
        assert len(fana) == len(kedges) - 1
        sep = np.logspace(-3, 3, 1000)
        fnum = get_correlation_function_tophat_derivative(kedges, k=1. / sep[::-1], ell=ell)
        assert len(fnum) == len(kedges) - 1

        if plot:
            ax = plt.gca()
            ax.plot(sep, fana[1](sep), label='analytic')
            ax.plot(sep, fnum[1](sep), label='numerical')
            ax.legend()
            ax.set_xscale('log')
            plt.show()


def test_fft_window():

    boxsize = 2000.
    nmesh = 64
    kedges = np.linspace(0., 0.1, 7)
    ells = (0, 2)
    resampler = 'tsc'
    interlacing = 2
    dtype = 'f8'
    cdtype = 'c16'
    boxcenter = np.array([1e6, 0., 0.])[None, :]

    for los in ['x', 'firstpoint', 'endpoint']:

        data = Catalog.read(data_fn)
        randoms = Catalog.read(randoms_fn)

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
            power.mpicomm.Barrier()
            power = CatalogFFTPower.load(fn)

        edgesin = np.linspace(0.1, 0.11, 3)
        window = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power, position_type='pos', dtype=dtype)
        assert np.allclose(window.poles.weight, power.poles.wnorm)
        with tempfile.TemporaryDirectory() as tmp_dir:
            fn = data.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            window.save(fn)
            window.mpicomm.Barrier()
            window = CatalogFFTWindow.load(fn)
            window.mpicomm = data.mpicomm
            window.save(fn)

        mesh1 = CatalogMesh(data_positions=randoms['Position'], data_weights=randoms['Weight'], nmesh=power.attrs['nmesh'], boxsize=power.attrs['boxsize'], boxcenter=power.attrs['boxcenter'],
                            resampler=power.attrs['resampler1'], interlacing=power.attrs['interlacing1'], position_type='pos', dtype=dtype)
        window_mesh = MeshFFTWindow(mesh1, edgesin=edgesin, power_ref=power)
        assert np.allclose(window_mesh.poles.value, window.poles.value)
        assert np.allclose(window_mesh.poles.weight, power.poles.wnorm)
        bak = window_mesh.value.copy()
        window_mesh2 = window_mesh + window_mesh
        assert np.allclose(window_mesh2.value, window_mesh.value)
        assert np.allclose(window_mesh2.value, bak)

        mesh1 = CatalogMesh(data_positions=randoms['Position'], data_weights=randoms['Weight'], nmesh=power.attrs['nmesh'], boxsize=power.attrs['boxsize'], boxcenter=power.attrs['boxcenter'],
                            resampler=power.attrs['resampler1'], interlacing=power.attrs['interlacing1'], position_type='pos', dtype=dtype).to_mesh()
        mesh1_bak = mesh1.copy()
        window_mesh = MeshFFTWindow(mesh1, mesh2=mesh1, edgesin=edgesin, power_ref=power, wnorm=window.poles.wnorm, shotnoise=window_mesh.shotnoise)
        assert np.allclose(mesh1.value, mesh1_bak.value)
        assert np.allclose(window_mesh.poles.value, window.poles.value)

        mesh2 = mesh1.copy()
        mesh2_bak = mesh2.copy()
        window_mesh = MeshFFTWindow(mesh1, mesh2=mesh1, edgesin=edgesin, power_ref=power, wnorm=window.poles.wnorm, shotnoise=window_mesh.shotnoise)
        assert np.allclose(mesh1.value, mesh1_bak.value)
        assert np.allclose(mesh2.value, mesh2_bak.value)
        assert np.allclose(window_mesh.poles.value, window.poles.value)
        del mesh1, mesh2, mesh1_bak, mesh2_bak

        mesh1 = CatalogMesh(data_positions=randoms['Position'], data_weights=randoms['Weight'], nmesh=power.attrs['nmesh'], boxsize=power.attrs['boxsize'], boxcenter=power.attrs['boxcenter'],
                            resampler=power.attrs['resampler1'], interlacing=power.attrs['interlacing1'], position_type='pos', dtype=dtype).to_mesh().r2c()
        window_mesh = MeshFFTWindow(mesh1, edgesin=edgesin, power_ref=power, wnorm=window.poles.wnorm, shotnoise=window_mesh.shotnoise)
        assert np.allclose(window_mesh.poles.value, window.poles.value)

        power_f4 = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'],
                                   boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=kedges, position_type='pos', dtype='f4').poles
        window_f4 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power_f4, position_type='pos')
        assert window_f4.dtype.itemsize == 4

        windowc = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, power_ref=power, position_type='pos', dtype=cdtype)
        # print(windowc.poles.value/window.poles.value)
        # assert np.allclose(windowc.poles.value, window.poles.value, rtol=0.5)
        # if window.attrs['los_type'] == 'global':
        #    print(windowc.wedges.value/window.wedges.value)
        #    assert np.allclose(windowc.wedges.value, window.wedges.value, rtol=0.5)

        window1 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin[:2], projsin=window.poles.projsin, ells=(0,), power_ref=power, position_type='pos', dtype=dtype)
        window2 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin[1:], projsin=window.poles.projsin, ells=(0,), power_ref=power, position_type='pos', dtype=dtype)
        windowc = window1.concatenate_x(window1, window2)
        assert np.allclose(windowc.poles.value, window.poles.value[:, :len(window.poles.xout[0])])

        randoms['Position'][0] += boxsize
        window1 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, projsin=window.poles.projsin[:1], power_ref=power.poles, wrap=True, position_type='pos', dtype=dtype)
        window2 = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, projsin=window.poles.projsin[1:], power_ref=power.poles, wrap=True, position_type='pos', dtype=dtype)
        windowc = window1.concatenate_proj(window1, window2)
        assert np.allclose(windowc.poles.value, window.poles.value)

        randoms['Position'][0] -= boxsize
        projsin = [(ell, 0) for ell in ells]
        if los in ['firstpoint', 'endpoint']: projsin += [(ell, 1) for ell in range(1, max(ells) + 2, 2)]
        alpha = data['Weight'].csum() / randoms['Weight'].csum()
        window_noref = CatalogFFTWindow(randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'], edgesin=edgesin, projsin=projsin, edges=edges, ells=ells, los=los,
                                        boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos', wnorm=power.poles.wnorm / alpha**2, dtype=dtype)
        assert np.allclose(window_noref.poles.value, window.poles.value)

        positions = mpi.gather(randoms['Position'], mpiroot=0, mpicomm=catalog.mpicomm)
        weights = mpi.gather(randoms['Weight'], mpiroot=0, mpicomm=catalog.mpicomm)
        window_root = CatalogFFTWindow(randoms_positions1=positions, randoms_weights1=weights, edgesin=edgesin, power_ref=power, position_type='pos', dtype=dtype, mpiroot=0)
        assert np.allclose(window_root.poles.value, window.poles.value)

        if randoms.mpicomm.rank == 0:
            window_root = CatalogFFTWindow(randoms_positions1=positions, randoms_weights1=weights, edgesin=edgesin, power_ref=power, position_type='pos', dtype=dtype, mpicomm=mpi.COMM_SELF)
            assert np.allclose(window_root.poles.value, window.poles.value)

        window.poles.resum_input_odd_wide_angle()
        if window.attrs['los_type'] == 'global':
            window.wedges.resum_input_odd_wide_angle()
            # window_global_poles = window.poles

        window.poles.rebin_x(factorout=2)
        assert len(window.poles.xout[0]) == (len(kedges) - 1) // 2

        if window.attrs['los_type'] == 'global':
            window = MeshFFTWindow(edgesin=(0.03, 0.04), power_ref=power, periodic=True)
            assert not np.allclose(window.poles.value, 0.)


if __name__ == '__main__':

    setup_logging()
    # test_deriv()
    test_fft_window()
