import os
import tempfile

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo.fiducial import DESI
from mockfactory import LagrangianLinearMock, Catalog
from mockfactory.make_survey import RandomBoxCatalog

from pypower import MeshFFTPower, CatalogFFTPower, CatalogMesh, PowerSpectrumStatistics, mpi, utils, setup_logging
from pypower.fft_power import normalization, normalization_from_nbar, find_unique_edges, get_real_Ylm, project_to_basis


base_dir = 'catalog'
data_fn = os.path.join(base_dir, 'lognormal_data.fits')
randoms_fn = os.path.join(base_dir, 'lognormal_randoms.fits')


def save_lognormal():
    z = 1.
    boxsize = 600.
    boxcenter = 0.
    los = 'x'
    nbar = 1e-3
    bias = 2.0
    nmesh = 256
    seed = 42
    power = DESI().get_fourier().pk_interpolator().to_1d(z=z)
    f = 0.8
    mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias-1.)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=seed, resampler='cic', compensate=True)
    mock.set_rsd(f=f, los=los)
    #mock.set_rsd(f=f)
    data = mock.to_catalog()
    offset = mock.boxcenter - mock.boxsize / 2.
    data['Position'] = (data['Position'] - offset) % mock.boxsize + offset
    randoms = RandomBoxCatalog(nbar=4.*nbar, boxsize=boxsize, boxcenter=boxcenter, seed=44)

    for catalog in [data, randoms]:
        catalog['NZ'] = nbar*catalog.ones()
        catalog['WEIGHT_FKP'] = np.ones(catalog.size, dtype='f8')

    data.save_fits(data_fn)
    randoms.save_fits(randoms_fn)


def test_interp():
    x, y = (np.linspace(0., 10., 10),)*2
    from scipy.interpolate import UnivariateSpline, RectBivariateSpline
    assert UnivariateSpline(x, y, k=1, s=0, ext=3)(-1) == 0.
    assert RectBivariateSpline(x, y, y[:,None] * y, kx=1, ky=1, s=0)(12, 8, grid=False) == 80


def test_power_statistic():

    mpicomm = mpi.COMM_WORLD

    for dtype in ['f8', 'c16']:

        edges = np.linspace(0., 0.2, 11)
        modes = (edges[:-1] + edges[1:])/2.
        nmodes = np.ones_like(modes, dtype='i8')
        ells = (0, 2, 4)
        power = [np.ones_like(modes, dtype='c16')]*len(ells)
        power = PowerSpectrumStatistics(edges, modes, power, nmodes, ells, statistic='multipole')
        power_ref = power.copy()
        power.rebin(factor=2)
        assert power.power.shape[1] == power_ref.power.shape[1]//2 # poles are first dimension
        assert np.allclose(power.k, (power_ref.modes[0][::2] + power_ref.modes[0][1::2])/2.)
        assert np.allclose(power.kedges, np.linspace(0., 0.2, 6))
        assert power.shape == (modes.size//2,)
        assert np.allclose(power_ref[::2].power_nonorm, power.power_nonorm)
        power2 = power_ref.copy()
        power2.select((0., 0.1))
        assert np.all(power2.modes[0] <= 0.1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            #tmp_dir = '_tests'
            #power.mpicomm = mpicomm # to get a Barrier (otherwise the directory on root=0 may be deleted before other ranks access it)
            #fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            fn = os.path.join(tmp_dir, 'tmp.npy')
            power.save(fn)
            test = PowerSpectrumStatistics.load(fn)
            assert np.all(test.power == power.power)
        power2 = power.copy()
        power2.modes[0] = 1
        assert np.all(power.modes[0] == test.modes[0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            #tmp_dir = '_tests'
            fn = os.path.join(tmp_dir, 'tmp_poles.txt')
            power.save_txt(fn, complex=True)

        for complex in [False, True]:
            assert np.allclose(power(complex=complex, return_k=True)[1], power.get_power(complex=complex), equal_nan=True)
            assert np.allclose(power(complex=complex), power.get_power(complex=complex), equal_nan=True)
            assert np.isnan(power(k=-1., ell=0, complex=complex))
            assert not np.isnan(power(k=modes, complex=complex)).any()
            assert power(k=[0.1,0.2]).shape == (len(power.ells), 2)
            assert np.allclose(power(k=[0.1,0.2], ell=power.ells), power(k=[0.1,0.2]))

        edges = (edges, np.linspace(0., 1., 21))
        modes = np.meshgrid(*((e[:-1] + e[1:])/2 for e in edges), indexing='ij')
        nmodes = np.ones(tuple(len(e)-1 for e in edges), dtype='i8')
        power = np.ones_like(nmodes, dtype='c16')
        power = PowerSpectrumStatistics(edges, modes, power, nmodes, statistic='wedge')
        power_ref = power.copy()
        power.rebin(factor=(2, 2))
        assert power.power.shape[0] == power_ref.power.shape[0]//2
        assert power.modes[0].shape == (5, 10)
        assert not np.isnan(power(0.,0.))
        assert np.isnan(power(-1., 0.))
        power.rebin(factor=(1, 10))
        assert power.power_nonorm.shape == (5, 1)
        assert np.allclose(power_ref[::2,::2].power_nonorm, power.power_nonorm)
        assert power_ref[1:7:2].shape[0] == 3
        power2 = power_ref.copy()
        power2.select(None, (0., 0.5))
        assert np.all(power2.modes[1] <= 0.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            #tmp_dir = '_tests'
            fn = os.path.join(tmp_dir, 'tmp_wedges.txt')
            power.save_txt(fn, complex=True)

        for complex in [False, True]:
            assert np.allclose(power(complex=complex, return_k=True, return_mu=True)[2], power.get_power(complex=complex), equal_nan=True)
            assert np.allclose(power(complex=complex, return_k=True)[1], power.get_power(complex=complex), equal_nan=True)
            assert np.allclose(power(complex=complex), power.get_power(complex=complex), equal_nan=True)
            assert not np.isnan(power(0., 0., complex=complex))
            assert np.isnan(power([-1.]*5, 0., complex=complex)).all()
            assert np.isnan(power(-1., [0.]*5, complex=complex)).all()
            assert power(k=[0.1,0.2], mu=[0.3]).shape == (2, 1)


def test_ylm():
    rng = np.random.RandomState(seed=42)
    size = 1000
    x, y, z = [rng.uniform(0., 1., size) for i in range(3)]
    r = np.sqrt(x**2 + y**2 + z**2)
    r[r == 0.] = 1.
    xhat, yhat, zhat = x/r, y/r, z/r
    for ell in range(8):
        for m in range(-ell, ell+1):
            ylm = get_real_Ylm(ell, m)(xhat, yhat, zhat)
            ylm_scipy = get_real_Ylm(ell, m, modules='scipy')(xhat, yhat, zhat)
            assert np.allclose(ylm_scipy, ylm)


def test_find_edges():
    x = np.meshgrid(np.arange(10.), np.arange(10.), indexing='ij')
    x0 = np.ones(len(x), dtype='f8')
    edges = find_unique_edges(x, x0, xmin=0., xmax=np.inf, mpicomm=mpi.COMM_WORLD)


def test_project():

    z = 1.
    bias, nbar, nmesh, boxsize, boxcenter = 2.0, 1e-3, 64, 1000., 500.
    power = DESI().get_fourier().pk_interpolator().to_1d(z=z)
    mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=42, unitary_amplitude=False)
    # This is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias-1)
    mesh = mock.mesh_delta_r + 1.
    sum = mesh.csum()
    mesh = mesh.r2c()
    for islab in range(mesh.shape[0]):
        mesh[islab] = mesh[islab].conj() * mesh[islab]
    edges = (np.linspace(0., 0.1, 11), np.linspace(-1., 1., 5))
    result = project_to_basis(mesh, edges, los=(0, 0, 1), ells=None, antisymmetric=False, exclude_zero=False)
    assert len(result) == 2
    assert result[1] is None
    result = project_to_basis(mesh, edges, los=(0, 0, 1), ells=(0,), antisymmetric=False, exclude_zero=False)
    assert len(result) == 2
    assert result[1] is not None
    ells = (0, 2)
    result = project_to_basis(mesh, edges, los=(0, 0, 1), ells=ells, antisymmetric=False, exclude_zero=True)
    zero = sum**2/mesh.pm.Nmesh.prod()**2
    assert np.allclose(result[0][-1], [((e[0] <= 0.) & (e[1] > 0.))*zero for e in zip(edges[1][:-1], edges[1][1:])])
    from scipy import special
    assert np.allclose(result[1][-1], [(2*ell+1) * zero * special.legendre(ell)(0.) for ell in ells])
    #power = MeshFFTPower(mesh_real, ells=(0, 2), los='x', edges=({'step':0.001}, np.linspace(-1., 1., 3)))


def test_field_power():

    z = 1.
    bias, nbar, nmesh, boxsize, boxcenter = 2.0, 1e-3, 64, 1000., 500.
    power = DESI().get_fourier().pk_interpolator().to_1d(z=z)
    mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=42, unitary_amplitude=False)
    # This is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias-1)
    mesh_real = mock.mesh_delta_r + 1.

    kedges = np.linspace(0., 0.4, 11)
    muedges = np.linspace(-1., 1., 6)
    dk = kedges[1] - kedges[0]
    ells = (0, 1, 2, 3, 4)

    def get_ref_power(mesh, los):
        from nbodykit.lab import FFTPower, FieldMesh
        return FFTPower(FieldMesh(mesh), mode='2d', poles=ells, Nmu=len(muedges) - 1, los=los, dk=dk, kmin=kedges[0], kmax=kedges[-1])

    def get_mesh_power(mesh, los, mesh2=None, edges=(kedges, muedges)):
        mesh_bak = mesh.copy()
        if mesh2 is not None: mesh2_bak = mesh2.copy()
        toret = MeshFFTPower(mesh, mesh2=mesh2, ells=ells, los=los, edges=edges)
        assert np.allclose(mesh.value, mesh_bak.value)
        if mesh2 is not None: assert np.allclose(mesh2.value, mesh2_bak.value)
        return toret

    def check_wedges(power, ref_power):
        for imu, muavg in enumerate(power.muavg):
            mask = power.nmodes[:,imu] > 0
            if hasattr(ref_power, 'k'):
                k, mu, modes, pk = ref_power.k[:,imu], ref_power.mu[:,imu], ref_power.nmodes[:,imu], ref_power.power[:,imu] + ref_power.shotnoise
            else:
                k, mu, modes, pk = ref_power['k'][:,imu], ref_power['mu'][:,imu], ref_power['modes'][:,imu], ref_power['power'][:,imu].conj()
                #n = (power.edges[1][imu] <= 0.) & (power.edges[1][imu+1] > 0.)
                #assert power.nmodes[0, imu] == modes[0] - n # we do not include k = 0
                #mask &= (power.edges[0][:-1] > 0.)
            assert np.allclose(power.nmodes[mask,imu], modes[mask], atol=1e-6, rtol=3e-3, equal_nan=True)
            assert np.allclose(power.k[mask,imu], k[mask], atol=1e-6, rtol=3e-3, equal_nan=True)
            assert np.allclose(power.mu[mask,imu], mu[mask], atol=1e-6, rtol=3e-3, equal_nan=True)
            assert np.allclose(power(mu=muavg)[mask] + power.shotnoise, pk[mask], atol=1e-6, rtol=1e-3, equal_nan=True)

    def check_poles(power, ref_power):
        for ell in power.ells:
            mask = power.nmodes > 0
            if hasattr(ref_power, 'k'):
                k, modes, pk = ref_power.k, ref_power.nmodes, ref_power(ell=ell) + ref_power.shotnoise
            else:
                k, modes, pk = ref_power['k'], ref_power['modes'], ref_power['power_{}'.format(ell)].conj()
                #assert power.nmodes[0] == modes[0] - 1
                #mask &= (power.edges[0][:-1] > 0.)
            assert np.allclose(power.nmodes[mask], modes[mask], atol=1e-6, rtol=5e-3)
            assert np.allclose(power.k[mask], k[mask], atol=1e-6, rtol=5e-3)
            mask[0] = False
            assert np.allclose(power(ell=ell)[mask] + (ell == 0)*power.shotnoise, pk[mask], atol=1e-3, rtol=1e-6)
            assert np.allclose(power(ell=ell)[0] + (ell == 0)*power.shotnoise, pk[0], atol=1e-3, rtol=2e-3)

    from pypower import ParticleMesh
    pm = ParticleMesh(BoxSize=mesh_real.pm.BoxSize, Nmesh=mesh_real.pm.Nmesh, dtype='c16', comm=mesh_real.pm.comm)
    mesh_complex = pm.create(type='real')
    mesh_complex[...] = mesh_real[...]

    for los in [(1,0,0), (0,1,0), (0,0,1)]:
        ref_power = get_ref_power(mesh_complex, los)
        ref_kedges = ref_power.power.edges['k']
        power = get_mesh_power(mesh_real, los, edges=(ref_kedges, muedges))
        check_wedges(power.wedges, ref_power.power)
        check_poles(power.poles, ref_power.poles)

        c_power = get_mesh_power(mesh_complex, los, edges=(ref_kedges, muedges))
        check_wedges(power.wedges, c_power.wedges)
        check_poles(power.poles, c_power.poles)

        power = get_mesh_power(mesh_real.r2c(), los, mesh2=mesh_real.r2c(), edges=(ref_kedges, muedges))
        check_wedges(power.wedges, ref_power.power)
        check_poles(power.poles, ref_power.poles)

        c_power = get_mesh_power(mesh_complex.r2c(), los, edges=(ref_kedges, muedges))
        check_wedges(power.wedges, c_power.wedges)
        check_poles(power.poles, c_power.poles)

        #power = get_mesh_power(mesh_real, los, edges=(np.insert(ref_kedges, 0, -0.1), muedges))
        #assert np.allclose(power.wedges.nmodes[0], [0, 0, 1, 0, 0]) and power.wedges.k[0,2] == 0.
        #assert power.poles.nmodes[0] == 1 and power.poles.k[0] == 0.
        #check_wedges(power.wedges[1:], ref_power.power)
        #check_poles(power.poles[1:], ref_power.poles)


def test_mesh_power():
    boxsize = 600.
    boxcenter = 0.
    nmesh = 128
    kedges = np.linspace(0., 0.3, 11)
    muedges = np.linspace(-1., 1., 5)
    dk = kedges[1] - kedges[0]
    ells = (0, 1, 2, 4)
    resampler = 'cic'
    interlacing = 2
    dtype = 'f8'
    data = Catalog.load_fits(data_fn)

    def get_ref_power(data, los, dtype='c16'):
        los_array = [1. if ax == los else 0. for ax in 'xyz']
        from nbodykit.lab import FFTPower
        mesh = data.to_nbodykit().to_mesh(position='Position', BoxSize=boxsize, Nmesh=nmesh, resampler=resampler, interlaced=bool(interlacing), compensated=True, dtype=dtype)
        return FFTPower(mesh, mode='2d', poles=ells, Nmu=len(muedges) - 1, los=los_array, dk=dk, kmin=kedges[0], kmax=kedges[-1])

    def get_mesh_power(data, los, edges=(kedges, muedges), dtype=dtype, as_cross=False):
        mesh = CatalogMesh(data_positions=data['Position'], boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos', dtype=dtype)
        if as_cross:
            mesh2 = CatalogMesh(data_positions=data['Position'].T, boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='xyz')
        else:
            mesh2 = None
        return MeshFFTPower(mesh, mesh2=mesh2, ells=ells, los=los, edges=edges)

    def get_mesh_power_compensation(data, los):
        mesh = CatalogMesh(data_positions=data['Position'], boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos', dtype=dtype).to_mesh()
        return MeshFFTPower(mesh, ells=ells, los=los, edges=(kedges, muedges), compensations=resampler)

    def check_wedges(power, ref_power):
        for imu, mu in enumerate(power.muavg):
            assert np.allclose(power.nmodes[:,imu], ref_power['modes'][:,imu], atol=1e-6, rtol=3e-3, equal_nan=True)
            assert np.allclose(power.k[:,imu], ref_power['k'][:,imu], atol=1e-6, rtol=3e-3, equal_nan=True)
            assert np.allclose(power(mu=mu) + power.shotnoise, ref_power['power'][:,imu].conj(), atol=1e-6, rtol=3e-3, equal_nan=True)

    def check_poles(power, ref_power):
        for ell in power.ells:
            #assert np.allclose(power(ell=ell).real + (ell == 0)*power.shotnoise, ref_power.poles['power_{}'.format(ell)].real, atol=1e-6, rtol=3e-3)
            # Exact if offset = 0. in to_mesh()
            assert np.allclose(power.nmodes, ref_power['modes'], atol=1e-6, rtol=5e-3)
            assert np.allclose(power.k, ref_power['k'], atol=1e-6, rtol=5e-3)
            assert np.allclose(power(ell=ell) + (ell == 0)*power.shotnoise, ref_power['power_{}'.format(ell)].conj(), atol=1e-2, rtol=1e-2)

    for los in ['x', 'z']:

        ref_power = get_ref_power(data, los=los)
        ref_kedges = ref_power.power.edges['k']

        list_options = []
        list_options.append({'los':los, 'edges':(ref_kedges, muedges)})
        list_options.append({'los':[1. if ax == los else 0. for ax in 'xyz'], 'edges':(ref_kedges, muedges)})
        list_options.append({'los':los, 'edges':({'min':ref_kedges[0], 'max':ref_kedges[-1], 'step':ref_kedges[1] - ref_kedges[0]}, muedges)})
        list_options.append({'los':los, 'edges':(ref_kedges, muedges), 'dtype':'f4'})
        list_options.append({'los':los, 'edges':(ref_kedges, muedges[:-1]), 'dtype':'f4'})
        list_options.append({'los':los, 'edges':(ref_kedges, muedges[:-1]), 'dtype':'c8'})

        for options in list_options:
            power = get_mesh_power(data, **options)
            with tempfile.TemporaryDirectory() as tmp_dir:
                #tmp_dir = '_tests'
                fn = power.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
                fn_txt = power.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.txt'), root=0)
                power.save(fn)
                power.poles.save_txt(fn_txt)
                power = MeshFFTPower.load(fn)
            check_wedges(power.wedges, ref_power.power)

            if power.wedges.edges[1][-1] == 1.:
                check_poles(power.poles, ref_power.poles)

    power = get_mesh_power(data, los='x').poles

    power_compensation = get_mesh_power_compensation(data, los='x').poles
    for ill, ell in enumerate(power.ells):
        assert np.allclose(power_compensation.power_nonorm[ill]/power_compensation.wnorm, power.power_nonorm[ill]/power.wnorm)

    power_cross = get_mesh_power(data, los='x', as_cross=True).poles
    for ell in ells:
        assert np.allclose(power_cross(ell=ell) - (ell == 0)*power.shotnoise, power(ell=ell))

    randoms = Catalog.load_fits(randoms_fn)

    def get_ref_power(data, randoms, los, dtype='c16'):
        los_array = [1. if ax == los else 0. for ax in 'xyz']
        from nbodykit.lab import FFTPower
        mesh_data = data.to_nbodykit().to_mesh(position='Position', BoxSize=boxsize, Nmesh=nmesh, resampler=resampler, interlaced=bool(interlacing), compensated=True, dtype=dtype)
        mesh_randoms = randoms.to_nbodykit().to_mesh(position='Position', BoxSize=boxsize, Nmesh=nmesh, resampler=resampler, interlaced=bool(interlacing), compensated=True, dtype=dtype)
        mesh = mesh_data.compute() - mesh_randoms.compute()
        return FFTPower(mesh, mode='2d', poles=ells, Nmu=len(muedges) - 1, los=los_array, dk=dk, kmin=kedges[0], kmax=kedges[-1]+1e-9)

    def get_power(data, randoms, los, dtype=dtype):
        mesh = CatalogMesh(data_positions=data['Position'], randoms_positions=randoms['Position'], boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos', dtype=dtype)
        wnorm = normalization(mesh, uniform=True)
        return MeshFFTPower(mesh, ells=ells, los=los, edges=(kedges, muedges), wnorm=wnorm)

    ref_power = get_ref_power(data, randoms, los='x')
    power = get_power(data, randoms, los='x')
    check_wedges(power.wedges, ref_power.power)
    check_poles(power.poles, ref_power.poles)


def test_normalization():
    boxsize = 1000.
    nmesh = 128
    resampler = 'tsc'
    interlacing = False
    boxcenter = np.array([3000.,0.,0.])[None,:]
    los = None
    dtype = 'f8'
    data = Catalog.load_fits(data_fn)
    randoms = Catalog.load_fits(randoms_fn)
    for catalog in [data, randoms]:
        catalog['Position'] += boxcenter
        catalog['Weight'] = catalog.ones()
    mesh = CatalogMesh(data_positions=data['Position'], data_weights=data['Weight'], randoms_positions=randoms['Position'], randoms_weights=randoms['Weight'],
                       boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos', dtype=dtype)
    old = normalization_from_nbar(randoms['NZ'], randoms['Weight'], data_weights=data['Weight'], mpicomm=mesh.mpicomm)
    new = normalization(mesh)
    assert np.allclose(new, old, atol=0, rtol=1e-1)


def test_catalog_power():
    boxsize = 1000.
    nmesh = 128
    kedges = np.linspace(0., 0.3, 6)
    dk = kedges[1] - kedges[0]
    ells = (0, 1, 2, 3, 4)
    resampler = 'tsc'
    interlacing = 2
    boxcenter = np.array([3000.,0.,0.])[None,:]
    los = None
    dtype = 'f8'
    data = Catalog.load_fits(data_fn)
    randoms = Catalog.load_fits(randoms_fn)
    for catalog in [data, randoms]:
        catalog['Position'] += boxcenter
        catalog['Weight'] = catalog.ones()

    def get_ref_power(data, randoms, dtype='c16'):
        from nbodykit.lab import FKPCatalog, ConvolvedFFTPower
        fkp = FKPCatalog(data.to_nbodykit(), randoms.to_nbodykit(), nbar='NZ')
        mesh = fkp.to_mesh(position='Position', comp_weight='Weight', nbar='NZ', BoxSize=boxsize, Nmesh=nmesh, resampler=resampler, interlaced=bool(interlacing), compensated=True, dtype=dtype)
        return ConvolvedFFTPower(mesh, poles=ells, dk=dk, kmin=kedges[0], kmax=kedges[-1]+1e-9)

    def get_catalog_power(data, randoms, position_type='pos', edges=kedges, dtype=dtype, as_cross=False, **kwargs):
        data_positions, randoms_positions = data['Position'], randoms['Position']
        if position_type == 'xyz':
            data_positions, randoms_positions = data['Position'].T, randoms['Position'].T
        elif position_type == 'rdd':
            data_positions, randoms_positions = utils.cartesian_to_sky(data['Position'].T), utils.cartesian_to_sky(randoms['Position'].T)
        if as_cross:
            kwargs.update(data_positions2=data_positions, data_weights2=data['Weight'], randoms_positions2=randoms_positions, randoms_weights2=randoms['Weight'])
        return CatalogFFTPower(data_positions1=data_positions, data_weights1=data['Weight'], randoms_positions1=randoms_positions, randoms_weights1=randoms['Weight'],
                               boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=edges, position_type=position_type, dtype=dtype, **kwargs)

    def get_catalog_mesh_power(data, randoms, dtype=dtype):
        mesh = CatalogMesh(data_positions=data['Position'], data_weights=data['Weight'], randoms_positions=randoms['Position'], randoms_weights=randoms['Weight'],
                           boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos', dtype=dtype)
        return MeshFFTPower(mesh, ells=ells, los=los, edges=kedges)

    def get_mesh_power(data, randoms, dtype=dtype, as_complex=False, as_cross=False):
        mesh = CatalogMesh(data_positions=data['Position'], data_weights=data['Weight'], randoms_positions=randoms['Position'], randoms_weights=randoms['Weight'],
                           boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos', dtype=dtype)
        wnorm = np.real(normalization(mesh))
        shotnoise = mesh.unnormalized_shotnoise()/wnorm
        field = mesh.to_mesh()
        if as_complex: field = field.r2c()
        field2 = None
        if as_cross: field2 = field
        field_bak = field.copy()
        toret = MeshFFTPower(field, mesh2=field2, ells=ells, los=los, edges=kedges, wnorm=wnorm, shotnoise=shotnoise, compensations=[mesh.compensation, mesh.compensation], boxcenter=mesh.boxcenter)
        assert np.allclose(field.value, field_bak.value)
        if as_cross: assert np.allclose(field2.value, field_bak.value)
        return toret

    def check_poles(power, ref_power):
        norm = power.wnorm
        ref_norm = ref_power.attrs['randoms.norm']
        for ell in power.ells:
            # precision is 1e-7 if offset = self.boxcenter - self.boxsize/2. + 0.5*self.boxsize
            ref = ref_power.poles['power_{}'.format(ell)]
            if power.attrs['los_type'] == 'firstpoint': ref = ref.conj()
            assert np.allclose((power(ell=ell) + (ell == 0)*power.shotnoise)*norm/ref_norm, ref, atol=1e-6, rtol=5e-2)
            assert np.allclose(power.k, ref_power.poles['k'], atol=1e-6, rtol=5e-3)
            assert np.allclose(power.nmodes, ref_power.poles['modes'], atol=1e-6, rtol=5e-3)

    ref_power = get_ref_power(data, randoms)
    f_power = get_catalog_power(data, randoms, dtype='f8')
    c_power = get_catalog_power(data, randoms, dtype='c16')
    ref_kedges = ref_power.poles.edges['k']

    list_options = []
    list_options.append({'position_type':'pos'})
    list_options.append({'position_type':'xyz'})
    list_options.append({'position_type':'rdd'})
    list_options.append({'edges':{'min':ref_kedges[0], 'max':ref_kedges[-1], 'step':ref_kedges[1] - ref_kedges[0]}})

    for options in list_options:
        power = get_catalog_power(data, randoms, **options)

        with tempfile.TemporaryDirectory() as tmp_dir:
            #tmp_dir = '_tests'
            fn = power.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
            fn_txt = power.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.txt'), root=0)
            power.save(fn)
            power.poles.save_txt(fn_txt)
            power = CatalogFFTPower.load(fn)

        check_poles(power.poles, ref_power)
        for ell in ells:
            atol = 2e-1 if ell % 2 == 0 else 1e-5
            assert np.allclose(power.poles(ell=ell).imag, c_power.poles(ell=ell).imag, atol=atol, rtol=1e-3)
            atol = 2e-1 if ell % 2 else 1e-5
            assert np.allclose(power.poles(ell=ell).real, c_power.poles(ell=ell).real, atol=atol, rtol=1e-3)

    power_mesh = get_catalog_mesh_power(data, randoms)
    for ell in ells:
        assert np.allclose(power_mesh.poles(ell=ell), f_power.poles(ell=ell))

    power_mesh = get_mesh_power(data, randoms, as_complex=False)
    for ell in ells:
        assert np.allclose(power_mesh.poles(ell=ell), f_power.poles(ell=ell))

    power_mesh = get_mesh_power(data, randoms, as_complex=True, as_cross=True)
    for ell in ells:
        assert np.allclose(power_mesh.poles(ell=ell), f_power.poles(ell=ell))

    power_cross = get_catalog_power(data, randoms, as_cross=True)
    for ell in ells:
        assert np.allclose(power_cross.poles(ell=ell) - (ell == 0)*f_power.shotnoise, f_power.poles(ell=ell))

    position = data['Position'].copy()
    data['Position'][0] += boxsize
    power_wrap = get_catalog_power(data, randoms, position_type='pos', edges=kedges, wrap=True, boxcenter=f_power.attrs['boxcenter'], dtype=dtype)
    for ell in ells:
        assert np.allclose(power_wrap.poles(ell=ell), f_power.poles(ell=ell))
    data['Position'] = position

    def get_catalog_mesh_no_randoms_power(data):
        mesh = CatalogMesh(data_positions=data['Position'], boxsize=600., nmesh=nmesh, wrap=True, resampler=resampler, interlacing=interlacing, position_type='pos')
        return MeshFFTPower(mesh, ells=ells, los=los, edges=kedges)

    def get_catalog_no_randoms_power(data):
        return CatalogFFTPower(data_positions1=data['Position'], ells=ells, los=los, edges=kedges, boxsize=600., nmesh=nmesh, wrap=True, resampler=resampler, interlacing=interlacing, position_type='pos')

    ref_power = get_catalog_mesh_no_randoms_power(data)
    power = get_catalog_no_randoms_power(data)
    for ell in ells:
        assert np.allclose(power.poles(ell=ell), ref_power.poles(ell=ell))

    def get_catalog_shifted_power(data, randoms, as_cross=False):
        kwargs = {}
        if as_cross:
            kwargs = dict(data_positions2=data['Position'].T, data_weights2=data['Weight'],
                          randoms_positions2=randoms['Position'].T, randoms_weights2=randoms['Weight'],
                          shifted_positions2=randoms['Position'].T, shifted_weights2=randoms['Weight'])
        return CatalogFFTPower(data_positions1=data['Position'].T, data_weights1=data['Weight'],
                               randoms_positions1=randoms['Position'].T, randoms_weights1=randoms['Weight'],
                               shifted_positions1=randoms['Position'].T, shifted_weights1=randoms['Weight'], **kwargs,
                               boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=kedges, position_type='xyz')

    power_shifted = get_catalog_shifted_power(data, randoms)
    for ell in ells:
        assert np.allclose(power_shifted.poles(ell=ell), f_power.poles(ell=ell))

    power_shifted = get_catalog_shifted_power(data, randoms, as_cross=True)
    for ell in ells:
        assert np.allclose(power_shifted.poles(ell=ell) - (ell == 0)*f_power.shotnoise, f_power.poles(ell=ell))

    def get_catalog_shifted_no_randoms_power(data, randoms):
        return CatalogFFTPower(data_positions1=data['Position'], shifted_positions1=randoms['Position'], ells=ells, los=los, edges=kedges, boxsize=boxsize, nmesh=nmesh, wrap=True, resampler=resampler, interlacing=interlacing, position_type='pos')

    power_shifted = get_catalog_shifted_no_randoms_power(data, randoms)
    for ell in ells:
        assert np.allclose(power_shifted.poles(ell=ell)*power_shifted.wnorm/f_power.wnorm, f_power.poles(ell=ell))


def test_mpi():

    boxsize = 1000.
    nmesh = 128
    kedges = np.linspace(0., 0.1, 6)
    dk = kedges[1] - kedges[0]
    ells = (0,)
    resampler = 'tsc'
    interlacing = 2
    boxcenter = np.array([3000.,0.,0.])[None,:]
    dtype = 'f8'
    cdtype = 'c16'
    los = None
    data = Catalog.load_fits(data_fn, mpicomm=mpi.COMM_WORLD)
    randoms = Catalog.load_fits(randoms_fn, mpicomm=mpi.COMM_WORLD)
    for catalog in [data, randoms]:
        catalog['Position'] += boxcenter
        catalog['Weight'] = catalog.ones()

    def run(mpiroot=None, mpicomm=mpi.COMM_WORLD):
        return CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'],
                               boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=kedges, position_type='pos',
                               dtype=dtype, mpiroot=mpiroot, mpicomm=mpicomm).poles

    ref_power = run(mpiroot=None)
    for catalog in [data, randoms]:
        catalog['Position'] = mpi.gather_array(catalog['Position'], root=0, mpicomm=catalog.mpicomm)
        catalog['Weight'] = mpi.gather_array(catalog['Weight'], root=0, mpicomm=catalog.mpicomm)

    power_root = run(mpiroot=0)
    for ell in ref_power.ells:
        assert np.allclose(power_root(ell=ell), ref_power(ell=ell))

    if data.mpicomm.rank == 0:
        power_root = run(mpiroot=0, mpicomm=mpi.COMM_SELF)
        for ell in ref_power.ells:
            assert np.allclose(power_root(ell=ell), ref_power(ell=ell))


def test_interlacing():

    from matplotlib import pyplot as plt
    boxsize = 1000.
    nmesh = 128
    kedges = {'min':0., 'step':0.005}
    ells = (0,)
    resampler = 'ngp'
    boxcenter = np.array([3000.,0.,0.])[None,:]

    data = Catalog.load_fits(data_fn)
    randoms = Catalog.load_fits(randoms_fn)
    for catalog in [data, randoms]:
        catalog['Position'] += boxcenter
        catalog['Weight'] = catalog.ones()

    def run(interlacing=2):
        return CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], randoms_weights1=randoms['Weight'],
                               boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los='firstpoint', edges=kedges, position_type='pos').poles

    for interlacing, linestyle in zip([False, 2, 3, 4], ['-', '--', ':', '-.']):
        power = run(interlacing=interlacing)
        for ill, ell in enumerate(power.ells):
            plt.plot(power.k, power.k * power(ell=ell).real, color='C{:d}'.format(ill), linestyle=linestyle, label='interlacing = {}'.format(interlacing))
    plt.legend()
    plt.show()


if __name__ == '__main__':

    setup_logging()
    #save_lognormal()
    #test_mesh_power()
    #test_interlacing()
    #test_mesh_power()
    test_power_statistic()
    test_find_edges()
    test_ylm()
    test_field_power()
    test_mesh_power()
    test_catalog_power()
    test_normalization()
    test_mpi()
