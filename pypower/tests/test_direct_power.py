import os
import tempfile

import numpy as np
from scipy import special
from mpi4py import MPI

from pypower import DirectPower, CatalogFFTPower, mpi, utils, setup_logging


def generate_catalogs(size=100, boxsize=(1000,)*3, offset=(1000.,0.,0.), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size)*b for o,b in zip(offset, boxsize)]
        weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64*n_bitwise_weights)], dtype=np.uint64)
        #weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(33)], dtype=np.uint64)
        #weights = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions+weights)
    return toret


def diff(position1, position2):
    return [p2-p1 for p1,p2 in zip(position1,position2)]


def midpoint(position1, position2):
    return [p2+p1 for p1,p2 in zip(position1,position2)]


def norm(position):
    return (sum(p**2 for p in position))**0.5


def dotproduct(position1, position2):
    return sum(x1*x2 for x1,x2 in zip(position1,position2))


def dotproduct_normalized(position1, position2):
    return dotproduct(position1, position2)/(norm(position1)*norm(position2))


def wiip(weights, nrealizations=None, noffset=1, default_value=0.):
    denom = noffset + utils.popcount(*weights)
    mask = denom == 0
    denom[mask] = 1.
    toret = nrealizations/denom
    toret[mask] = default_value
    return toret


def wpip_single(weights1, weights2, nrealizations=None, noffset=1, default_value=0.):
    denom = noffset + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1, weights2))
    return default_value if denom == 0 else nrealizations/denom


def wiip_single(weights, nrealizations=None, noffset=1, default_value=0.):
    denom = noffset + utils.popcount(*weights)
    return default_value if denom == 0 else nrealizations/denom


def get_weight(xyz1, xyz2, weights1, weights2, n_bitwise_weights=0, twopoint_weights=None, nrealizations=None, noffset=1, default_value=0., weight_type='auto'):
    weight = 1
    if nrealizations is not None:
        weight *= wpip_single(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)
    if twopoint_weights is not None:
        sep_twopoint_weights = twopoint_weights.sep
        twopoint_weights = twopoint_weights.weight
        costheta = sum(x1*x2 for x1,x2 in zip(xyz1,xyz2))/(norm(xyz1)*norm(xyz2))
        if (sep_twopoint_weights[0] <= costheta < sep_twopoint_weights[-1]):
            ind_costheta = np.searchsorted(sep_twopoint_weights, costheta, side='right', sorter=None) - 1
            frac = (costheta - sep_twopoint_weights[ind_costheta])/(sep_twopoint_weights[ind_costheta+1] - sep_twopoint_weights[ind_costheta])
            weight *= (1-frac)*twopoint_weights[ind_costheta] + frac*twopoint_weights[ind_costheta+1]
    if weight_type == 'inverse_bitwise_minus_individual':
        weight -= wiip_single(weights1[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)\
                 * wiip_single(weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)
    for w1, w2 in zip(weights1[n_bitwise_weights:], weights2[n_bitwise_weights:]):
        weight *= w1 * w2
    return weight


def ref_theta(modes, limits, data1, data2=None, boxsize=None, los='midpoint', ells=(0, 2, 4), **kwargs):
    toret = [np.zeros_like(modes, dtype='c16') for ell in ells]
    legendre = [special.legendre(ell) for ell in ells]
    if data2 is None: data2 = data1
    npairs = 0
    for xyzw1 in zip(*data1):
        for xyzw2 in zip(*data2):
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            dist = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1))) # min to avoid rounding errors
            if dist > 0 and limits[0] <= dist < limits[-1]:
                dxyz = diff(xyz1, xyz2)
                dist = norm(dxyz)
                if dist > 0:
                    npairs += 1
                    if los == 'midpoint': mu = dotproduct_normalized(dxyz, midpoint(xyz1, xyz2))
                    elif los == 'endpoint': mu = dotproduct_normalized(dxyz, xyz2)
                    elif los == 'firstpoint': mu = dotproduct_normalized(dxyz, xyz1)
                    weights1, weights2 = xyzw1[3:], xyzw2[3:]
                    weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                    for ill, ell in enumerate(ells):
                        toret[ill] += weight * (2*ell + 1) * (-1j)**ell * special.spherical_jn(ell, modes * dist) * legendre[ill](mu)
    return np.asarray(toret)


def ref_s(modes, limits, data1, data2=None, boxsize=None, los='midpoint', ells=(0, 2, 4), **kwargs):
    if los not in ['firstpoint', 'endpoint', 'midpoint']:
        los = [1 if i == 'xyz'.index(los) else 0 for i in range(3)]
    toret = [np.zeros_like(modes, dtype='c16') for ell in ells]
    legendre = [special.legendre(ell) for ell in ells]
    if data2 is None: data2 = data1
    for xyzw1 in zip(*data1):
        for xyzw2 in zip(*data2):
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            dxyz = diff(xyz2, xyz1)
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    if dxyz[idim] > 0.5*b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5*b: dxyz[idim] += b
            dist = norm(dxyz)
            if dist > 0 and limits[0] <= dist < limits[-1]:
                if los == 'midpoint': mu = dotproduct_normalized(dxyz, midpoint(xyz1, xyz2))
                elif los == 'endpoint': mu = dotproduct_normalized(dxyz, xyz2)
                elif los == 'firstpoint': mu = dotproduct_normalized(dxyz, xyz1)
                else: mu = dotproduct_normalized(dxyz, los)
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                for ill, ell in enumerate(ells):
                    toret[ill] += weight * (2*ell + 1) * (-1j)**ell * special.spherical_jn(ell, modes * dist) * legendre[ill](mu)
    return np.asarray(toret)


def test_bitwise_weight():
    from pypower.direct_power import get_inverse_probability_weight
    size = 10
    n_bitwise_weights = 2
    rng = np.random.RandomState(seed=42)
    weights = [utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64*n_bitwise_weights)], dtype=np.uint64) for i in range(2)]
    nrealizations = 1 + len(weights[0]) * 64
    wpip = get_inverse_probability_weight(*weights, noffset=1, nrealizations=None, default_value=1.)
    ref = [wpip_single([w[ii] for w in weights[0]], [w[ii] for w in weights[1]],
                      nrealizations=nrealizations, noffset=1, default_value=1.) for ii in range(size)]
    assert np.allclose(wpip, ref)


def test_direct_power():
    ref_funcs = {'theta':ref_theta, 's':ref_s}
    list_engine = ['kdtree']
    modes = np.linspace(0.01, 0.1, 11)
    size = 100
    boxsize = (1000,)*3
    from pypower.direct_power import KDTreeDirectPowerEngine
    _slab_npairs_max = KDTreeDirectPowerEngine._slab_npairs_max
    list_options = []
    list_options.append({})
    list_options.append({'los':'midpoint'})
    list_options.append({'los':'firstpoint'})
    list_options.append({'los':'endpoint', 'slab_npairs_max':10})
    list_options.append({'autocorr':True})
    list_options.append({'n_individual_weights':1})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'weight_type':'inverse_bitwise_minus_individual', 'slab_npairs_max':10})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'iip':1, 'dtype':'f4'})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'bitwise_type': 'i4', 'iip':1})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'bitwise_type': 'i4', 'iip':1, 'limit_type':'s'})
    list_options.append({'n_individual_weights':2, 'n_bitwise_weights':2, 'iip':2, 'position_type':'rdd', 'weight_attrs':{'nrealizations':42, 'noffset':3}})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':2, 'iip':2, 'weight_attrs':{'noffset':0, 'default_value':0.8}})
    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
    twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))
    list_options.append({'twopoint_weights':twopoint_weights, 'n_bitwise_weights':2, 'weight_type':'inverse_bitwise_minus_individual'})
    list_options.append({'autocorr':True, 'n_individual_weights':2, 'n_bitwise_weights':2, 'twopoint_weights':twopoint_weights, 'dtype':'f4'})

    for engine in list_engine:
        for options in list_options:
            options = options.copy()
            n_individual_weights = options.pop('n_individual_weights',0)
            n_bitwise_weights = options.pop('n_bitwise_weights',0)
            data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=42)
            #data1, data2 = data1[:3], data2[:3]
            #data1 = data1[:3] + [data1[3]*data1[4]]
            #data2 = data2[:3] + [data2[3]*data2[4]]
            limit_type = options.pop('limit_type', 'theta')
            if limit_type == 'theta':
                limits = (0., 1.)
            else:
                limits = (0., 50.)
            autocorr = options.pop('autocorr', False)
            options.setdefault('boxsize', None)
            options.setdefault('los', 'x' if options['boxsize'] is not None else 'firstpoint')
            bin_type = options.pop('bin_type', 'auto')
            mpicomm = options.pop('mpicomm', None)
            bitwise_type = options.pop('bitwise_type', None)
            iip = options.pop('iip', False)
            position_type = options.pop('position_type', 'xyz')
            dtype = options.pop('dtype', None)

            slab_npairs_max = options.pop('slab_npairs_max', None)
            if slab_npairs_max is not None:
                KDTreeDirectPowerEngine._slab_npairs_max = slab_npairs_max
            else:
                KDTreeDirectPowerEngine._slab_npairs_max = _slab_npairs_max

            refoptions = options.copy()
            weight_attrs = refoptions.pop('weight_attrs', {}).copy()

            def setdefaultnone(di, key, value):
                if di.get(key, None) is None:
                    di[key] = value

            setdefaultnone(weight_attrs, 'nrealizations', n_bitwise_weights * 64 + 1)
            setdefaultnone(weight_attrs, 'noffset', 1)
            set_default_value = 'default_value' in weight_attrs
            if set_default_value:
                for w in data1[3:3+n_bitwise_weights] + data2[3:3+n_bitwise_weights]: w[:] = 0 # set to zero to make sure default_value is used
            setdefaultnone(weight_attrs, 'default_value', 0)
            refdata1, refdata2 = data1.copy(), data2.copy()
            mpicomm = mpi.COMM_WORLD
            refdata1 = [mpi.gather_array(d, root=None, mpicomm=mpicomm) for d in refdata1]
            refdata2 = [mpi.gather_array(d, root=None, mpicomm=mpicomm) for d in refdata2]

            def dataiip(data):
                kwargs = {name: weight_attrs[name] for name in ['nrealizations', 'noffset', 'default_value']}
                return data[:3] + [wiip(data[3:3+n_bitwise_weights], **kwargs)] + data[3+n_bitwise_weights:]

            if iip:
                refdata1 = dataiip(refdata1)
                refdata2 = dataiip(refdata2)
            if iip == 1:
                data1 = dataiip(data1)
            elif iip == 2:
                data2 = dataiip(data2)
            if iip:
                n_bitwise_weights = 0
                weight_attrs['nrealizations'] = None

            itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
            tol = {'atol':1e-8, 'rtol':1e-2} if itemsize <= 4 else {'atol':1e-8, 'rtol':1e-6}

            if dtype is not None:
                for ii in range(len(refdata1)):
                    if np.issubdtype(refdata1[ii].dtype, np.floating):
                        refdata1[ii] = np.asarray(refdata1[ii], dtype=dtype)
                        refdata2[ii] = np.asarray(refdata2[ii], dtype=dtype)

            twopoint_weights = refoptions.pop('twopoint_weights', None)
            if twopoint_weights is not None:
                twopoint_weights = TwoPointWeight(np.cos(np.radians(twopoint_weights.sep[::-1], dtype=dtype)), np.asarray(twopoint_weights.weight[::-1], dtype=dtype))

            ref = ref_funcs[limit_type](modes, limits, refdata1, data2=None if autocorr else refdata2, n_bitwise_weights=n_bitwise_weights, twopoint_weights=twopoint_weights, **refoptions, **weight_attrs)

            if bitwise_type is not None and n_bitwise_weights > 0:

                def update_bit_type(data):
                    return data[:3] + utils.reformat_bitarrays(*data[3:3+n_bitwise_weights], dtype=bitwise_type) + data[3+n_bitwise_weights:]

                data1 = update_bit_type(data1)
                data2 = update_bit_type(data2)

            if position_type != 'xyz':

                def update_pos_type(data):
                    rdd = list(utils.cartesian_to_sky(data[:3]))
                    return rdd + data[3:]

                data1 = update_pos_type(data1)
                data2 = update_pos_type(data2)

            def run(**kwargs):
                return DirectPower(modes, positions1=data1[:3], positions2=None if autocorr else data2[:3],
                                   weights1=data1[3:], weights2=None if autocorr else data2[3:], position_type=position_type,
                                   limits=limits, limit_type=limit_type, engine=engine, **kwargs, **options)

            test = run()
            assert np.allclose(test.power_nonorm, ref, **tol)

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = test.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
                test.save(fn)
                test2 = DirectPower.load(fn)
                assert np.allclose(test.power_nonorm, ref, **tol)

            mpiroot = 0
            data1 = [mpi.gather_array(d, root=mpiroot, mpicomm=mpicomm) for d in data1]
            data2 = [mpi.gather_array(d, root=mpiroot, mpicomm=mpicomm) for d in data2]
            test_mpi = run(mpiroot=mpiroot)
            assert np.allclose(test_mpi.power_nonorm, test.power_nonorm, **tol)

            if test.mpicomm.rank == 0:
                test_mpi = run(mpiroot=mpiroot, mpicomm=MPI.COMM_SELF)
            assert np.allclose(test_mpi.power_nonorm, test.power_nonorm, **tol)


def test_catalog_power():
    nmesh = 100
    kedges = np.linspace(0., 0.1, 6)
    ells = (0, 2)
    resampler = 'tsc'
    interlacing = 2
    boxcenter = np.array([3000.,0.,0.])[None,:]
    dtype = 'f8'
    data1, data2 = generate_catalogs(size=10000, boxsize=(1000.,)*3, n_individual_weights=1, n_bitwise_weights=2, seed=42)
    randoms1, randoms2 = generate_catalogs(size=10000, boxsize=(1000.,)*3, n_individual_weights=1, n_bitwise_weights=0, seed=84)
    limits = (0., 1.)
    limit_type = 'theta'

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_limits=limits, direct_limit_type=limit_type)
    direct = DirectPower(power.poles.k, positions1=data1[:3], weights1=data1[3:], position_type='xyz',
                         ells=ells, limits=limits, limit_type=limit_type, weight_type='inverse_bitwise_minus_individual')
    assert np.allclose(power.poles.power_direct_nonorm, direct.power_nonorm)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            data_positions2=data2[:3], data_weights2=data2[3:], randoms_positions2=randoms2[:3], randoms_weights2=randoms2[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_limits=limits, direct_limit_type=limit_type)
    direct = DirectPower(power.poles.k, positions1=data1[:3], positions2=data2[:3], weights1=data1[3:], weights2=data2[3:], position_type='xyz',
                         ells=ells, limits=limits, limit_type=limit_type, weight_type='inverse_bitwise_minus_individual')
    assert np.allclose(power.poles.power_direct_nonorm, direct.power_nonorm)

    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
    twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_limits=limits, direct_limit_type=limit_type, D1D2_twopoint_weights=twopoint_weights, D1R2_twopoint_weights=twopoint_weights)
    direct1 = DirectPower(power.poles.k, positions1=data1[:3], weights1=data1[3:], position_type='xyz',
                         ells=ells, limits=limits, limit_type=limit_type, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)
    direct2 = DirectPower(power.poles.k, positions1=data1[:3], positions2=randoms1[:3], weights1=data1[3:], weights2=randoms1[3:], position_type='xyz',
                         ells=ells, limits=limits, limit_type=limit_type, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)
    assert np.allclose(power.poles.power_direct_nonorm, direct1.power_nonorm - 2*direct2.power_nonorm)


if __name__ == '__main__':

    setup_logging()
    test_bitwise_weight()
    test_direct_power()
    test_catalog_power()
