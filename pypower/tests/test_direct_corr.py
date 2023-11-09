import os
import tempfile

import numpy as np
from scipy import special
from mpi4py import MPI

from pypower import DirectCorr, CatalogFFTPower, mpi, utils, setup_logging

from test_direct_power import generate_catalogs, MemoryMonitor


def diff(position1, position2):
    return [p2 - p1 for p1, p2 in zip(position1, position2)]


def midpoint(position1, position2):
    return [p2 + p1 for p1, p2 in zip(position1, position2)]


def norm(position):
    return (sum(p**2 for p in position))**0.5


def dotproduct(position1, position2):
    return sum(x1 * x2 for x1, x2 in zip(position1, position2))


def dotproduct_normalized(position1, position2):
    return dotproduct(position1, position2) / (norm(position1) * norm(position2))


def wiip(weights, nrealizations=None, noffset=1, default_value=0.):
    denom = noffset + utils.popcount(*weights)
    mask = denom == 0
    denom[mask] = 1.
    toret = nrealizations / denom
    toret[mask] = default_value
    return toret


def wpip_single(weights1, weights2, nrealizations=None, noffset=1, default_value=0.):
    denom = noffset + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1, weights2))
    return default_value if denom == 0 else nrealizations / denom


def wiip_single(weights, nrealizations=None, noffset=1, default_value=0.):
    denom = noffset + utils.popcount(*weights)
    return default_value if denom == 0 else nrealizations / denom


def get_weight(xyz1, xyz2, weights1, weights2, n_bitwise_weights=0, twopoint_weights=None, nrealizations=None, noffset=1, default_value=0., weight_type='auto'):
    weight = 1
    if nrealizations is not None:
        weight *= wpip_single(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)
    if twopoint_weights is not None:
        sep_twopoint_weights = twopoint_weights.sep
        twopoint_weights = twopoint_weights.weight
        costheta = sum(x1 * x2 for x1, x2 in zip(xyz1, xyz2)) / (norm(xyz1) * norm(xyz2))
        if (sep_twopoint_weights[0] <= costheta < sep_twopoint_weights[-1]):
            ind_costheta = np.searchsorted(sep_twopoint_weights, costheta, side='right', sorter=None) - 1
            frac = (costheta - sep_twopoint_weights[ind_costheta]) / (sep_twopoint_weights[ind_costheta + 1] - sep_twopoint_weights[ind_costheta])
            weight *= (1 - frac) * twopoint_weights[ind_costheta] + frac * twopoint_weights[ind_costheta + 1]
    if weight_type == 'inverse_bitwise_minus_individual':
        # print(1./nrealizations * weight, 1./nrealizations * wiip_single(weights1[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)\
        #          * wiip_single(weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value))
        weight -= wiip_single(weights1[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)\
                  * wiip_single(weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)
    for w1, w2 in zip(weights1[n_bitwise_weights:], weights2[n_bitwise_weights:]):
        weight *= w1 * w2
    return weight


def ref_theta(edges, data1, data2=None, boxsize=None, los='midpoint', ells=(0, 2, 4), autocorr=False, selection_attrs=None, **kwargs):
    if data2 is None: data2 = data1
    counts = np.zeros(len(edges) - 1, dtype='f8')
    sep = np.zeros(len(edges) - 1, dtype='f8')
    poles = [np.zeros(len(edges) - 1, dtype='c16') for ell in ells]
    legendre = [special.legendre(ell) for ell in ells]
    selection_attrs = dict(selection_attrs or {})
    theta_limits = selection_attrs.get('theta', None)
    if theta_limits is not None:
        costheta_limits = np.cos(np.deg2rad(theta_limits)[::-1])
    rp_limits = selection_attrs.get('rp', None)
    npairs = 0
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            if theta_limits is not None:
                theta = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1)))  # min to avoid rounding errors
                if theta < theta_limits[0] or theta >= theta_limits[1]: continue
                #if all(x1 == x2 for x1, x2 in zip(xyz1, xyz2)): costheta = 1.
                #else: costheta = min(dotproduct_normalized(xyz1, xyz2), 1)
                #if costheta <= costheta_limits[0] or costheta > costheta_limits[1]: continue
            dxyz = diff(xyz1, xyz2)
            dist = norm(dxyz)
            npairs += 1
            if dist > 0:
                if los == 'midpoint': mu = dotproduct_normalized(dxyz, midpoint(xyz1, xyz2))
                elif los == 'endpoint': mu = dotproduct_normalized(dxyz, xyz2)
                elif los == 'firstpoint': mu = dotproduct_normalized(dxyz, xyz1)
            else:
                mu = 0.
            if dist < edges[0] or dist >= edges[-1]: continue
            if rp_limits is not None:
                rp2 = (1. - mu**2) * dist**2
                if rp2 < rp_limits[0]**2 or rp2 >= rp_limits[1]**2: continue
            ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
            weights1, weights2 = xyzw1[3:], xyzw2[3:]
            weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
            counts[ind] += weight
            sep[ind] += weight * dist
            for ill, ell in enumerate(ells):
                poles[ill][ind] += weight * (2 * ell + 1) * legendre[ill](mu)
    return np.asarray(poles), sep / counts


def test_direct_corr():
    list_engine = ['kdtree', 'corrfunc']
    edges = np.linspace(0., 100, 11)
    size = 100
    boxsize = (100,) * 3
    from pypower.direct_corr import KDTreeDirectCorrEngine
    _slab_npairs_max = KDTreeDirectCorrEngine._slab_npairs_max
    _slab_nobjs_max = KDTreeDirectCorrEngine._slab_nobjs_max
    list_options = []

    for autocorr in [False, True]:
        list_options.append({'autocorr': autocorr})
        # one-column of weights
        list_options.append({'autocorr': autocorr, 'weights_one': [1]})
        # position type
        for position_type in ['rdd', 'pos', 'xyz']:
            list_options.append({'autocorr': autocorr, 'position_type': position_type})
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'nthreads': 2})
        # pip
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'weight_type': 'inverse_bitwise_minus_individual', 'slab_npairs_max': 10, 'slab_nobjs_max': 10})
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'dtype': 'f4'})

        list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'bitwise_type': 'i4'})
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'iip': 1})
        if not autocorr:
            list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'iip': 2})
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'weight_attrs': {'nrealizations': 42, 'noffset': 3}})
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 2, 'weight_attrs': {'noffset': 0, 'default_value': 0.8}})
        # los
        for los in ['midpoint', 'firstpoint', 'endpoint']:
            list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': los, 'ells': (0, 1, 2, 3, 4, 5, 6, 8)})
        # selection
        for los in ['midpoint', 'firstpoint', 'endpoint']:
            list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'los': los, 'selection_attrs': {'rp': (0., 10.)}})
            list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'los': los, 'selection_attrs': {'theta': (0., 5.)}})
        # twopoint_weights
        from collections import namedtuple
        TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
        twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))
        list_options.append({'autocorr': autocorr, 'twopoint_weights': twopoint_weights})
        list_options.append({'autocorr': autocorr, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights, 'weight_type': 'inverse_bitwise_minus_individual'})
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights})

    for engine in list_engine:
        for options in list_options:
            options = options.copy()
            print(engine, options)
            nthreads = options.pop('nthreads', None)
            weights_one = options.pop('weights_one', [])
            n_individual_weights = options.pop('n_individual_weights', 0)
            n_bitwise_weights = options.pop('n_bitwise_weights', 0)
            data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=42)
            data1 = [np.concatenate([d, d]) for d in data1]  # that will get us some pairs at sep = 0
            selection_attrs = options.pop('selection_attrs', {'theta': (0., 1.)})
            autocorr = options.pop('autocorr', False)
            options.setdefault('boxsize', None)
            options.setdefault('los', 'x' if options['boxsize'] is not None else 'firstpoint')
            mpicomm = options.pop('mpicomm', None)
            bitwise_type = options.pop('bitwise_type', None)
            iip = options.pop('iip', False)
            position_type = options.pop('position_type', 'xyz')
            dtype = options.pop('dtype', None)

            KDTreeDirectCorrEngine._slab_npairs_max = options.pop('slab_npairs_max', _slab_npairs_max)
            KDTreeDirectCorrEngine._slab_nobjs_max = options.pop('slab_nobjs_max', _slab_nobjs_max)

            ref_options = options.copy()
            weight_attrs = ref_options.pop('weight_attrs', {}).copy()

            def setdefaultnone(di, key, value):
                if di.get(key, None) is None:
                    di[key] = value

            setdefaultnone(weight_attrs, 'nrealizations', n_bitwise_weights * 64 + 1)
            setdefaultnone(weight_attrs, 'noffset', 1)
            set_default_value = 'default_value' in weight_attrs
            if set_default_value:
                for w in data1[3:3 + n_bitwise_weights] + data2[3:3 + n_bitwise_weights]: w[:] = 0  # set to zero to make sure default_value is used
            setdefaultnone(weight_attrs, 'default_value', 0)
            data1_ref, data2_ref = data1.copy(), data2.copy()
            mpicomm = mpi.COMM_WORLD
            # data1_ref = [mpi.gather(d, mpiroot=None, mpicomm=mpicomm) for d in data1_ref]
            # data2_ref = [mpi.gather(d, mpiroot=None, mpicomm=mpicomm) for d in data2_ref]

            def dataiip(data):
                kwargs = {name: weight_attrs[name] for name in ['nrealizations', 'noffset', 'default_value']}
                return data[:3] + [wiip(data[3:3 + n_bitwise_weights], **kwargs)] + data[3 + n_bitwise_weights:]

            if iip:
                data1_ref = dataiip(data1_ref)
                data2_ref = dataiip(data2_ref)
            if iip == 1:
                data1 = dataiip(data1)
            elif iip == 2:
                data2 = dataiip(data2)
            if iip:
                n_bitwise_weights = 0
                weight_attrs['nrealizations'] = None

            itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
            tol = {'atol': 1e-8, 'rtol': 1e-2} if itemsize <= 4 else {'atol': 1e-8, 'rtol': 1e-5}

            if dtype is not None:
                for ii in range(len(data1_ref)):
                    if np.issubdtype(data1_ref[ii].dtype, np.floating):
                        data1_ref[ii] = np.asarray(data1_ref[ii], dtype=dtype)
                        data2_ref[ii] = np.asarray(data2_ref[ii], dtype=dtype)

            twopoint_weights = ref_options.pop('twopoint_weights', None)
            if twopoint_weights is not None:
                twopoint_weights = TwoPointWeight(np.cos(np.radians(twopoint_weights.sep[::-1], dtype=dtype)), np.asarray(twopoint_weights.weight[::-1], dtype=dtype))
            poles_ref, sep_ref = ref_theta(edges, data1_ref, data2=data2_ref if not autocorr else None, autocorr=autocorr, n_bitwise_weights=n_bitwise_weights, twopoint_weights=twopoint_weights, selection_attrs=selection_attrs, **ref_options, **weight_attrs)

            if bitwise_type is not None and n_bitwise_weights > 0:

                def update_bit_type(data):
                    return data[:3] + utils.reformat_bitarrays(*data[3:3 + n_bitwise_weights], dtype=bitwise_type) + data[3 + n_bitwise_weights:]

                data1 = update_bit_type(data1)
                data2 = update_bit_type(data2)

            if position_type == 'rdd':

                def update_pos_type(data):
                    rdd = list(utils.cartesian_to_sky(data[:3]))
                    return rdd + data[3:]

                data1 = update_pos_type(data1)
                data2 = update_pos_type(data2)

            for label, catalog in zip([1, 2], [data1, data2]):
                if label in weights_one:
                    catalog.append(np.ones_like(catalog[0]))

            def run(pass_none=False, pass_zero=False, **kwargs):
                positions1 = data1[:3]
                positions2 = data2[:3]
                weights1 = data1[3:]
                weights2 = data2[3:]

                def get_zero(arrays):
                    if isinstance(arrays, list):
                        return [array[:0] if array is not None else None for array in arrays]
                    elif arrays is not None:
                        return arrays[:0]
                    return None

                if pass_zero:
                    positions1 = get_zero(positions1)
                    positions2 = get_zero(positions2)
                    weights1 = get_zero(weights1)
                    weights2 = get_zero(weights2)

                if position_type == 'pos':
                    positions1 = np.array(positions1).T
                    positions2 = np.array(positions2).T

                return DirectCorr(edges, positions1=None if pass_none else positions1, positions2=None if pass_none or autocorr else positions2,
                                  weights1=None if pass_none else weights1, weights2=None if pass_none or autocorr else weights2, position_type=position_type,
                                  selection_attrs=selection_attrs, engine=engine, nthreads=nthreads, **kwargs, **options)

            data1 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data1]
            data2 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data2]
            test = run(mpiroot=None)
            test.to_power(modes=np.linspace(0.01, 0.2, 10))

            print(test.corr_nonorm, poles_ref)
            print(test.corr_nonorm - poles_ref)
            assert np.allclose(test.corr_nonorm, poles_ref, **tol)
            assert np.allclose(test.sep, sep_ref, equal_nan=True, **tol)
            test_zero = run(mpiroot=None, pass_none=False, pass_zero=True)
            assert np.allclose(test_zero.corr_nonorm, 0.)

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = test.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
                test.save(fn)
                test.mpicomm.Barrier()
                test2 = DirectCorr.load(fn)
                assert np.allclose(test.corr_nonorm, poles_ref, **tol)
                fn = os.path.join(tmp_dir, 'tmp.npy')
                test2.save(fn)

            mpiroot = 0
            data1_ref, data2_ref = data1, data2
            data1 = [mpi.gather(d, mpiroot=mpiroot, mpicomm=mpicomm) for d in data1]
            data2 = [mpi.gather(d, mpiroot=mpiroot, mpicomm=mpicomm) for d in data2]
            test_mpi = run(mpiroot=mpiroot)
            assert np.allclose(test_mpi.corr_nonorm, test.corr_nonorm, **tol)
            test_mpi = run(mpiroot=mpiroot, pass_none=mpicomm.rank > 0)
            assert np.allclose(test_mpi.corr_nonorm, test.corr_nonorm, **tol)
            if test.mpicomm.rank == mpiroot:
                test_mpi = run(mpiroot=mpiroot, mpicomm=MPI.COMM_SELF)
            assert np.allclose(test_mpi.corr_nonorm, test.corr_nonorm, **tol)
            data1 = [d if mpicomm.rank == mpiroot else dref[:0] for d, dref in zip(data1, data1_ref)]
            data2 = [d if mpicomm.rank == mpiroot else dref[:0] for d, dref in zip(data2, data2_ref)]
            test_mpi = run(mpiroot=None)
            assert np.allclose(test_mpi.corr_nonorm, poles_ref, **tol)
            test_mpi = run(mpiroot=mpiroot, pass_zero=True)


def test_catalog_power():
    nmesh = 300
    kedges = np.linspace(0., 0.1, 6)
    ells = (0, 2)
    resampler = 'tsc'
    interlacing = 2
    data1, data2 = generate_catalogs(size=50000, boxsize=(2000.,) * 3, n_individual_weights=1, n_bitwise_weights=2, seed=42)
    randoms1, randoms2 = generate_catalogs(size=100000, boxsize=(2000.,) * 3, n_individual_weights=1, n_bitwise_weights=0, seed=84)
    selection_attrs = {'theta': (0., 1.)}

    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
    twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))

    def get_alpha(attrs, label):
        label1, label2 = label
        labelc = {'D': 'data', 'R': 'randoms', 'S': 'shifted'}
        return attrs['sum_data_weights1'] * attrs['sum_data_weights2'] / (attrs['sum_{}_weights1'.format(labelc[label1])] * attrs['sum_{}_weights2'.format(labelc[label2])])

    direct_edges = {'step': 0.2, 'max': 100}
    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs=selection_attrs, direct_edges=direct_edges)
    direct = DirectCorr(direct_edges, positions1=data1[:3], weights1=data1[3:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual')
    assert np.allclose(power.poles.corr_direct_nonorm, direct.corr_nonorm)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            data_positions2=data2[:3], data_weights2=data2[3:], randoms_positions2=randoms2[:3], randoms_weights2=randoms2[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs=selection_attrs, direct_edges=direct_edges)
    direct = DirectCorr(direct_edges, positions1=data1[:3], positions2=data2[:3], weights1=data1[3:], weights2=data2[3:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual')
    assert np.allclose(power.poles.corr_direct_nonorm, direct.corr_nonorm)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[-1:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            data_positions2=data2[:3], data_weights2=data2[-1:], randoms_positions2=randoms2[:3], randoms_weights2=randoms2[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs=selection_attrs, direct_edges=direct_edges)
    direct = DirectCorr(direct_edges, positions1=data1[:3], positions2=data2[:3], weights1=data1[-1:], weights2=data2[-1:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').corr_nonorm
    direct -= get_alpha(power.attrs, 'DR') * DirectCorr(direct_edges, positions1=data1[:3], positions2=randoms2[:3], weights1=data1[-1:], weights2=randoms2[3:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').corr_nonorm
    direct -= get_alpha(power.attrs, 'RD') * DirectCorr(direct_edges, positions1=randoms1[:3], positions2=data2[:3], weights1=randoms1[3:], weights2=data2[-1:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').corr_nonorm
    direct += get_alpha(power.attrs, 'RR') * DirectCorr(direct_edges, positions1=randoms1[:3], positions2=randoms2[:3], weights1=randoms1[3:], weights2=randoms2[3:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').corr_nonorm
    assert np.allclose(power.poles.corr_direct_nonorm, -direct)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[-1:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            data_positions2=data2[:3], data_weights2=data2[-1:], randoms_positions2=randoms2[:3], randoms_weights2=randoms2[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs={**selection_attrs, 'counts': ['D1D2', 'D1R2']}, direct_edges=direct_edges)
    direct = DirectCorr(direct_edges, positions1=data1[:3], positions2=data2[:3], weights1=data1[-1:], weights2=data2[-1:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').corr_nonorm
    direct -= get_alpha(power.attrs, 'DR') * DirectCorr(direct_edges, positions1=data1[:3], positions2=randoms2[:3], weights1=data1[-1:], weights2=randoms2[3:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').corr_nonorm
    assert np.allclose(power.poles.corr_direct_nonorm, -direct)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs=selection_attrs, direct_edges=direct_edges, D1D2_twopoint_weights=twopoint_weights, D1R2_twopoint_weights=twopoint_weights)

    direct_D1D2 = DirectCorr(direct_edges, positions1=data1[:3], weights1=data1[3:], position_type='xyz',
                             ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)
    direct_D1R2 = DirectCorr(direct_edges, positions1=data1[:3], positions2=randoms1[:3], weights1=data1[3:], weights2=randoms1[3:], position_type='xyz',
                             ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)
    direct_R1D2 = DirectCorr(direct_edges, positions1=randoms1[:3], positions2=data1[:3], weights1=randoms1[3:], weights2=data1[3:], position_type='xyz',
                             ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)
    assert np.allclose(power.poles.corr_direct_nonorm, get_alpha(power.attrs, 'DD') * direct_D1D2.corr_nonorm - get_alpha(power.attrs, 'DR') * direct_D1R2.corr_nonorm - get_alpha(power.attrs, 'RD') * direct_R1D2.corr_nonorm)

    power = power.poles
    power2 = power + power
    assert np.allclose(power2.power, power.power, equal_nan=True)
    assert np.allclose(power2.wnorm, 2. * power.wnorm, equal_nan=True)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], data_weights2=data2[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz', direct_attrs={'nthreads': 4},
                            direct_selection_attrs=selection_attrs, direct_edges=direct_edges, D1D2_twopoint_weights=twopoint_weights, D1R2_twopoint_weights=twopoint_weights)

    with MemoryMonitor() as mem:
        direct_D1D2 = DirectCorr(direct_edges, positions1=data1[:3], weights1=data1[3:], weights2=data2[3:], position_type='xyz',
                                ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)
        direct_D1R2 = DirectCorr(direct_edges, positions1=data1[:3], positions2=randoms1[:3], weights1=data1[3:], weights2=randoms1[3:], position_type='xyz',
                                ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)
        direct_R1D2 = DirectCorr(direct_edges, positions1=randoms1[:3], positions2=data1[:3], weights1=randoms1[3:], weights2=data2[3:], position_type='xyz',
                                ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)

    assert np.allclose(power.poles.corr_direct_nonorm, get_alpha(power.attrs, 'DD') * direct_D1D2.corr_nonorm - get_alpha(power.attrs, 'DR') * direct_D1R2.corr_nonorm - get_alpha(power.attrs, 'RD') * direct_R1D2.corr_nonorm)
    assert direct_D1D2.same_shotnoise
    assert power.poles.shotnoise != 0.

    power2 = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                             nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=4, edges=kedges, position_type='xyz',
                             direct_selection_attrs=selection_attrs, direct_edges=direct_edges, D1D2_twopoint_weights=twopoint_weights, D1R2_twopoint_weights=twopoint_weights)

    poles = power.poles.concatenate_proj(power.poles, power2.poles)
    assert poles.ells == (0, 2, 4)
    assert np.allclose(poles.power[:2], power.poles.power)
    assert np.allclose(poles.power[2:], power2.poles.power)

    direct_edges = {'step': 0.2}
    power2 = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                             nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=4, edges=kedges, position_type='xyz',
                             direct_selection_attrs=selection_attrs, direct_edges=direct_edges, D1D2_twopoint_weights=twopoint_weights, D1R2_twopoint_weights=twopoint_weights)


def test_mem():
    from test_fft_power import MemoryMonitor
    nmesh = 100
    kedges = np.linspace(0., 0.1, 6)
    ells = (0, 2)
    resampler = 'tsc'
    interlacing = 2
    data1, data2 = generate_catalogs(size=50000, boxsize=(2000.,) * 3, n_individual_weights=1, n_bitwise_weights=0, seed=42)
    randoms1, randoms2 = generate_catalogs(size=100000, boxsize=(2000.,) * 3, n_individual_weights=1, n_bitwise_weights=0, seed=84)
    selection_attrs = {'rp': (0., 5.)}
    direct_edges = {'step': 0.1, 'max': 10000}

    with MemoryMonitor() as mem:
        for i in range(5):
            power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                                    nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                                    direct_selection_attrs=selection_attrs, direct_edges=direct_edges)
            mem()


if __name__ == '__main__':

    setup_logging()

    test_direct_corr()
    test_catalog_power()
    #test_mem()
