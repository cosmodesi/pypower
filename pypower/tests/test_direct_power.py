import os
import tempfile

import numpy as np
from scipy import special

from pypower import DirectPower, utils, setup_logging


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


def get_weight(weights1, weights2, n_bitwise_weights=0, nrealizations=None, noffset=1, default_value=0.):
    if nrealizations is None:
        weight = 1
    else:
        denom = noffset + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights]))
        weight = default_value if denom == 0 else nrealizations/denom
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
            xyz1, xyz2 = xyzw2[:3], xyzw1[:3]
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
                    weight = get_weight(weights1, weights2, **kwargs)
                    for ill, ell in enumerate(ells):
                        toret[ill] += weight * (2*ell + 1) * (-1j)**ell * special.spherical_jn(ell, modes * dist) * legendre[ill](mu)
    #print('npairs', npairs)
    return np.asarray(toret)


def ref_s(modes, limits, data1, data2=None, boxsize=None, los='midpoint', ells=(0, 2, 4), **kwargs):
    if los not in ['firstpoint', 'endpoint', 'midpoint']:
        los = [1 if i == 'xyz'.index(los) else 0 for i in range(3)]
    toret = [np.zeros_like(modes, dtype='c16') for ell in ells]
    legendre = [special.legendre(ell) for ell in ells]
    if data2 is None: data2 = data1
    for xyzw1 in zip(*data1):
        for xyzw2 in zip(*data2):
            xyz1, xyz2 = xyzw2[:3], xyzw1[:3]
            dxyz = diff(xyzw2[:3], xyzw1[:3])
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
                weight = get_weight(weights1, weights2, **kwargs)
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
    ref = [get_weight([w[ii] for w in weights[0]], [w[ii] for w in weights[1]], n_bitwise_weights=len(weights[0]),
                      nrealizations=nrealizations, noffset=1, default_value=1.) for ii in range(size)]
    assert np.allclose(wpip, ref)


def test_direct_power():
    ref_funcs = {'theta':ref_theta, 's':ref_s}
    list_engine = ['kdtree']
    modes = np.linspace(0.01, 0.1, 11)
    size = 100
    boxsize = (1000,)*3
    list_options = []
    list_options.append({})
    list_options.append({'autocorr':True})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'iip':1})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'bitwise_type': 'i4', 'iip':1})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'bitwise_type': 'i4', 'iip':1, 'limit_type':'s'})
    list_options.append({'n_individual_weights':2, 'n_bitwise_weights':2, 'iip':2, 'position_type':'rdd', 'weight_attrs':{'nrealizations':42, 'noffset':3}})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':2, 'iip':2, 'weight_attrs':{'noffset':0, 'default_value':0.8}})

    for engine in list_engine:
        for options in list_options:
            options = options.copy()
            n_individual_weights = options.pop('n_individual_weights',0)
            n_bitwise_weights = options.pop('n_bitwise_weights',0)
            data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=42)
            #n_individual_weights = 0
            #data1, data2 = data1[:3], data2[:3]
            limit_type = options.pop('limit_type', 'theta')
            if limit_type == 'theta':
                limits = (0., 1.)
            else:
                limits = (0., 50.)
            autocorr = options.pop('autocorr', False)
            options.setdefault('boxsize', None)
            options['los'] = 'x' if options['boxsize'] is not None else 'firstpoint'
            bin_type = options.pop('bin_type', 'auto')
            mpicomm = options.pop('mpicomm', None)
            bitwise_type = options.pop('bitwise_type', None)
            iip = options.pop('iip', False)
            position_type = options.pop('position_type', 'xyz')
            dtype = options.pop('dtype', None)
            refoptions = options.copy()
            weight_attrs = refoptions.pop('weight_attrs', {}).copy()

            def setdefaultnone(di, key, value):
                if di.get(key, None) is None:
                    di[key] = value

            setdefaultnone(weight_attrs, 'nrealizations', n_bitwise_weights * 64 + 1)
            setdefaultnone(weight_attrs, 'noffset', 1)
            set_default_value = 'default_value' in weight_attrs
            setdefaultnone(weight_attrs, 'default_value', 0)
            refdata1, refdata2 = data1.copy(), data2.copy()
            if set_default_value:
                for w in data1[3:3+n_bitwise_weights] + data2[3:3+n_bitwise_weights]: w[:] = 0 # set to zero to make sure default_value is used

            def wiip(weights):
                denom = weight_attrs['noffset'] + utils.popcount(*weights)
                mask = denom == 0
                denom[mask] = 1.
                toret = weight_attrs['nrealizations']/denom
                toret[mask] = weight_attrs['default_value']
                return toret

            def dataiip(data):
                return data[:3] + [wiip(data[3:3+n_bitwise_weights])] + data[3+n_bitwise_weights:]

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
            tol = {'atol':1e-8, 'rtol':1e-3} if itemsize <= 4 else {'atol':1e-8, 'rtol':1e-6}

            if dtype is not None:
                for ii in range(len(data1)):
                    if np.issubdtype(data1[ii].dtype, np.floating):
                        refdata1[ii] = np.asarray(data1[ii], dtype=dtype)
                        refdata2[ii] = np.asarray(data2[ii], dtype=dtype)

            ref = ref_funcs[limit_type](modes, limits, refdata1, data2=None if autocorr else refdata2, n_bitwise_weights=n_bitwise_weights, **refoptions, **weight_attrs)

            if bitwise_type is not None and n_bitwise_weights > 0:

                def update_bit_type(data):
                    return data[:3] + utils.reformat_bitarrays(*data[3:3+n_bitwise_weights], dtype=bitwise_type) + data[3+n_bitwise_weights:]

                data1 = update_bit_type(data1)
                data2 = update_bit_type(data2)

            npos = 3
            if position_type != 'xyz':

                def update_pos_type(data):
                    rdd = list(utils.cartesian_to_sky(data[:3]))
                    return rdd + data[3:]

                data1 = update_pos_type(data1)
                data2 = update_pos_type(data2)

            def run(**kwargs):
                return DirectPower(modes, positions1=data1[:npos], positions2=None if autocorr else data2[:npos],
                                   weights1=data1[npos:], weights2=None if autocorr else data2[npos:], position_type=position_type, bin_type=bin_type,
                                   limits=limits, limit_type=limit_type, engine=engine, **kwargs, **options)

            test = run()
            assert np.allclose(test.power_nonorm, ref, **tol)

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = os.path.join(tmp_dir, 'tmp.npy')
                test.save(fn)
                test2 = DirectPower.load(fn)
                assert np.allclose(test.power_nonorm, ref, **tol)


if __name__ == '__main__':

    setup_logging()
    test_bitwise_weight()
    test_direct_power()
