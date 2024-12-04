import os
import time
import tempfile

import numpy as np
from scipy import special
from mpi4py import MPI

from pypower import DirectPower, CatalogFFTPower, mpi, utils, setup_logging


class MemoryMonitor(object):
    """
    Class that monitors memory usage and clock, useful to check for memory leaks.

    >>> with MemoryMonitor() as mem:
            '''do something'''
            mem()
            '''do something else'''
    """
    def __init__(self, pid=None):
        """
        Initalize :class:`MemoryMonitor` and register current memory usage.

        Parameters
        ----------
        pid : int, default=None
            Process identifier. If ``None``, use the identifier of the current process.
        """
        import psutil
        self.proc = psutil.Process(os.getpid() if pid is None else pid)
        self.mem = self.proc.memory_info().rss / 1e6
        self.time = time.time()
        msg = 'using {:.3f} [Mb]'.format(self.mem)
        print(msg, flush=True)

    def __enter__(self):
        """Enter context."""
        return self

    def __call__(self, log=None):
        """Update memory usage."""
        mem = self.proc.memory_info().rss / 1e6
        t = time.time()
        msg = 'using {:.3f} [Mb] (increase of {:.3f} [Mb]) after {:.3f} [s]'.format(mem, mem - self.mem, t - self.time)
        if log:
            msg = '[{}] {}'.format(log, msg)
        print(msg, flush=True)
        self.mem = mem
        self.time = t

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        self()


@np.vectorize
def spherical_bessel(x, ell=0):

    ABS, SIN, COS = np.abs, np.sin, np.cos

    absx = ABS(x)
    threshold_even = 4e-2
    threshold_odd = 2e-2
    if (ell == 0):
        if (absx < threshold_even):
            x2 = x * x
            return 1 - x2 / 6 + (x2 * x2) / 120
        return SIN(x) / x
    if (ell == 2):
        x2 = x * x
        if (absx < threshold_even): return x2 / 15 - (x2 * x2) / 210
        return (3 / x2 - 1) * SIN(x) / x - 3 * COS(x) / x2
    if (ell == 4):
        x2 = x * x
        x4 = x2 * x2
        if (absx < threshold_even): return x4 / 945
        return 5 * (2 * x2 - 21) * COS(x) / x4 + (x4 - 45 * x2 + 105) * SIN(x) / (x * x4)
    if (ell == 1):
        if (absx < threshold_odd): return x / 3 - x * x * x / 30
        return SIN(x) / (x * x) - COS(x) / x
    if (ell == 3):
        if (absx < threshold_odd): return x * x * x / 105
        x2 = x * x
        return (x2 - 15) * COS(x) / (x * x2) - 3 * (2 * x2 - 5) * SIN(x) / (x2 * x2)


def legendre(x, ell):

    if (ell == 0):
        return 1.
    if (ell == 2):
        x2 = x * x
        return (3 * x2 - 1) / 2
    if (ell == 4):
        x2 = x * x
        return (35 * x2 * x2 - 30 * x2 + 3) / 8
    if (ell == 1):
        return x
    if (ell == 3):
        return (5 * x * x * x - 3 * x) / 2


def test_legendre_bessel():
    mu = np.linspace(-1., 1., 1000)
    x = np.geomspace(1e-9, 100, 1000)
    for ell in range(5):
        assert np.allclose(legendre(mu, ell), special.legendre(ell)(mu), atol=0, rtol=1e-9)
        assert np.allclose(spherical_bessel(x, ell), special.spherical_jn(ell, x, derivative=False), atol=1e-7, rtol=1e-3)


def generate_catalogs(size=100, boxsize=(1000,) * 3, offset=(1000., 0., 0.), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size) * b for o, b in zip(offset, boxsize)]
        weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64 * n_bitwise_weights)], dtype=np.uint64)
        # weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(33)], dtype=np.uint64)
        # weights = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions + weights)
    return toret


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


def wpip_single(weights1, weights2, nrealizations=None, noffset=1, default_value=0., correction=None):
    denom = noffset + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1, weights2))
    if denom == 0:
        weight = default_value
    else:
        weight = nrealizations / denom
        if correction is not None:
            c = tuple(sum(bin(w).count('1') for w in weights) for weights in [weights1, weights2])
            weight /= correction[c]
    return weight


def wiip_single(weights, nrealizations=None, noffset=1, default_value=0.):
    denom = noffset + utils.popcount(*weights)
    return default_value if denom == 0 else nrealizations / denom


def get_weight(xyz1, xyz2, weights1, weights2, n_bitwise_weights=0, twopoint_weights=None, nrealizations=None, noffset=1, default_value=0., correction=None, weight_type='auto'):
    weight = 1
    if nrealizations is not None:
        weight *= wpip_single(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value, correction=correction)
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


def ref_theta(modes, data1, data2=None, boxsize=None, los='midpoint', ells=(0, 2, 4), autocorr=False, selection_attrs=None, **kwargs):
    if data2 is None: data2 = data1
    poles = [np.zeros_like(modes, dtype='c16') for ell in ells]
    legendre = [special.legendre(ell) for ell in ells]
    selection_attrs = dict(selection_attrs or {})
    theta_limits = selection_attrs.get('theta', None)
    rp_limits = selection_attrs.get('rp', None)
    npairs = 0
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            if theta_limits is not None:
                theta = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1)))  # min to avoid rounding errors
                if theta < theta_limits[0] or theta >= theta_limits[1]: continue
            dxyz = diff(xyz1, xyz2)
            dist = norm(dxyz)
            npairs += 1
            if dist > 0:
                if los == 'midpoint': mu = dotproduct_normalized(dxyz, midpoint(xyz1, xyz2))
                elif los == 'endpoint': mu = dotproduct_normalized(dxyz, xyz2)
                elif los == 'firstpoint': mu = dotproduct_normalized(dxyz, xyz1)
            else:
                mu = 0.
            if rp_limits is not None:
                rp2 = (1. - mu**2) * dist**2
                if rp2 < rp_limits[0]**2 or rp2 >= rp_limits[1]**2: continue
            weights1, weights2 = xyzw1[3:], xyzw2[3:]
            weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
            for ill, ell in enumerate(ells):
                poles[ill] += (-1j)**ell * weight * (2 * ell + 1) * legendre[ill](mu) * special.spherical_jn(ell, modes * dist)
    return np.asarray(poles)


def test_bitwise_weight():
    from pypower.direct_power import get_inverse_probability_weight
    size = 10
    n_bitwise_weights = 2
    rng = np.random.RandomState(seed=42)
    weights = [utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64 * n_bitwise_weights)], dtype=np.uint64) for i in range(2)]
    nrealizations = 1 + len(weights[0]) * 64
    wpip = get_inverse_probability_weight(*weights, noffset=1, nrealizations=None, default_value=1.)
    ref = [wpip_single([w[ii] for w in weights[0]], [w[ii] for w in weights[1]],
                       nrealizations=nrealizations, noffset=1, default_value=1.) for ii in range(size)]
    assert np.allclose(wpip, ref)


def test_direct_power():
    list_engine = ['kdtree', 'corrfunc']
    modes = np.linspace(0.01, 0.1, 11)
    size = 100
    boxsize = (100,) * 3
    from pypower.direct_power import KDTreeDirectPowerEngine
    _slab_npairs_max = KDTreeDirectPowerEngine._slab_npairs_max
    _slab_nobjs_max = KDTreeDirectPowerEngine._slab_nobjs_max
    list_options = []

    for autocorr in [False, True]:
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 2, 'weight_attrs': {'normalization': 'counter'}})
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
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 2, 'weight_attrs': {'normalization': 'counter'}})
        list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 0, 'weight_attrs': {'normalization': 'counter'}})
        # los
        for los in ['midpoint', 'firstpoint', 'endpoint']:
            list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': los})
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

            KDTreeDirectPowerEngine._slab_npairs_max = options.pop('slab_npairs_max', _slab_npairs_max)
            KDTreeDirectPowerEngine._slab_nobjs_max = options.pop('slab_nobjs_max', _slab_nobjs_max)

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

            if n_bitwise_weights == 0:
                weight_attrs['nrealizations'] = None
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

            if dtype is not None:
                for ii in range(len(data1_ref)):
                    if np.issubdtype(data1_ref[ii].dtype, np.floating):
                        data1_ref[ii] = np.asarray(data1_ref[ii], dtype=dtype)
                        data2_ref[ii] = np.asarray(data2_ref[ii], dtype=dtype)

            twopoint_weights = ref_options.pop('twopoint_weights', None)
            if twopoint_weights is not None:
                twopoint_weights = TwoPointWeight(np.cos(np.radians(twopoint_weights.sep[::-1], dtype=dtype)), np.asarray(twopoint_weights.weight[::-1], dtype=dtype))

            if n_bitwise_weights and weight_attrs.get('normalization', None) == 'counter':
                nalways = weight_attrs.get('nalways', 0)
                noffset = weight_attrs.get('noffset', 1)
                nrealizations = weight_attrs['nrealizations']
                noffset = weight_attrs['noffset']
                joint = utils.joint_occurences(nrealizations, noffset=noffset + nalways, default_value=weight_attrs['default_value'])
                correction = np.ones((n_bitwise_weights * 64 + 1,) * 2, dtype='f8')
                for c1 in range(nalways, min(nrealizations - noffset, n_bitwise_weights * 64) + 1):
                    for c2 in range(nalways, min(nrealizations - noffset, n_bitwise_weights * 64) + 1):
                        correction[c1][c2] = joint[c1 - nalways][c2 - nalways] if c2 <= c1 else joint[c2 - nalways][c1 - nalways]
                        correction[c1][c2] /= (nrealizations / (noffset + c1) * nrealizations / (noffset + c2))
                weight_attrs['correction'] = correction
            weight_attrs.pop('normalization', None)

            ref = ref_theta(modes, data1_ref, data2=data2_ref if not autocorr else None, autocorr=autocorr, n_bitwise_weights=n_bitwise_weights, twopoint_weights=twopoint_weights, selection_attrs=selection_attrs, **ref_options, **weight_attrs)

            itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
            tol = {'atol': 1e-8, 'rtol': 1e-2} if itemsize <= 4 else {'atol': 1e-8, 'rtol': 1e-5}

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

                return DirectPower(modes, positions1=None if pass_none else positions1, positions2=None if pass_none or autocorr else positions2,
                                   weights1=None if pass_none else weights1, weights2=None if pass_none or autocorr else weights2, position_type=position_type,
                                   selection_attrs=selection_attrs, engine=engine, nthreads=nthreads, **kwargs, **options)

            data1 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data1]
            data2 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data2]
            test = run(mpiroot=None)
            assert np.allclose(test.power_nonorm, ref, **tol)
            test_zero = run(mpiroot=None, pass_none=False, pass_zero=True)
            assert np.allclose(test_zero.power_nonorm, 0.)

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = test.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
                test.save(fn)
                test.mpicomm.Barrier()
                test2 = DirectPower.load(fn)
                assert np.allclose(test.power_nonorm, ref, **tol)
                fn = os.path.join(tmp_dir, 'tmp.npy')
                test2.save(fn)

            mpiroot = 0
            data1_ref, data2_ref = data1, data2
            data1 = [mpi.gather(d, mpiroot=mpiroot, mpicomm=mpicomm) for d in data1]
            data2 = [mpi.gather(d, mpiroot=mpiroot, mpicomm=mpicomm) for d in data2]
            test_mpi = run(mpiroot=mpiroot)
            assert np.allclose(test_mpi.power_nonorm, test.power_nonorm, **tol)
            test_mpi = run(mpiroot=mpiroot, pass_none=mpicomm.rank > 0)
            assert np.allclose(test_mpi.power_nonorm, test.power_nonorm, **tol)
            if test.mpicomm.rank == mpiroot:
                test_mpi = run(mpiroot=mpiroot, mpicomm=MPI.COMM_SELF)
            assert np.allclose(test_mpi.power_nonorm, test.power_nonorm, **tol)
            data1 = [d if mpicomm.rank == mpiroot else dref[:0] for d, dref in zip(data1, data1_ref)]
            data2 = [d if mpicomm.rank == mpiroot else dref[:0] for d, dref in zip(data2, data2_ref)]
            test_mpi = run(mpiroot=None)
            assert np.allclose(test_mpi.power_nonorm, ref, **tol)
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

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs=selection_attrs)
    direct = DirectPower(power.poles.k, positions1=data1[:3], weights1=data1[3:], position_type='xyz',
                         ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual')

    assert np.allclose(power.poles.power_direct_nonorm, direct.power_nonorm)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            data_positions2=data2[:3], data_weights2=data2[3:], randoms_positions2=randoms2[:3], randoms_weights2=randoms2[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs=selection_attrs)
    direct = DirectPower(power.poles.k, positions1=data1[:3], positions2=data2[:3], weights1=data1[3:], weights2=data2[3:], position_type='xyz',
                         ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual')
    assert np.allclose(power.poles.power_direct_nonorm, direct.power_nonorm)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[-1:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            data_positions2=data2[:3], data_weights2=data2[-1:], randoms_positions2=randoms2[:3], randoms_weights2=randoms2[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs=selection_attrs)
    direct = DirectPower(power.poles.k, positions1=data1[:3], positions2=data2[:3], weights1=data1[-1:], weights2=data2[-1:], position_type='xyz',
                         ells=ells, selection_attrs=selection_attrs, weight_type='auto').power_nonorm
    direct -= get_alpha(power.attrs, 'DR') * DirectPower(power.poles.k, positions1=data1[:3], positions2=randoms2[:3], weights1=data1[-1:], weights2=randoms2[3:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').power_nonorm
    direct -= get_alpha(power.attrs, 'RD') * DirectPower(power.poles.k, positions1=randoms1[:3], positions2=data2[:3], weights1=randoms1[3:], weights2=data2[-1:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').power_nonorm
    direct += get_alpha(power.attrs, 'RR') * DirectPower(power.poles.k, positions1=randoms1[:3], positions2=randoms2[:3], weights1=randoms1[3:], weights2=randoms2[3:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').power_nonorm
    assert np.allclose(power.poles.power_direct_nonorm, -direct)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[-1:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            data_positions2=data2[:3], data_weights2=data2[-1:], randoms_positions2=randoms2[:3], randoms_weights2=randoms2[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs={**selection_attrs, 'counts': ['D1D2', 'D1R2']})
    direct = DirectPower(power.poles.k, positions1=data1[:3], positions2=data2[:3], weights1=data1[-1:], weights2=data2[-1:], position_type='xyz',
                         ells=ells, selection_attrs=selection_attrs, weight_type='auto').power_nonorm
    direct -= get_alpha(power.attrs, 'DR') * DirectPower(power.poles.k, positions1=data1[:3], positions2=randoms2[:3], weights1=data1[-1:], weights2=randoms2[3:], position_type='xyz',
                        ells=ells, selection_attrs=selection_attrs, weight_type='auto').power_nonorm
    assert np.allclose(power.poles.power_direct_nonorm, -direct)

    weight_attrs = {'normalization': 'counter'}
    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz',
                            direct_selection_attrs=selection_attrs, D1D2_twopoint_weights=twopoint_weights, D1R2_twopoint_weights=twopoint_weights, weight_attrs=weight_attrs)

    direct_D1D2 = DirectPower(power.poles.k, positions1=data1[:3], weights1=data1[3:], position_type='xyz',
                              ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights) #, weight_attrs=weight_attrs)
    direct_D1R2 = DirectPower(power.poles.k, positions1=data1[:3], positions2=randoms1[:3], weights1=data1[3:], weights2=randoms1[3:], position_type='xyz',
                              ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights, weight_attrs=weight_attrs)
    direct_R1D2 = DirectPower(power.poles.k, positions1=randoms1[:3], positions2=data1[:3], weights1=randoms1[3:], weights2=data1[3:], position_type='xyz',
                              ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights, weight_attrs=weight_attrs)
    assert np.allclose(power.poles.power_direct_nonorm, get_alpha(power.attrs, 'DD') * direct_D1D2.power_nonorm - get_alpha(power.attrs, 'DR') * direct_D1R2.power_nonorm - get_alpha(power.attrs, 'RD') * direct_R1D2.power_nonorm)

    power = power.poles
    power2 = power + power
    assert np.allclose(power2.power, power.power, equal_nan=True)
    assert np.allclose(power2.wnorm, 2. * power.wnorm, equal_nan=True)

    power = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], data_weights2=data2[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                            nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, edges=kedges, position_type='xyz', direct_attrs={'nthreads': 4},
                            direct_selection_attrs=selection_attrs, D1D2_twopoint_weights=twopoint_weights, D1R2_twopoint_weights=twopoint_weights)

    with MemoryMonitor() as mem:
        direct_D1D2 = DirectPower(power.poles.k, positions1=data1[:3], weights1=data1[3:], weights2=data2[3:], position_type='xyz',
                                  ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)
        direct_D1R2 = DirectPower(power.poles.k, positions1=data1[:3], positions2=randoms1[:3], weights1=data1[3:], weights2=randoms1[3:], position_type='xyz',
                                  ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)
        direct_R1D2 = DirectPower(power.poles.k, positions1=randoms1[:3], positions2=data1[:3], weights1=randoms1[3:], weights2=data2[3:], position_type='xyz',
                                  ells=ells, selection_attrs=selection_attrs, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights)

    assert np.allclose(power.poles.power_direct_nonorm, get_alpha(power.attrs, 'DD') * direct_D1D2.power_nonorm - get_alpha(power.attrs, 'DR') * direct_D1R2.power_nonorm - get_alpha(power.attrs, 'RD') * direct_R1D2.power_nonorm)
    assert direct_D1D2.same_shotnoise
    assert power.poles.shotnoise != 0.

    power2 = CatalogFFTPower(data_positions1=data1[:3], data_weights1=data1[3:], randoms_positions1=randoms1[:3], randoms_weights1=randoms1[3:],
                             nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=4, edges=kedges, position_type='xyz',
                             direct_selection_attrs=selection_attrs, D1D2_twopoint_weights=twopoint_weights, D1R2_twopoint_weights=twopoint_weights)

    poles = power.poles.concatenate_proj(power.poles, power2.poles)
    assert poles.ells == (0, 2, 4)
    assert np.allclose(poles.power[:2], power.poles.power)
    assert np.allclose(poles.power[2:], power2.poles.power)


def format_pip_reference():
    ref_dir = 'reference_pip'
    for mode in ['gal_w_uncorrelated', 'gal_w_correlated'][1:]:
        fn = os.path.join(ref_dir, 'Arnaud_test_{}.dat'.format(mode))
        nbits = 62
        dtype = [(axis, np.float64) for axis in 'xyz']
        dtype += [('indweight', np.float64), ('bitweight0', np.int32), ('bitweight1', np.int32)]
        dtype += [('bit{:d}'.format(ibit), np.bool_) for ibit in range(nbits)]
        tmp = []
        with open(fn, 'r') as file:
            for line in file:
                if line.strip().startswith('#'): continue
                line = [el.strip() for el in line.split(' ')]
                line = [el for el in line if el]
                if line[3] == 'F':
                    continue
                for ii in range(4, len(line)):
                    if ii == 4:
                        line[ii] = float(line[ii])
                    elif ii in [5, 6]:
                        line[ii] = int(line[ii])
                    else:
                        line[ii] = line[ii] == 'T'
                tmp.append(tuple(line[:3] + line[4:]))
        tmp = np.array(tmp, dtype=dtype)
        np.save(os.path.join(ref_dir, '{}.npy'.format(mode)), tmp)
    #fn = os.path.join(ref_dir, 'Arnaud_test_ran.dat')
    #tmp = np.loadtxt(fn, dtype=[(axis, np.float64) for axis in 'xyz'])
    #np.save(os.path.join(ref_dir, 'ran.npy'), tmp)


def test_pip_counts_correction():

    from pypower.direct_power import get_inverse_probability_weight
    from pypower import DirectPower
    ref_dir = 'reference_pip'
    ref_fn = os.path.join(ref_dir, 'Arnaud_ng512_tm0.5_P_pipco_mp.dat')
    mode = 'correlated'
    no_indweight = False
    fn = os.path.join(ref_dir, 'gal_w_{}.npy'.format(mode))
    tmp = np.load(fn)
    nbits = len([name for name in tmp.dtype.names if name.startswith('bit') and 'weight' not in name])
    data_weights = [tmp['indweight']] + utils.pack_bitarrays(*[tmp['bit{:d}'.format(ibit)] for ibit in range(nbits)])
    data_weights_iip = [data_weights[0], get_inverse_probability_weight(data_weights[1:], noffset=1, nrealizations=1 + nbits)]
    if no_indweight:
        data_weights = data_weights[1:]
        data_weights_iip = data_weights_iip[1:]
    data_positions = [tmp[axis] for axis in 'xyz']
    #fn = os.path.join(ref_dir, 'ran.npy')
    #tmp = np.load(fn)
    #randoms_positions = [tmp[axis] for axis in 'xyz']
    kmid, kavg, p0, p2, p4, norm = np.loadtxt(ref_fn, unpack=True, usecols=[0, 1, 5, 6, 7, 9])

    ref = np.array([p0, p2, p4]) * norm
    los = 'midpoint'
    selection_attrs = {'theta': (0., 0.5)}
    ells = (0, 2, 4)

    result = DirectPower(kavg, positions1=data_positions, weights1=data_weights, ells=ells, los=los,
                         weight_type='inverse_bitwise_minus_individual', weight_attrs={'nrealizations': 1 + nbits, 'normalization': 'counter'},
                         selection_attrs=selection_attrs, position_type='xyz', nthreads=4)

    #result2 = DirectPower(kavg, positions1=data_positions, weights1=data_weights, ells=ells, los=los,
    #                      weight_type='inverse_bitwise_minus_individual', weight_attrs={'nrealizations': 1 + nbits},
    #                      selection_attrs=selection_attrs, position_type='xyz', nthreads=4)

    tol = {'rtol': 2e-1, 'atol': 0.}
    print(result.power_nonorm / ref)
    #print(result2.power_nonorm / result.power_nonorm)
    assert np.allclose(result.power_nonorm, ref, **tol)

    if True:
        from matplotlib import pyplot as plt

        ax = plt.gca()
        ax.plot([], [], color='k', linestyle='-', label='Arnaud')
        ax.plot([], [], color='k', linestyle='--', label='Davide')
        #ax.plot([], [], color='k', linestyle=':', label='Arnaud, no correction')
        for ill, ell in enumerate(result.ells):
            color = 'C{:d}'.format(ill)
            ax.plot(kavg, kavg * result.power_nonorm[ill].real, color=color, linestyle='-', label='$\ell = {:d}$'.format(ell))
            ax.plot(kavg, kavg * ref[ill], color=color, linestyle='--')
            #ax.plot(kavg, kavg * result2.power_nonorm[ill].real, color=color, linestyle=':')
        ax.legend()
        ax.set_xlabel('$k$ [$h/\mathrm{Mpc}$]')
        ax.set_ylabel('$kP(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        utils.savefig('_tests/tmp.png')
        plt.show()


if __name__ == '__main__':

    setup_logging()

    #test_legendre_bessel()
    #test_bitwise_weight()
    test_direct_power()
    test_catalog_power()
    #format_pip_reference()
    test_pip_counts_correction()
