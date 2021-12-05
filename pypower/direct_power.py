r"""
Implementation of direct estimation of result spectrum multipoles, i.e. summing over particle pairs.
This should be mostly used to sum over pairs at small separations, otherwise the calculation will be prohibitive.
"""

import time

import numpy as np
from scipy import special

from .utils import BaseClass
from . import mpi, utils


def _make_array(value, shape, dtype='f8'):
    # Return numpy array filled with value
    toret = np.empty(shape, dtype=dtype)
    toret[...] = value
    return toret


def get_default_nrealizations(weights):
    """Return default number of realizations given input bitwise weights = the number of bits in input weights plus one."""
    return 1 + 8 * sum(weight.dtype.itemsize for weight in weights)


def _vlogical_and(*arrays):
    # & between any number of arrays
    toret = arrays[0].copy()
    for array in arrays[1:]: toret &= array
    return toret


def get_inverse_probability_weight(*weights, noffset=1, nrealizations=None, default_value=0.):
    r"""
    Return inverse probability weight given input bitwise weights.
    Inverse probability weight is computed as::math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2} \& ...))`.
    If denominator is 0, weight is set to default_value.

    Parameters
    ----------
    weights : int arrays
        Bitwise weights.

    noffset : int, default=1
        The offset to be added to the bitwise counts in the denominator (defaults to 1).

    nrealizations : int, default=None
        Number of realizations (defaults to the number of bits in input weights plus one).

    default_value : float, default=0.
        Default weight value, if the denominator is zero (defaults to 0).
    """
    if nrealizations is None:
        nrealizations = get_default_nrealizations(weights[0])
    #denom = noffset + sum(utils.popcount(w1 & w2) for w1, w2 in zip(*weights))
    denom = noffset + sum(utils.popcount(_vlogical_and(*weight)) for weight in zip(*weights))
    mask = denom == 0
    denom[mask] = 1
    toret = nrealizations/denom
    toret[mask] = default_value
    return toret


def _format_positions(positions, position_type='xyz', dtype=None, mpicomm=None, mpiroot=None):
    # Format input array of positions
    # position_type in ["xyz", "rdd", "pos"]

    def __format_positions(positions, position_type=position_type, dtype=dtype):
        if position_type == 'pos': # array of shape (N, 3)
            positions = np.asarray(positions, dtype=dtype)
            if positions.shape[-1] != 3:
                return None, 'For position type = {}, please provide a (N, 3) array for positions'.format(position_type)
            return positions, None
        # Array of shape (3, N)
        positions = list(positions)
        for ip, p in enumerate(positions):
            # cast to the input dtype if exists (may be set by previous weights)
            positions[ip] = np.asarray(p, dtype=dtype)
        size = len(positions[0])
        dtype = positions[0].dtype
        if not np.issubdtype(dtype, np.floating):
            return None, 'Input position arrays should be of floating type, not {}'.format(dtype)
        for p in positions[1:]:
            if len(p) != size:
                return None, 'All position arrays should be of the same size'
            if p.dtype != dtype:
                return None, 'All position arrays should be of the same type, you can e.g. provide dtype'
        if len(positions) != 3:
            return None, 'For position type = {}, please provide a list of 3 arrays for positions'.format(position_type)
        if position_type == 'rdd': # RA, Dec, distance
            positions = utils.sky_to_cartesian(positions, degree=True)
        elif position_type != 'xyz':
            return None, 'Position type should be one of ["xyz", "rdd"]'
        return np.asarray(positions).T, None

    error = None
    if positions is not None and not all(position is None for position in positions):
        positions, error = __format_positions(positions) # return error separately to raise on all processes
    errors = [err for err in mpicomm.allgather(error) if err is not None]
    if errors:
        raise ValueError(errors[0])
    if mpiroot is not None and mpicomm.bcast(positions is not None, root=mpiroot):
        positions = mpi.scatter_array(positions, root=mpiroot, mpicomm=mpicomm)
    return positions


def _format_weights(weights, weight_type='auto', size=None, dtype=None, mpicomm=None, mpiroot=None):
    # Format input weights, as a list of n_bitwise_weights uint8 arrays, and optionally a float array for individual weights.
    # Return formated list of weights, and n_bitwise_weights.

    def __format_weights(weights, weight_type=weight_type, dtype=dtype):
        if weights is None or all(weight is None for weight in weights):
            return [], 0
        if np.ndim(weights[0]) == 0:
            weights = [weights]
        individual_weights = []
        bitwise_weights = []
        for w in weights:
            if np.issubdtype(w.dtype, np.integer):
                if weight_type == 'product_individual': # enforce float individual weight
                    individual_weights.append(w)
                else: # certainly bitwise weight
                    bitwise_weights.append(w)
            else:
                individual_weights.append(w)
        # any integer array bit size will be a multiple of 8
        bitwise_weights = utils.reformat_bitarrays(*bitwise_weights, dtype=np.uint8)
        n_bitwise_weights = len(bitwise_weights)
        weights = bitwise_weights
        if individual_weights:
            weights += [np.prod(individual_weights, axis=0, dtype=dtype)]
        return weights, n_bitwise_weights

    weights, n_bitwise_weights = __format_weights(weights)
    if mpiroot is not None:
        nw = mpicomm.bcast(len(weights), root=mpiroot)
        if mpicomm.rank != mpiroot: weights = [None]*nw
        weights = [mpi.scatter_array(weight, root=mpiroot, mpicomm=mpicomm) for weight in weights]
        n_bitwise_weights = mpicomm.bcast(n_bitwise_weights, root=mpiroot)
    if size is not None:
        if not all(len(weight) == size for weight in weights):
            raise ValueError('All weight arrays should be of the same size as position arrays')
    return weights, n_bitwise_weights


def get_direct_power_engine(engine='kdtree'):
    """
    Return :class:`BaseDirectPowerEngine`-subclass corresponding
    to input engine name.

    Parameters
    ----------
    engine : string, default='kdtree'
        Name of direct power engine, one of ["kdtree"].

    Returns
    -------
    engine : type
        Direct power engine class.
    """
    if isinstance(engine, str):

        if engine.lower() == 'kdtree':
            return KDTreeDirectPowerEngine

        raise ValuerError('Unknown engine {}.'.format(engine))

    return engine


class MetaDirectPower(type(BaseClass)):

    """Metaclass to return correct direct power engine."""

    def __call__(cls, *args, engine='kdtree', **kwargs):
        return get_direct_power_engine(engine)(*args, **kwargs)


class DirectPower(metaclass=MetaDirectPower):
    """
    Entry point to direct power engines.

    Parameters
    ----------
    engine : string, default='kdtree'
        Name of direct power engine, one of ["kdtree"].

    args : list
        Arguments for direct power engine, see :class:`BaseDirectPowerEngine`.

    kwargs : dict
        Arguments for direct power engine, see :class:`BaseDirectPowerEngine`.

    Returns
    -------
    engine : BaseDirectPowerEngine
    """
    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        return get_direct_power_engine(state.pop('name')).from_state(state)


class BaseDirectPowerEngine(BaseClass):

    """Direct power spectrum measurement, summing over particle pairs."""

    def __init__(self, modes, positions1, positions2=None, weights1=None, weights2=None, ells=(0, 2, 4), limits=(0., 2./60.), limit_type='degree',
                 position_type='xyz', weight_type='auto', weight_attrs=None, los='endpoint', boxsize=None, mpiroot=None, mpicomm=mpi.COMM_WORLD, **kwargs):
        r"""
        Initialize :class:`BaseDirectPowerEngine`.

        Parameters
        ----------
        modes : array
            Wavenumbers at which to compute power spectrum.

        positions1 : list, array
            Positions in the first data catalog. Typically of shape (3, N) or (N, 3).

        positions2 : list, array, default=None
            Optionally, for cross-power spectrum, positions in the second catalog. See ``positions1``.

        weights1 : array, list, default=None
            Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".
            See ``weight_type``.

        weights2 : array, list, default=None
            Optionally, for cross-pair counts, weights in the second catalog. See ``weights1``.

        ells : list, tuple, default=(0, 2, 4)
            Multipole orders.

        limits : tuple, default=(0., 2./60.)
            Limits of particle pair separations.

        limit_type : string, default='degree'
            Type of ``limits``; i.e. are those angular limits ("degree", "radian"), or 3D limits ("s")?

        position_type : string, default='xyz'
            Type of input positions, one of:

                - "pos": Cartesian positions of shape (N, 3)
                - "xyz": Cartesian positions of shape (3, N)
                - "rdd": RA/Dec in degree, distance of shape (3, N)

        weight_type : string, default='auto'
            The type of weighting to apply to provided weights. One of:

                - ``None``: no weights are applied.
                - "product_individual": each pair is weighted by the product of weights :math:`w_{1} w_{2}`.
                - "inverse_bitwise": each pair is weighted by :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2}))`.
                   Multiple bitwise weights can be provided as a list.
                   Individual weights can additionally be provided as float arrays.
                   In case of cross-correlations with floating weights, bitwise weights are automatically turned to IIP weights,
                   i.e. :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1}))`.
                - "auto": automatically choose weighting based on input ``weights1`` and ``weights2``,
                   i.e. ``None`` when ``weights1`` and ``weights2`` are ``None``,
                   "inverse_bitwise" if one of input weights is integer, else "product_individual".

        weight_attrs : dict, default=None
            Dictionary of weighting scheme attributes. In case ``weight_type`` is "inverse_bitwise",
            one can provide "nrealizations", the total number of realizations (*including* current one;
            defaulting to the number of bits in input weights plus one);
            "noffset", the offset to be added to the bitwise counts in the denominator (defaulting to 1)
            and "default_value", the default value of pairwise weights if the denominator is zero (defaulting to 0).
            Inverse probability weight is then computed as: :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2}))`.
            For example, for the "zero-truncated" estimator (arXiv:1912.08803), one would use noffset = 0.

        los : string, array, default=None
            If ``los`` is 'firstpoint' (resp. 'endpoint', 'midpoint'), use local (varying) first-point (resp. end-point, mid-point) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        boxsize : array, float, default=None
            For periodic wrapping, the side-length(s) of the periodic cube.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scatted across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=MPI.COMM_WORLD
            The MPI communicator.
        """
        self.mpicomm = mpicomm
        self._set_modes(modes)
        self._set_ells(ells)
        self._set_los(los)
        self._set_limits(limits, limit_type=limit_type)
        self._set_boxsize(boxsize=boxsize)
        self._set_positions(positions1, positions2=positions2, position_type=position_type, mpiroot=mpiroot)
        self._set_weights(weights1, weights2=weights2, weight_type=weight_type, weight_attrs=weight_attrs, mpiroot=mpiroot)
        self.attrs = kwargs
        self.run()

    def _set_modes(self, modes):
        self.modes = np.asarray(modes)

    def _set_ells(self, ells):
        self.ells = tuple(ells)

    def _set_los(self, los):
        # Set :attr:`los`
        self.los_type = 'global'
        if los is None:
            self.los_type = 'endpoint'
            self.los = None
        elif isinstance(los, str):
            los = los.lower()
            allowed_los = ['midpoint', 'endpoint', 'firstpoint', 'x', 'y', 'z']
            if los not in allowed_los:
                raise ValueError('los should be one of {}'.format(allowed_los))
            if los in ['midpoint', 'endpoint', 'firstpoint']:
                self.los_type = los
                self.los = None
            else:
                self.los_type = 'global'
                los = 'xyz'.index(los)
        if self.los_type == 'global':
            if np.ndim(los) == 0:
                ilos = los
                los = np.zeros(3, dtype='f8')
                los[ilos] = 1.
            los = np.array(los, dtype='f8')
            self.los = los/utils.distance(los)

    def _set_limits(self, limits, limit_type='degree'):
        limit_type = limit_type.lower()
        allowed_limit_types = ['degree', 'radian', 'theta', 's']
        if limit_type not in allowed_limit_types:
            raise ValueError('Limit should be in {}.'.format(allowed_limit_types))
        if limit_type == 'radian':
            limits = np.rad2deg(limits)
        self.limit_type = limit_type
        if limit_type in ['radian', 'degree']:
            self.limit_type = 'theta'
        if self.limit_type == 'theta':
            limits = 2 * np.sin(0.5 * np.deg2rad(limits))
        self.limits = tuple(limits)

    @property
    def periodic(self):
        """Whether periodic wrapping is used (i.e. :attr:`boxsize` is not ``None``)."""
        return self.boxsize is not None

    def _set_boxsize(self, boxsize):
        self.boxsize = boxsize
        if self.periodic:
            self.boxsize = _make_array(boxsize, 3, dtype='f8')

    def _set_positions(self, positions1, positions2=None, position_type='xyz', mpiroot=None):
        self.positions1 = _format_positions(positions1, position_type=position_type, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.positions2 = _format_positions(positions2, position_type=position_type, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.autocorr = positions2 is None

    def _set_weights(self, weights1, weights2=None, weight_type='auto', weight_attrs=None, mpiroot=None):

        if weight_type is not None: weight_type = weight_type.lower()
        allowed_weight_types = [None, 'auto', 'product_individual', 'inverse_bitwise', 'inverse_bitwise_minus_individual']
        if weight_type not in allowed_weight_types:
            raise ValueError('weight_type should be one of {}'.format(allowed_weight_types))
        self.weight_type = weight_type

        weight_attrs = weight_attrs or {}
        self.weight_attrs = {}

        self.n_bitwise_weights = 0
        if self.weight_type is None:
            self.weights1 = self.weights2 = []

        else:

            noffset = weight_attrs.get('noffset', 1)
            default_value = weight_attrs.get('default_value', 0.)
            self.weight_attrs.update(noffset=noffset, default_value=default_value)

            self.weights1, n_bitwise_weights1 = _format_weights(weights1, weight_type=weight_type, size=len(self.positions1), mpicomm=self.mpicomm, mpiroot=mpiroot)

            def get_nrealizations(weights):
                nrealizations = weight_attrs.get('nrealizations', None)
                if nrealizations is None: nrealizations = get_default_nrealizations(weights)
                return nrealizations

            if self.autocorr:

                self.weights2 = self.weights1
                self.weight_attrs['nrealizations'] = get_nrealizations(self.weights1[:n_bitwise_weights1])
                self.n_bitwise_weights = n_bitwise_weights1

            else:
                self.weights2, n_bitwise_weights2 = _format_weights(weights2, weight_type=weight_type, size=len(self.positions2), mpicomm=self.mpicomm, mpiroot=mpiroot)

                if n_bitwise_weights2 == n_bitwise_weights1:

                    self.weight_attrs['nrealizations'] = get_nrealizations(self.weights1[:n_bitwise_weights1])
                    self.n_bitwise_weights = n_bitwise_weights1

                else:
                    if n_bitwise_weights2 == 0:
                        indweights = self.weights1[n_bitwise_weights1] if len(self.weights1) > n_bitwise_weights1 else 1.
                        self.weight_attrs['nrealizations'] = get_nrealizations(self.weights1[:n_bitwise_weights1])
                        self.weights1 = [self._get_inverse_probability_weight(self.weights1[:n_bitwise_weights1])*indweights]
                        self.n_bitwise_weights = 0
                        self.log_info('Setting IIP weights for first catalog.')
                    elif n_bitwise_weights1 == 0:
                        indweights = self.weights2[n_bitwise_weights2] if len(self.weights2) > n_bitwise_weights2 else 1.
                        self.weight_attrs['nrealizations'] = get_nrealizations(self.weights2[:n_bitwise_weights2])
                        self.weights2 = [self._get_inverse_probability_weight(self.weights2[:n_bitwise_weights2])*indweights]
                        self.n_bitwise_weights = 0
                        self.log_info('Setting IIP weights for second catalog.')
                    else:
                        raise ValueError('Incompatible length of bitwise weights: {:d} and {:d} bytes'.format(n_bitwise_weights1, n_bitwise_weights2))

    def _get_inverse_probability_weight(self, *weights):
        return get_inverse_probability_weight(*weights, noffset=self.weight_attrs['noffset'], nrealizations=self.weight_attrs['nrealizations'],
                                              default_value=self.weight_attrs['default_value'])

    @property
    def with_mpi(self):
        """Whether to use MPI."""
        return self.mpicomm.size > 1

    def _mpi_decompose(self):
        positions1, positions2 = self.positions1, self.positions2
        weights1, weights2 = self.weights1, self.weights2
        if self.limit_type == 'theta': # we decompose on the unit sphere: normalize positions, and put original positions in weights for decomposition
            positions1 = self.positions1/utils.distance(self.positions1.T)[:,None]
            weights1 = [self.positions1] + weights1
            if not self.autocorr:
                positions2 = self.positions2/utils.distance(self.positions2.T)[:,None]
                weights2 = [self.positions2] + weights2
        if self.with_mpi:
            (positions1, weights1), (positions2, weights2) = mpi.domain_decompose(self.mpicomm, self.limits[1], positions1, weights1=weights1,
                                                                                  positions2=positions2, weights2=weights2, boxsize=self.boxsize)
        elif self.autocorr:
            positions2, weights2 = positions1, weights1
        if self.limit_type == 'theta': # we remove original positions from the list of weights
            (limit_positions1, positions1, weights1) = (positions1, weights1[0], weights1[1:])
            (limit_positions2, positions2, weights2) = (positions2, weights2[0], weights2[1:])
        else:
            (limit_positions1, positions1, weights1) = (positions1, positions1, weights1)
            (limit_positions2, positions2, weights2) = (positions2, positions2, weights2)
        return (limit_positions1, positions1, weights1), (limit_positions2, positions2, weights2)

    def run(self):
        """Method that computes the power spectrum and set :attr:`power_nonorm`, to be implemented in your new engine."""
        raise NotImplementedError('Implement method "run" in your {}'.format(self.__class__.__name__))

    def __getstate__(self):
        state = {}
        for name in ['name', 'autocorr', 'modes', 'power_nonorm', 'limits', 'limit_type',
                     'boxsize', 'los', 'weight_attrs', 'attrs']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def save(self, filename):
        """Save direct power to ``filename``."""
        if not self.with_mpi or self.mpicomm.rank == 0:
            super(BaseDirectPowerEngine, self).save(filename)
        self.mpicomm.Barrier()


class KDTreeDirectPowerEngine(BaseDirectPowerEngine):

    """Direct power spectrum measurement, summing over particle pairs, identified with KDTree."""

    name = 'kdtree'

    def run(self):
        # FIXME: We may run out-of-memory when too many pairs...
        from scipy import spatial
        rank = self.mpicomm.rank
        start = time.time()
        ells = sorted(set(self.ells))
        (dlimit_positions1, dpositions1, dweights1), (dlimit_positions2, dpositions2, dweights2) = self._mpi_decompose()
        autocorr = dpositions2 is dpositions1
        dlimit_positions = dlimit_positions1
        # Very unfortunately, cKDTree.query_pairs does not handle cross-correlations...
        # But I feel this could be changed super easily here:
        # https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/spatial/ckdtree/src/query_pairs.cxx#L210
        if not autocorr:
            dlimit_positions = np.concatenate([dlimit_positions1, dlimit_positions2], axis=0)

        kwargs = {'leafsize': 16, 'compact_nodes': True, 'copy_data': False, 'balanced_tree': True}
        for name in kwargs:
            if name in self.attrs: kwargs[name] = self.attrs[name]

        result = [np.zeros_like(self.modes) for ell in ells]
        legendre = [special.legendre(ell) for ell in ells]

        def normalize(los):
            return los/utils.distance(los.T)[:,None]

        def power_slab(distance, mu, weight):
            for ill, ell in enumerate(ells):
                tmp = weight * special.spherical_jn(ell, self.modes[:,None]*distance, derivative=False) * legendre[ill](mu)
                result[ill] += np.sum(tmp, axis=-1)

        tree = spatial.cKDTree(dlimit_positions, **kwargs, boxsize=None)
        pairs = tree.query_pairs(self.limits[1], p=2.0, eps=0, output_type='ndarray')
        distance = utils.distance((dlimit_positions[pairs[:,0]] - dlimit_positions[pairs[:,1]]).T)
        pairs = pairs[(distance > 0.) & (distance >= self.limits[0]) & (distance < self.limits[1])]

        if not autocorr: # Let us remove restrict to the pairs 1 <-> 2 (removing 1 <-> 1 and 2 <-> 2)
            pairs = pairs[(pairs[:,0] < dlimit_positions1.shape[0]) & (pairs[:,1] >= dlimit_positions1.shape[0])]
            pairs[:,1] -= dlimit_positions1.shape[0]
        del tree
        del dlimit_positions

        dpositions1, dpositions2 = dpositions1[pairs[:,0]], dpositions2[pairs[:,1]]
        dweights1, dweights2 = [w[pairs[:,0]] for w in dweights1], [w[pairs[:,1]] for w in dweights2]
        del pairs
        del distance

        weight = 1.
        if self.n_bitwise_weights:
            weight = self._get_inverse_probability_weight(dweights1[:self.n_bitwise_weights], dweights2[:self.n_bitwise_weights])
        if self.weight_type == 'inverse_bitwise_minus_individual':
            weight -= self._get_inverse_probability_weight(dweights1[:self.n_bitwise_weights]) * self._get_inverse_probability_weight(dweights2[:self.n_bitwise_weights])
        if len(dweights1) > self.n_bitwise_weights:
            weight *= dweights1[-1] * dweights2[-1] # single individual weight, at the end

        diff = dpositions2 - dpositions1
        distance = utils.distance(diff.T)
        if self.los_type == 'global':
            los = self.los
            mu = np.sum(diff * los, axis=-1)/distance
        else:
            if self.los_type in ['firstpoint', 'endpoint']:
                # Calculation using the endpoint los; we switch to the firstpoint los at the end
                mu1 = np.sum(diff * normalize(dpositions2), axis=-1)/distance
                if autocorr:
                    mu2 = - np.sum(diff * normalize(dpositions1), axis=-1)/distance # i>j and i<j
            elif self.los_type == 'midpoint':
                mu = np.sum(diff * normalize(dpositions1 + dpositions2), axis=-1)/distance
        del diff

        # To avoid memory issues when performing distance*modes product, work by slabs
        nslabs_pairs = len(ells) * len(self.modes)
        npairs = distance.size

        for islab in range(nslabs_pairs):
            sl = slice(islab*npairs//nslabs_pairs, (islab+1)*npairs//nslabs_pairs, 1)
            d = distance[sl]
            w = 1. if np.ndim(weight) == 0 else weight[sl]
            if self.los_type in ['global', 'midpoint']:
                power_slab(d, mu[sl], (1. + autocorr)*w) # 2 factor as i < j
            else: # firstpoint, endpoint
                power_slab(d, mu1[sl], w)
                if autocorr:
                    power_slab(d, mu2[sl], w)

        for ill, ell in enumerate(ells):
            # Note: in arXiv:1912.08803, eq. 26, should rather be sij = rj - ri
            result[ill] = (-1j)**ell * (2 * ell + 1) * result[ill]
            if ell % 2 == 1 and self.los_type == 'firstpoint':
                result[ill] = result[ill].conj()

        self.power_nonorm = self.mpicomm.allreduce(np.asarray([result[ells.index(ell)] for ell in self.ells]))

        stop = time.time()
        if rank == 0:
            self.log_info('Direct power computed in elapsed time {:.2f} s.'.format(stop - start))
