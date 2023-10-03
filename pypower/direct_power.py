r"""
Implementation of direct estimation of power spectrum multipoles, i.e. summing over particle pairs.
This should be mostly used to sum over pairs at small transverse separations, otherwise the calculation will be prohibitive.
"""

import os
import time

import numpy as np
from scipy import special

from .utils import BaseClass
from . import mpi, utils


def _normalize(array):
    return array / utils.distance(array.T)[:, None]


def get_default_nrealizations(weights):
    """Return default number of realizations given input bitwise weights = the number of bits in input weights plus one."""
    return 1 + 8 * sum(weight.dtype.itemsize for weight in weights)


def _vlogical_and(*arrays):
    # & between any number of arrays
    toret = arrays[0].copy()
    for array in arrays[1:]: toret &= array
    return toret


def get_inverse_probability_weight(*weights, noffset=1, nrealizations=None, default_value=0., dtype='f8'):
    r"""
    Return inverse probability weight given input bitwise weights.
    Inverse probability weight is computed as: :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2} \& ...))`.
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

    dtype : string, np.dtype
        Type for output weight.

    Returns
    -------
    weight : array
        IIP weight.
    """
    if nrealizations is None:
        nrealizations = get_default_nrealizations(weights[0])
    # denom = noffset + sum(utils.popcount(w1 & w2) for w1, w2 in zip(*weights))
    denom = noffset + sum(utils.popcount(_vlogical_and(*weight)) for weight in zip(*weights))
    mask = denom == 0
    denom[mask] = 1
    toret = np.empty_like(denom, dtype=dtype)
    toret[...] = nrealizations / denom
    toret[mask] = default_value
    return toret


def _format_positions(positions, position_type='xyz', dtype=None, copy=True, mpicomm=None, mpiroot=None):
    # Format input array of positions
    # position_type in ["xyz", "rdd", "pos"]

    def __format_positions(positions):
        if position_type == 'pos':  # array of shape (N, 3)
            positions = np.array(positions, dtype=dtype, copy=copy)
            if not np.issubdtype(positions.dtype, np.floating):
                return None, 'Input position arrays should be of floating type, not {}'.format(positions.dtype)
            if positions.shape[-1] != 3:
                return None, 'For position type = {}, please provide a (N, 3) array for positions'.format(position_type)
            return positions, None
        # Array of shape (3, N)
        positions = list(positions)
        for ip, p in enumerate(positions):
            # Cast to the input dtype if exists (may be set by previous positions)
            positions[ip] = np.array(p, dtype=dtype, copy=copy)
        size = len(positions[0])
        dt = positions[0].dtype
        if not np.issubdtype(dt, np.floating):
            return None, 'Input position arrays should be of floating type, not {}'.format(dt)
        for p in positions[1:]:
            if len(p) != size:
                return None, 'All position arrays should be of the same size'
            if p.dtype != dt:
                return None, 'All position arrays should be of the same type, you can e.g. provide dtype'
        if len(positions) != 3:
            return None, 'For position type = {}, please provide a list of 3 arrays for positions (found {:d})'.format(position_type, len(positions))
        if position_type == 'rdd':  # RA, Dec, distance
            positions = utils.sky_to_cartesian(positions, degree=True)
        elif position_type != 'xyz':
            return None, 'Position type should be one of ["pos", "xyz", "rdd"]'
        return np.asarray(positions).T, None

    error = None
    if mpiroot is None or (mpicomm.rank == mpiroot):
        if positions is not None and (position_type == 'pos' or not all(position is None for position in positions)):
            positions, error = __format_positions(positions)  # return error separately to raise on all processes
    if mpicomm is not None:
        error = mpicomm.allgather(error)
    else:
        error = [error]
    errors = [err for err in error if err is not None]
    if errors:
        raise ValueError(errors[0])
    if mpiroot is not None and mpicomm.bcast(positions is not None if mpicomm.rank == mpiroot else None, root=mpiroot):
        positions = mpi.scatter(positions, mpicomm=mpicomm, mpiroot=mpiroot)
    return positions


def _format_weights(weights, weight_type='auto', size=None, dtype=None, copy=True, mpicomm=None, mpiroot=None):
    # Format input weights, as a list of n_bitwise_weights uint8 arrays, and optionally a float array for individual weights.
    # Return formated list of weights, and n_bitwise_weights.
    def __format_weights(weights):
        islist = isinstance(weights, (tuple, list)) or getattr(weights, 'ndim', 1) == 2
        if not islist:
            weights = [weights]
        if all(weight is None for weight in weights):
            return [], 0
        individual_weights, bitwise_weights = [], []
        for w in weights:
            if np.issubdtype(w.dtype, np.integer):
                if weight_type == 'product_individual':  # enforce float individual weight
                    individual_weights.append(w)
                else:  # certainly bitwise weight
                    bitwise_weights.append(w)
            else:
                individual_weights.append(w)
        # any integer array bit size will be a multiple of 8
        bitwise_weights = utils.reformat_bitarrays(*bitwise_weights, dtype=np.uint8, copy=copy)
        n_bitwise_weights = len(bitwise_weights)
        weights = bitwise_weights
        if individual_weights:
            if len(individual_weights) > 1 or copy:
                weight = np.prod(individual_weights, axis=0, dtype=dtype)
            else:
                weight = individual_weights[0].astype(dtype, copy=False)
            weights += [weight]
        return weights, n_bitwise_weights

    weights, n_bitwise_weights = __format_weights(weights)
    if mpiroot is None:
        size_weights = mpicomm.allgather(len(weights))
        if len(set(size_weights)) != 1:
            raise ValueError('mpiroot = None but weights are None/empty on some ranks')
    else:
        n = mpicomm.bcast(len(weights) if mpicomm.rank == mpiroot else None, root=mpiroot)
        if mpicomm.rank != mpiroot: weights = [None] * n
        weights = [mpi.scatter(weight, mpicomm=mpicomm, mpiroot=mpiroot) for weight in weights]
        n_bitwise_weights = mpicomm.bcast(n_bitwise_weights, root=mpiroot)

    if size is not None:
        if not all(len(weight) == size for weight in weights):
            raise ValueError('All weight arrays should be of the same size as position arrays')
    return weights, n_bitwise_weights


def get_direct_power_engine(engine='corrfunc'):
    """
    Return :class:`BaseDirectPowerEngine`-subclass corresponding
    to input engine name.

    Parameters
    ----------
    engine : string, default='kdtree'
        Name of direct power engine, one of ['kdtree', 'corrfunc'].

    Returns
    -------
    engine : type
        Direct power engine class.
    """
    if isinstance(engine, str):

        try:
            engine = BaseDirectPowerEngine._registry[engine.lower()]
        except KeyError:
            raise ValueError('Unknown engine {}'.format(engine))

    return engine


class RegisteredDirectPowerEngine(type(BaseClass)):

    """Metaclass registering :class:`BaseDirectPowerEngine`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


class MetaDirectPower(type(BaseClass)):

    """Metaclass to return correct direct power engine."""

    def __call__(cls, *args, engine='corrfunc', **kwargs):
        return get_direct_power_engine(engine)(*args, **kwargs)


class DirectPower(metaclass=MetaDirectPower):
    """
    Entry point to direct power engines.

    Parameters
    ----------
    engine : string, default='kdtree'
        Name of direct power engine, one of ['kdtree', 'corrfunc'].

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


class BaseDirectPowerEngine(BaseClass, metaclass=RegisteredDirectPowerEngine):

    """Direct power spectrum measurement, summing over particle pairs."""

    name = 'base'
    _slab_nobjs_max = 1000 * 1000

    def __init__(self, modes, positions1, positions2=None, weights1=None, weights2=None, ells=(0, 2, 4), selection_attrs=None,
                 position_type='xyz', weight_type='auto', weight_attrs=None, twopoint_weights=None, los='firstpoint',
                 dtype='f8', mpiroot=None, mpicomm=mpi.COMM_WORLD, **kwargs):
        r"""
        Initialize :class:`BaseDirectPowerEngine`.

        Parameters
        ----------
        modes : array
            Wavenumbers at which to compute power spectrum.

        positions1 : list, array
            Positions in the first data catalog. Typically of shape (3, N) or (N, 3).

        positions2 : list, array, default=None
            Optionally, for cross-power, positions in the second catalog. See ``positions1``.

        weights1 : array, list, default=None
            Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".
            See ``weight_type``.

        weights2 : array, list, default=None
            Optionally, for cross-power, weights in the second catalog. See ``weights1``.

        ells : list, tuple, default=(0, 2, 4)
            Multipole orders.

        position_type : string, default='xyz'
            Type of input positions, one of:

                - "pos": Cartesian positions of shape (N, 3)
                - "xyz": Cartesian positions of shape (3, N)
                - "rdd": RA/Dec in degree, distance of shape (3, N)

            If ``position_type`` is "pos", positions are of (real) type ``dtype``, and ``mpiroot`` is ``None``,
            no internal copy of positions will be made, hence saving some memory.

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

            In addition, angular upweights can be provided with ``twopoint_weights``.
            If floating weights are of (real) type ``dtype`` and ``mpiroot`` is ``None``,
            no internal copy of weights will be made, hence saving some memory.

        weight_attrs : dict, default=None
            Dictionary of weighting scheme attributes. In case ``weight_type`` is "inverse_bitwise",
            one can provide "nrealizations", the total number of realizations (*including* current one;
            defaulting to the number of bits in input weights plus one);
            "noffset", the offset to be added to the bitwise counts in the denominator (defaulting to 1)
            and "default_value", the default value of pairwise weights if the denominator is zero (defaulting to 0).
            Inverse probability weight is then computed as: :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2}))`.
            For example, for the "zero-truncated" estimator (arXiv:1912.08803), one would use noffset = 0.

        twopoint_weights : WeightTwoPointEstimator, default=None
            Weights to be applied to each pair of particles.
            A :class:`WeightTwoPointEstimator` instance (from *pycorr*) or any object with arrays ``sep``
            (separations) and ``weight`` (weight at given separation) as attributes
            (i.e. to be accessed through ``twopoint_weights.sep``, ``twopoint_weights.weight``)
            or as keys (i.e. ``twopoint_weights['sep']``, ``twopoint_weights['weight']``)
            or as element (i.e. ``sep, weight = twopoint_weights``).

        selection_attrs : dict, default={'theta': (0., 2 / 60.)}
            To select pairs to be counted, provide mapping between the quantity (string)
            and the interval (tuple of floats),
            e.g. ``{'rp': (0., 20.)}`` to select pairs with 'rp' between 0 and 20.
            ``{'theta': (0., 1.)}`` to select pairs with 'theta' between 0 and 1 degree.

        los : string, array, default=None
            If ``los`` is 'firstpoint' (resp. 'endpoint', 'midpoint'), use local (varying) first-point (resp. end-point, mid-point) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        dtype : string, dtype, default='f8'
            The data type to use for input positions and weights.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scatted across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=MPI.COMM_WORLD
            The MPI communicator.
        """
        self.mpicomm = mpicomm
        self.dtype = np.dtype(dtype)
        self._set_modes(modes)
        self._set_los(los)
        self._set_ells(ells)
        self._set_positions(positions1, positions2=positions2, position_type=position_type, mpiroot=mpiroot)
        self._set_weights(weights1, weights2=weights2, weight_type=weight_type, twopoint_weights=twopoint_weights, weight_attrs=weight_attrs, mpiroot=mpiroot)
        self._set_selection(selection_attrs)
        self.is_reversible = self.autocorr or (self.los_type not in ['firstpoint', 'endpoint'])
        self.attrs = kwargs
        t0 = time.time()
        self.run()
        t1 = time.time()
        if self.mpicomm.rank == 0:
            self.log_info('Direct power computed in elapsed time {:.2f} s.'.format(t1 - t0))
        del self.positions1, self.positions2, self.weights1, self.weights2

    def _set_ells(self, ells):
        # Set :attr:`ells`
        if ells is None:
            if self.los_type != 'global':
                raise ValueError('Specify non-empty list of ells')
            self.ells = None
        else:
            if np.ndim(ells) == 0:
                ells = (ells,)
            self.ells = tuple(ells)
            if self.los is None and not self.ells:
                raise ValueError('Specify non-empty list of ells')
            if any(ell < 0 for ell in self.ells):
                raise ValueError('Multipole numbers must be non-negative integers')

    def _set_los(self, los):
        # Set :attr:`los`
        self.los_type = 'midpoint'
        if los is None:
            self.los_type = 'firstpoint'
            self.los = None
        elif isinstance(los, str):
            los = los.lower()
            allowed_los = ['firstpoint', 'endpoint', 'midpoint']
            if los not in allowed_los:
                raise ValueError('los should be one of {}'.format(allowed_los))
            self.los_type = los
            self.los = None
        else:
            raise ValueError('Wrong input los')

    def _set_modes(self, modes):
        self.modes = np.asarray(modes)

    def _set_positions(self, positions1, positions2=None, position_type='xyz', mpiroot=None):
        if position_type is not None: position_type = position_type.lower()
        self.position_type = position_type

        self.positions1 = _format_positions(positions1, position_type=self.position_type, dtype=self.dtype, copy=False, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.positions2 = _format_positions(positions2, position_type=self.position_type, dtype=self.dtype, copy=False, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.autocorr = self.positions2 is None
        self.size1 = self.size2 = self.mpicomm.allreduce(len(self.positions1))
        if not self.autocorr:
            self.size2 = self.mpicomm.allreduce(len(self.positions2))

    def _set_weights(self, weights1, weights2=None, weight_type='auto', twopoint_weights=None, weight_attrs=None, mpiroot=None):

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

            self.weights1, n_bitwise_weights1 = _format_weights(weights1, weight_type=weight_type, size=len(self.positions1), dtype=self.dtype, copy=False, mpicomm=self.mpicomm, mpiroot=mpiroot)

            def get_nrealizations(weights):
                nrealizations = weight_attrs.get('nrealizations', None)
                if nrealizations is None: nrealizations = get_default_nrealizations(weights)
                return nrealizations

            self.weights2, n_bitwise_weights2 = _format_weights(weights2, weight_type=weight_type, size=len(self.positions1 if self.autocorr else self.positions2), dtype=self.dtype, copy=False, mpicomm=self.mpicomm, mpiroot=mpiroot)
            self.same_shotnoise = self.autocorr and bool(self.weights2)

            if self.same_shotnoise:
                self.positions2 = self.positions1
                self.autocorr = False

            if self.autocorr:

                self.weights2 = self.weights1
                self.weight_attrs['nrealizations'] = get_nrealizations(self.weights1[:n_bitwise_weights1])
                self.n_bitwise_weights = n_bitwise_weights1

            else:
                if n_bitwise_weights2 == n_bitwise_weights1:

                    self.weight_attrs['nrealizations'] = get_nrealizations(self.weights1[:n_bitwise_weights1])
                    self.n_bitwise_weights = n_bitwise_weights1

                else:
                    if n_bitwise_weights2 == 0:
                        indweights = self.weights1[n_bitwise_weights1] if len(self.weights1) > n_bitwise_weights1 else 1.
                        self.weight_attrs['nrealizations'] = get_nrealizations(self.weights1[:n_bitwise_weights1])
                        self.weights1 = [self._get_inverse_probability_weight(self.weights1[:n_bitwise_weights1]) * indweights]
                        self.n_bitwise_weights = 0
                        if self.mpicomm.rank == 0: self.log_info('Setting IIP weights for first catalog.')
                    elif n_bitwise_weights1 == 0:
                        indweights = self.weights2[n_bitwise_weights2] if len(self.weights2) > n_bitwise_weights2 else 1.
                        self.weight_attrs['nrealizations'] = get_nrealizations(self.weights2[:n_bitwise_weights2])
                        self.weights2 = [self._get_inverse_probability_weight(self.weights2[:n_bitwise_weights2]) * indweights]
                        self.n_bitwise_weights = 0
                        if self.mpicomm.rank == 0: self.log_info('Setting IIP weights for second catalog.')
                    else:
                        raise ValueError('Incompatible length of bitwise weights: {:d} and {:d} bytes'.format(n_bitwise_weights1, n_bitwise_weights2))

        if len(self.weights1) == len(self.weights2) + 1:
            self.weights2.append(np.ones(len(self.positions2), dtype=self.dtype))
        elif len(self.weights1) == len(self.weights2) - 1:
            self.weights1.append(np.ones(len(self.positions1), dtype=self.dtype))
        elif len(self.weights1) != len(self.weights2):
            raise ValueError('Something fishy happened with weights; number of weights1/weights2 is {:d}/{:d}'.format(len(self.weights1), len(self.weights2)))

        self.twopoint_weights = twopoint_weights
        self.cos_twopoint_weights = None
        if twopoint_weights is not None:
            from collections import namedtuple
            TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
            try:
                sep = twopoint_weights.sep
                weight = twopoint_weights.weight
            except AttributeError:
                try:
                    sep = twopoint_weights['sep']
                    weight = twopoint_weights['weight']
                except IndexError:
                    sep, weight = twopoint_weights
            # just to make sure we use the correct dtype
            sep = np.cos(np.radians(np.array(sep, dtype=self.dtype)))
            argsort = np.argsort(sep)
            self.cos_twopoint_weights = TwoPointWeight(sep=np.array(sep[argsort], dtype=self.dtype), weight=np.array(weight[argsort], dtype=self.dtype))

    def _get_inverse_probability_weight(self, *weights):
        return get_inverse_probability_weight(*weights, noffset=self.weight_attrs['noffset'], nrealizations=self.weight_attrs['nrealizations'],
                                              default_value=self.weight_attrs['default_value'], dtype=self.dtype)

    def _set_selection(self, selection_attrs=None):
        self.selection_attrs = {str(name): tuple(float(v) for v in value) for name, value in (selection_attrs or {'theta': (0., 2. / 60.)}).items()}
        allowed_selections = ['theta', 'rp']
        for name in self.selection_attrs:
            if name not in allowed_selections:
                raise ValueError('selections should be one of {}.'.format(allowed_selections))
        self.rlimits = [0., 2.]
        if 'theta' in self.selection_attrs:
            self.rlimits = 2 * np.sin(0.5 * np.deg2rad(self.selection_attrs['theta']))
        if 'rp' in self.selection_attrs:
            rmin = min(self.mpicomm.allgather(min(np.min(utils.distance(positions.T)) if positions is not None and positions.size else np.inf for positions in [self.positions1, self.positions2])))
            rmax = max(self.mpicomm.allgather(max(np.max(utils.distance(positions.T)) if positions is not None and positions.size else -np.inf for positions in [self.positions1, self.positions2])))
            limits = np.array(self.selection_attrs['rp'], dtype='f8') / np.array([rmax, rmin])
            self.rlimits = [max(limits[0], self.rlimits[0]), min(limits[1], self.rlimits[1])]
        self.rlimits = (max(self.rlimits[0], 0.), min(self.rlimits[1], 2.))

    def _mpi_decompose(self):
        positions1, weights1 = self.positions1, self.weights1
        positions2, weights2 = self.positions2, self.weights2
        limit_positions1 = self.positions1 / utils.distance(self.positions1.T)[:, None]
        limit_positions2 = self.positions2 / utils.distance(self.positions2.T)[:, None] if not self.autocorr else None
        if not self.with_mpi:
            yield (limit_positions1, positions1, weights1), (limit_positions2, positions2, weights2)

        if self.with_mpi:
            if self.autocorr:
                limit_positions2, positions2, weights2 = limit_positions1, positions1, weights1
            size2 = len(positions2)
            csize2 = self.mpicomm.allreduce(size2)
            nslabs = min(int(csize2 / self._slab_nobjs_max + 1.), max(csize2, 1))
            for islab in range(nslabs):
                sl = slice(islab * size2 // nslabs, (islab + 1) * size2 // nslabs)
                tmp_positions1, tmp_weights1 = limit_positions1, [positions1] + weights1
                if nslabs == 1 and self.autocorr:
                    tmp_positions2, tmp_weights2 = None, None
                else:
                    tmp_positions2, tmp_weights2 = limit_positions2[sl], [positions2[sl]] + [weight[sl] for weight in weights2]
                (tmp_limit_positions1, tmp_weights1), (tmp_limit_positions2, tmp_weights2) = mpi.domain_decompose(self.mpicomm, self.rlimits[1], tmp_positions1, weights1=tmp_weights1,
                                                                                                                  positions2=tmp_positions2, weights2=tmp_weights2)
                # We remove original positions from the list of weights
                tmp_positions1, tmp_weights1 = tmp_weights1[0], tmp_weights1[1:]
                if tmp_weights2 is not None:
                    tmp_positions2, tmp_weights2 = tmp_weights2[0], tmp_weights2[1:]
                yield (tmp_limit_positions1, tmp_positions1, tmp_weights1), (tmp_limit_positions2, tmp_positions2, tmp_weights2)

    def _twopoint_weights(self, weights1, weights2=None, positions1=None, positions2=None):
        weights = np.array(1., dtype=self.dtype)
        if self.twopoint_weights is not None:
            if positions1 is not None and positions2 is not None:
                costheta = np.sum(_normalize(positions1) * _normalize(positions2), axis=-1)
            else:
                costheta = 1.
            weights = weights * np.interp(costheta, self.cos_twopoint_weights.sep, self.cos_twopoint_weights.weight, left=1., right=1.)
        if self.n_bitwise_weights:
            weights = weights * self._get_inverse_probability_weight(weights1[:self.n_bitwise_weights], weights2[:self.n_bitwise_weights])
        if self.weight_type == 'inverse_bitwise_minus_individual':
            if self.n_bitwise_weights:
                weights = weights - self._get_inverse_probability_weight(weights1[:self.n_bitwise_weights]) * self._get_inverse_probability_weight(weights2[:self.n_bitwise_weights])
            else:
                if self.twopoint_weights is None:
                    raise ValueError('{} without bitwise weights and twopoint_weights will yield zero total weights!'.format(self.weight_type))
                weights = weights - 1.  # twopoint_weights are provided, so we compute twopoint_weights - 1
        if len(weights1) > self.n_bitwise_weights:
            weights = weights * weights1[-1] * weights2[-1]  # single individual weight, at the end
        return weights

    def _sum_auto_weights(self):
        """Return auto-counts, that are pairs of same objects."""
        if not self.autocorr and not self.same_shotnoise:
            return 0.
        weights = self._twopoint_weights(self.weights1, self.weights2)
        if weights.ndim == 0:
            return self.size1 * weights
        weights = np.sum(weights)
        if self.with_mpi:
            weights = self.mpicomm.allreduce(weights)
        return weights

    def run(self):
        """Method that computes the power spectrum and set :attr:`power_nonorm`, to be implemented in your new engine."""
        raise NotImplementedError('Implement method "run" in your {}'.format(self.__class__.__name__))

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def __getstate__(self):
        state = {}
        for name in ['name', 'autocorr', 'is_reversible', 'modes', 'ells', 'power_nonorm', 'size1', 'size2', 'rlimits',
                     'los', 'los_type', 'weight_attrs', 'selection_attrs', 'attrs']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        super(BaseDirectPowerEngine, self).__setstate__(state)
        # Backward-compatibility
        if not hasattr(self, 'selection_attrs'):
            self.selection_attrs = {}
        if not hasattr(self, 'rlimits'):
            self.rlimits = self.limits

    def reversed(self):
        if not self.is_reversible:
            raise ValueError('This measurement is not reversible')
        new = self.deepcopy()
        new.size1, new.size2 = new.size2, new.size1
        for ill, ell in enumerate(self.ells):
            new.power_nonorm[ill] *= (-1)**(ell % 2)
        return new


class KDTreeDirectPowerEngine(BaseDirectPowerEngine):

    """Direct power spectrum measurement, summing over particle pairs, identified with KDTree."""

    name = 'kdtree'
    _slab_npairs_max = 1000 * 1000

    def run(self):
        from scipy import spatial
        rank = self.mpicomm.rank
        ells = sorted(set(self.ells))

        kwargs = {'leafsize': 16, 'compact_nodes': True, 'copy_data': False, 'balanced_tree': True}
        for name in kwargs:
            if name in self.attrs: kwargs[name] = self.attrs[name]

        legendre = [special.legendre(ell) for ell in ells]

        # We proceed by slab to avoid blowing up the memory
        def tree_slab(d1, d2, **kwargs):
            if d2[0] is None: d2 = d1
            swap = len(d2[0]) < len(d1[0])
            if swap:
                d1, d2 = d2, d1
            # First estimate number of pairs from a subsample
            size1, size2 = len(d1[0]), len(d2[0])
            min_npairs, seed = 100, 42
            npairs, size_max = 0, -1
            while (npairs < min_npairs) and ((size_max < size1) or (size_max < size2)):
                size_max += 10000
                size1_downsample, size2_downsample = min(size1, size_max), min(size2, size_max)
                rng = np.random.RandomState(seed=seed)
                dpositions = np.concatenate([d[0][rng.choice(size, size_downsample, replace=False)] for d, size, size_downsample
                                             in zip([d1, d2], [size1, size2], [size1_downsample, size2_downsample])])
                tree = spatial.cKDTree(dpositions, **kwargs, boxsize=None)
                npairs = len(tree.query_pairs(self.rlimits[1], p=2.0, eps=0, output_type='ndarray'))
            npairs_downsample = 1 + 3 / max(npairs, 1)**0.5  # 3 sigma margin
            npairs_downsample *= size1 / max(size1_downsample, 1) * size2 / max(size2_downsample, 1)  # scale to size of d1 & d2
            nslabs = min(int(npairs_downsample / self._slab_npairs_max + 1.), len(d2[0]))
            if nslabs == 1:  # do not touch autocorrelation
                yield (d2, d1) if swap else (d1, d2)
            else:
                for islab in range(nslabs):
                    sl = slice(islab * size2 // nslabs, (islab + 1) * size2 // nslabs)
                    tmp2 = tuple(d[sl] for d in d2[:-1]) + ([d[sl] for d in d2[-1]],)
                    yield (tmp2, d1) if swap else (d1, tmp2)

        def power_slab(poles, distance, mu, weight, ells):
            for ill, ell in enumerate(ells):
                tmp = weight * special.spherical_jn(ell, self.modes[:, None] * distance, derivative=False) * legendre[ill](mu)
                poles[ill] += np.sum(tmp, axis=-1)

        sum_poles = 0.
        delta_tree, delta_sum = 0., 0.

        for d1, d2 in self._mpi_decompose():

            poles = np.zeros((len(ells), len(self.modes)), dtype=self.dtype)

            for (dlimit_positions1, dpositions1, dweights1), (dlimit_positions2, dpositions2, dweights2) in tree_slab(d1, d2, **kwargs):

                # dlimit_positions = dlimit_positions1
                # Very unfortunately, cKDTree.query_pairs does not handle cross-correlations...
                # But I feel this could be changed super easily here:
                # https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/spatial/ckdtree/src/query_pairs.cxx#L210
                dlimit_positions = np.concatenate([dlimit_positions1, dlimit_positions2], axis=0)

                start_i = time.time()
                tree = spatial.cKDTree(dlimit_positions, **kwargs, boxsize=None)
                pairs = tree.query_pairs(self.rlimits[1], p=2.0, eps=0, output_type='ndarray')
                delta_tree += time.time() - start_i
                start_i = time.time()
                distance = utils.distance((dlimit_positions[pairs[:, 0]] - dlimit_positions[pairs[:, 1]]).T)
                pairs = pairs[(distance >= self.rlimits[0]) & (distance < self.rlimits[1])]

                # if not autocorr: # Let us remove restrict to the pairs 1 <-> 2 (removing 1 <-> 1 and 2 <-> 2)
                pairs = pairs[(pairs[:, 0] < dlimit_positions1.shape[0]) & (pairs[:, 1] >= dlimit_positions1.shape[0])]
                pairs[:, 1] -= dlimit_positions1.shape[0]
                del tree
                del dlimit_positions

                dpositions1, dpositions2 = dpositions1[pairs[:, 0]], dpositions2[pairs[:, 1]]
                dweights1, dweights2 = [w[pairs[:, 0]] for w in dweights1], [w[pairs[:, 1]] for w in dweights2]
                del pairs
                del distance

                weights = self._twopoint_weights(weights1=dweights1, weights2=dweights2, positions1=dpositions1, positions2=dpositions2)

                diff = dpositions2 - dpositions1
                distances = utils.distance(diff.T)
                mask_zero = distances == 0
                distances[mask_zero] = 1.
                if self.los_type == 'global':
                    los = self.los
                    mu = np.sum(diff * los, axis=-1) / distances
                else:
                    if self.los_type == 'firstpoint':
                        mu = np.sum(diff * _normalize(dpositions1), axis=-1) / distances
                        # if autocorr:
                        #     mu2 = - np.sum(diff * _normalize(dpositions2), axis=-1)/distances # i>j and i<j
                    elif self.los_type == 'endpoint':
                        mu = np.sum(diff * _normalize(dpositions2), axis=-1) / distances
                        # if autocorr:
                        #     mu2 = - np.sum(diff * _normalize(dpositions1), axis=-1)/distances # i>j and i<j
                    elif self.los_type == 'midpoint':
                        mu = np.sum(diff * _normalize(dpositions1 + dpositions2), axis=-1) / distances
                del diff
                distances[mask_zero] = 0.
                if 'rp' in self.selection_attrs:
                    rp2 = (1. - mu**2) * distances**2
                    mask = (rp2 >= self.selection_attrs['rp'][0]**2) & (rp2 < self.selection_attrs['rp'][1]**2)
                    distances, mu, weights = distances[mask], mu[mask], (weights[mask] if weights.ndim else weights)

                # To avoid memory issues when performing distance*modes product, work by slabs
                nslabs_pairs = len(ells) * len(self.modes)
                npairs = distances.size

                for islab in range(nslabs_pairs):
                    sl = slice(islab * npairs // nslabs_pairs, (islab + 1) * npairs // nslabs_pairs, 1)
                    d = distances[sl]
                    w = 1. if weights.ndim == 0 else weights[sl]
                    if self.los_type in ['global', 'midpoint']:
                        power_slab(poles, d, mu[sl], w, ells)
                    else:  # firstpoint, endpoint
                        power_slab(poles, d, mu[sl], w, ells)
                        # if autocorr:
                        #     power_slab(poles, d, mu2[sl], w, ells)

                delta_sum += time.time() - start_i

            sum_poles += self.mpicomm.allreduce(poles)

        if rank == 0:
            self.log_info('Building tree took {:.2f} s.'.format(delta_tree))
            self.log_info('Sum over pairs took {:.2f} s.'.format(delta_sum))

        self.power_nonorm = sum_poles
        with_auto_pairs = self.rlimits[0] <= 0. and all(limits[0] <= 0. for limits in self.selection_attrs.values())
        if self.autocorr and with_auto_pairs:  # remove auto-pairs
            power_slab(self.power_nonorm, 0., 0., -self._sum_auto_weights(), ells)

        self.power_nonorm = self.power_nonorm.astype('c16')
        for ill, ell in enumerate(ells):
            # Note: in arXiv:1912.08803, eq. 26, should rather be sij = rj - ri
            self.power_nonorm[ill] = (-1j)**ell * (2 * ell + 1) * self.power_nonorm[ill]

        self.power_nonorm = self.power_nonorm[[ells.index(ell) for ell in self.ells]]


class CorrfuncDirectPowerEngine(BaseDirectPowerEngine):

    """Direct power spectrum measurement, using Corrfunc."""

    name = 'corrfunc'

    @property
    def nthreads(self):
        nthreads = self.attrs.get('nthreads', None)
        if nthreads is None:
            nthreads = int(os.getenv('OMP_NUM_THREADS', '1'))
        return nthreads

    def run(self):
        from Corrfunc import mocks
        ells = sorted(set(self.ells))

        autocorr = self.autocorr and not self.with_mpi
        sum_poles = 0.

        for (dlimit_positions1, dpositions1, dweights1), (dlimit_positions2, dpositions2, dweights2) in self._mpi_decompose():

            if self.los_type not in ['firstpoint', 'endpoint', 'midpoint']:
                raise ValueError('Corrfunc only supports midpoint / firstpoint / endpoint line-of-sight')
            los_type = self.los_type
            if self.los_type == 'endpoint':
                los_type = 'firstpoint'
                if not self.autocorr:
                    dlimit_positions1, dlimit_positions2 = dlimit_positions2, dlimit_positions1
                    dpositions1, dpositions2 = dpositions2, dpositions1
                    dweights1, dweights2 = dweights2, dweights1

            weight_type = None
            weight_attrs = None

            weights1, weights2 = dweights1.copy(), dweights2.copy()  # copy lists
            if self.n_bitwise_weights:
                weight_type = 'inverse_bitwise'
                dtype = {4: np.int32, 8: np.int64}[self.dtype.itemsize]

                def reformat_bitweights(weights):
                    return utils.reformat_bitarrays(*weights[:self.n_bitwise_weights], dtype=dtype) + weights[self.n_bitwise_weights:]

                weights1 = reformat_bitweights(dweights1)
                if not autocorr:
                    weights2 = reformat_bitweights(dweights2)
                weight_attrs = (self.weight_attrs['noffset'], self.weight_attrs['default_value'] / self.weight_attrs['nrealizations'])

            elif dweights1:
                weight_type = 'pair_product'

            pair_weights, sep_pair_weights = None, None
            if self.cos_twopoint_weights is not None:
                weight_type = 'inverse_bitwise'
                pair_weights = self.cos_twopoint_weights.weight
                sep_pair_weights = self.cos_twopoint_weights.sep

            prefactor = self.weight_attrs['nrealizations'] if self.n_bitwise_weights else 1

            if self.weight_type == 'inverse_bitwise_minus_individual':  # let's add weight to be subtracted
                weight_type = 'inverse_bitwise'
                if not dweights1[self.n_bitwise_weights:]:
                    weights1.append(np.ones(len(dlimit_positions1), dtype=self.dtype))
                    if not autocorr:
                        weights2.append(np.ones(len(dlimit_positions2), dtype=self.dtype))
                if self.n_bitwise_weights:
                    weights1.append(1. / prefactor**0.5 * self._get_inverse_probability_weight(dweights1[:self.n_bitwise_weights]) * np.prod(dweights1[self.n_bitwise_weights:], axis=0))
                    if not autocorr:
                        weights2.append(1. / prefactor**0.5 * self._get_inverse_probability_weight(dweights2[:self.n_bitwise_weights]) * np.prod(dweights2[self.n_bitwise_weights:], axis=0))
                else:
                    if self.twopoint_weights is None:
                        raise ValueError('{} without bitwise weights and twopoint_weights will yield zero total weights!'.format(self.weight_type))
                    weights1.append(np.ones(len(dlimit_positions1), dtype=self.dtype) * np.prod(dweights1, axis=0))
                    if not autocorr:
                        weights2.append(np.ones(len(dlimit_positions2), dtype=self.dtype) * np.prod(dweights2, axis=0))

            weights1, weights2 = weights1 if weights1 else None, weights2 if weights2 else None

            kwargs = {'weights1': weights1, 'weights2': weights2,
                    'weight_type': weight_type,
                    'pair_weights': pair_weights, 'sep_pair_weights': sep_pair_weights,
                    'attrs_pair_weights': weight_attrs, 'verbose': False,
                    'isa': self.attrs.get('isa', 'fastest')}

            if 'rp' in self.selection_attrs:
                kwargs['attrs_selection'] = {'rp': self.selection_attrs['rp']}

            limit_positions1, positions1 = dlimit_positions1.T, dpositions1.T
            if autocorr:
                limit_positions2, positions2 = [None] * 3, [None] * 3
            else:
                limit_positions2, positions2 = dlimit_positions2.T, dpositions2.T

            def call_corrfunc(method, *args, **kwargs):
                try:
                    return method(*args, **kwargs)
                except TypeError as exc:
                    raise ValueError('Please reinstall relevant Corrfunc branch (including PIP weights):\n\
                                    > pip uninstall Corrfunc\n\
                                    > pip install git+https://github.com/adematti/Corrfunc@desi\n') from exc

            if self.size1 or self.size2:  # else rlimits is 0, 0 and raise error
                poles = call_corrfunc(mocks.DDbessel_mocks, autocorr, nthreads=self.nthreads,
                                    X1=limit_positions1[0], Y1=limit_positions1[1], Z1=limit_positions1[2], XP1=positions1[0], YP1=positions1[1], ZP1=positions1[2],
                                    X2=limit_positions2[0], Y2=limit_positions2[1], Z2=limit_positions2[2], XP2=positions2[0], YP2=positions2[1], ZP2=positions2[2],
                                    binfile=self.modes, ells=ells, rmin=self.rlimits[0], rmax=self.rlimits[1], mumax=1., los_type=los_type, **kwargs)['poles']
            else:
                poles = np.zeros(len(self.modes) * len(ells), dtype=self.dtype)
            sum_poles += (self.mpicomm.allreduce(poles.reshape(len(self.modes), len(ells)).T) * prefactor).astype('c16')

        self.power_nonorm = sum_poles
        with_auto_pairs = self.rlimits[0] <= 0. and all(limits[0] <= 0. for limits in self.selection_attrs.values())
        if self.autocorr and with_auto_pairs:  # remove auto-pairs
            weights = self._sum_auto_weights()
            for ill, ell in enumerate(ells):
                self.power_nonorm[ill] -= weights * (2 * ell + 1) * special.legendre(ell)(0.) * special.spherical_jn(ell, 0., derivative=False)

        for ill, ell in enumerate(ells):
            # Note: in arXiv:1912.08803, eq. 26, should rather be sij = rj - ri
            self.power_nonorm[ill] = (-1j)**ell * (-1) ** (self.los_type == 'endpoint' and ell % 2) * self.power_nonorm[ill]

        self.power_nonorm = self.power_nonorm[[ells.index(ell) for ell in self.ells]]
