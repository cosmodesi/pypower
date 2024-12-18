r"""
Implementation of direct estimation of correlation function multipoles, i.e. summing over particle pairs.
This should be mostly used to sum over pairs at small transverse separations, otherwise the calculation will be prohibitive.
"""

import os
import time

import numpy as np
from scipy import special

from .utils import BaseClass, _get_box
from .direct_power import get_direct_power_engine, BaseDirectPowerEngine, _normalize
from . import mpi, utils


def get_direct_corr_engine(engine='corrfunc'):
    """
    Return :class:`BaseDirectCorrEngine`-subclass corresponding
    to input engine name.

    Parameters
    ----------
    engine : string, default='kdtree'
        Name of direct corr engine, one of ['kdtree', 'corrfunc'].

    Returns
    -------
    engine : type
        Direct power engine class.
    """
    if isinstance(engine, str):

        try:
            engine = BaseDirectCorrEngine._registry[engine.lower()]
        except KeyError:
            raise ValueError('Unknown engine {}'.format(engine))

    return engine


class RegisteredDirectCorrEngine(type(BaseClass)):

    """Metaclass registering :class:`BaseDirectCorrEngine`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


class MetaDirectCorr(type(BaseClass)):

    """Metaclass to return correct direct correlation engine."""

    def __call__(cls, *args, engine='corrfunc', **kwargs):
        return get_direct_corr_engine(engine)(*args, **kwargs)


class DirectCorr(metaclass=MetaDirectCorr):
    """
    Entry point to direct correlation engines.

    Parameters
    ----------
    engine : string, default='kdtree'
        Name of direct correlation engine, one of ['kdtree', 'corrfunc'].

    args : list
        Arguments for direct correlation engine, see :class:`BaseDirectCorrEngine`.

    kwargs : dict
        Arguments for direct correlation engine, see :class:`BaseDirectCorrEngine`.

    Returns
    -------
    engine : BaseDirectCorrEngine
    """
    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        return get_direct_corr_engine(state.pop('name')).from_state(state)


class BaseDirectCorrEngine(BaseClass, metaclass=RegisteredDirectCorrEngine):

    """Direct measurement of correlation multipoles, summing over particle pairs."""

    name = 'base'

    def __init__(self, edges, positions1, positions2=None, weights1=None, weights2=None, ells=(0, 2, 4), selection_attrs=None,
                 position_type='xyz', weight_type='auto', weight_attrs=None, twopoint_weights=None, los='firstpoint',
                 dtype='f8', mpiroot=None, mpicomm=mpi.COMM_WORLD, **kwargs):
        r"""
        Initialize :class:`BaseDirectCorrEngine`.

        Parameters
        ----------
        edges : array, dict
            Separation bin edges.
            May be a dictionary, with keys 'min' (minimum :math:`s`, defaults to 0),
            'max' (maximum :math:`s`, defaults to maximum separation given input positions),
            and 'step' (defaults to 1).

        positions1 : list, array
            Positions in the first data catalog. Typically of shape (3, N) or (N, 3).

        positions2 : list, array, default=None
            Optionally, for cross-correlation, positions in the second catalog. See ``positions1``.

        weights1 : array, list, default=None
            Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".
            See ``weight_type``.

        weights2 : array, list, default=None
            Optionally, for cross-correlation, weights in the second catalog. See ``weights1``.

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
            The method used to compute the normalization of PIP weights can be specified with the keyword "normalization":
            "counter" to normalize each pair by eq. 19 of arXiv:1912.08803.
            In this case "nalways" specifies the number of bits systematically set to 1 minus the number of bits systematically set to 0 (defaulting to 0).
            For example, for the "zero-truncated" estimator (arXiv:1912.08803), one would use noffset = 0, nalways = 1.

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
            e.g. ``{'rp': (0., 20.)}`` to select pairs with transverse separation 'rp' between 0 and 20,
            `{'theta': (0., 20.)}`` to select pairs with separation angle 'theta' between 0 and 20 degrees.

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
        self._set_los(los)
        self._set_ells(ells)
        self._set_positions(positions1, positions2=positions2, position_type=position_type, mpiroot=mpiroot)
        self._set_edges(edges)
        self._set_weights(weights1, weights2=weights2, weight_type=weight_type, twopoint_weights=twopoint_weights, weight_attrs=weight_attrs, mpiroot=mpiroot)
        self._set_selection(selection_attrs)
        self.is_reversible = self.autocorr or (self.los_type not in ['firstpoint', 'endpoint'])
        self.attrs = kwargs
        t0 = time.time()
        self.run()
        t1 = time.time()
        if self.mpicomm.rank == 0:
            self.log_info('Direct correlation computed in elapsed time {:.2f} s.'.format(t1 - t0))
        del self.positions1, self.positions2, self.weights1, self.weights2

    def _set_edges(self, edges):
        if isinstance(edges, dict):
            smin = edges.get('min', 0.)
            smax = edges.get('max', None)
            ds = edges.get('step', 1.)
            if smax is None:
                pos_min, pos_max = _get_box(self.positions1, self.positions2) if self.positions2 is not None else _get_box(self.positions1)
                pos_min, pos_max = np.min(self.mpicomm.allgather(pos_min), axis=0), np.max(self.mpicomm.allgather(pos_max), axis=0)
                smax = np.sum((pos_max - pos_min)**2)**0.5 * (1. + ds)
            edges = np.arange(smin, smax + 1e-5 * ds, ds)
        if self.mpicomm.rank == 0:
            self.log_info('Using {:d} s-bins between {:.3f} and {:.3f}.'.format(len(edges) - 1, edges[0], edges[-1]))
        self.edges = np.asarray(edges)
        self._set_bin_type()

    def _set_bin_type(self):
        self.bin_type = 'custom'
        if np.allclose(self.edges, np.linspace(self.edges[0], self.edges[-1], len(self.edges))):
            self.bin_type = 'lin'

    def run(self):
        """Method that computes the correlation function and set :attr:`sep` and :attr:`corr_nonorm`, to be implemented in your new engine."""
        raise NotImplementedError('Implement method "run" in your {}'.format(self.__class__.__name__))

    def __getstate__(self):
        state = {}
        for name in ['name', 'autocorr', 'is_reversible', 'edges', 'bin_type', 'ells', 'sep', 'corr_nonorm', 'size1', 'size2', 'rlimits',
                     'los', 'los_type', 'weight_attrs', 'selection_attrs', 'attrs']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        super(BaseDirectCorrEngine, self).__setstate__(state)
        # Backward-compatibility
        if not hasattr(self, 'selection_attrs'):
            self.selection_attrs = {}
        if not hasattr(self, 'rlimits'):
            self.rlimits = self.limits

    def to_power(self, modes):
        state = self.__getstate__()
        for name in ['edges', 'sep', 'corr_nonorm']: state.pop(name)
        modes = state['modes'] = np.asarray(modes)
        sep = self.sep.copy()
        sep[np.isnan(sep)] = (self.edges[:-1] + self.edges[1:])[np.isnan(sep)] / 2.
        power_nonorm = []
        for ill, ell in enumerate(self.ells):
            value = (-1j)**ell * np.sum(self.corr_nonorm[ill] * special.spherical_jn(ell, modes[:, None] * sep, derivative=False), axis=-1)
            power_nonorm.append(value)
        state['power_nonorm'] = np.array(power_nonorm)
        return get_direct_power_engine(engine=self.name).from_state(state)


for name in ['_set_ells', '_set_los', '_set_modes', '_set_positions', '_set_weights', '_get_inverse_probability_weight', '_set_selection', '_mpi_decompose',
             '_twopoint_weights', '_sum_auto_weights', 'deepcopy', 'reversed', '_slab_nobjs_max']:
    setattr(BaseDirectCorrEngine, name, getattr(BaseDirectPowerEngine, name))


class KDTreeDirectCorrEngine(BaseDirectCorrEngine):

    """Direct measurement of correlation multipoles, summing over particle pairs, identified with KDTree."""

    name = 'kdtree'
    _slab_npairs_max = 1000 * 1000

    def run(self):
        from scipy import spatial
        rank = self.mpicomm.rank
        ells = sorted(set((0,) + self.ells))

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

        def corr_slab(poles, distance, mu, weight, ells, sep=None):
            index = np.searchsorted(self.edges, distance, side='right') - 1
            if sep is not None:
                np.add.at(sep, index, distance * weight)
            for ill, ell in enumerate(ells):
                tmp = weight * legendre[ill](mu)
                np.add.at(poles[ill], index, tmp)

        sum_sep = sum_poles = 0.
        delta_tree = delta_sum = 0.

        for d1, d2 in self._mpi_decompose():

            sep = np.zeros(len(self.edges) - 1, dtype=self.dtype)
            poles = np.zeros((len(ells), len(self.edges) - 1), dtype=self.dtype)

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
                mask = (distances >= self.edges[0]) & (distances < self.edges[-1])
                distances, mu, weights = distances[mask], mu[mask], (weights[mask] if weights.ndim else weights)
                if 'rp' in self.selection_attrs:
                    rp2 = (1. - mu**2) * distances**2
                    mask = (rp2 >= self.selection_attrs['rp'][0]**2) & (rp2 < self.selection_attrs['rp'][1]**2)
                    distances, mu, weights = distances[mask], mu[mask], (weights[mask] if weights.ndim else weights)

                # To avoid memory issues when performing distance*modes product, work by slabs
                nslabs_pairs = len(ells) * (len(self.edges) - 1)
                npairs = distances.size

                for islab in range(nslabs_pairs):
                    sl = slice(islab * npairs // nslabs_pairs, (islab + 1) * npairs // nslabs_pairs, 1)
                    d = distances[sl]
                    w = 1. if weights.ndim == 0 else weights[sl]
                    if self.los_type in ['global', 'midpoint']:
                        corr_slab(poles, d, mu[sl], w, ells, sep=sep)
                    else:  # firstpoint, endpoint
                        corr_slab(poles, d, mu[sl], w, ells, sep=sep)
                        # if autocorr:
                        #     power_slab(poles, d, mu2[sl], w, ells)

                delta_sum += time.time() - start_i

            sum_sep += self.mpicomm.allreduce(sep)
            sum_poles += self.mpicomm.allreduce(poles)

        if rank == 0:
            self.log_info('Building tree took {:.2f} s.'.format(delta_tree))
            self.log_info('Sum over pairs took {:.2f} s.'.format(delta_sum))

        with_auto_pairs = self.rlimits[0] <= 0. and self.edges[0] <= 0. and all(limits[0] <= 0. for limits in self.selection_attrs.values())
        if self.autocorr and with_auto_pairs:  # remove auto-pairs
            corr_slab(sum_poles, 0., 0., -self._sum_auto_weights(), ells)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.sep = sum_sep / sum_poles[ells.index(0)]
        self.corr_nonorm = sum_poles.astype('f8')
        for ill, ell in enumerate(ells):
            # Note: in arXiv:1912.08803, eq. 26, should rather be sij = rj - ri
            self.corr_nonorm[ill] = (2 * ell + 1) * self.corr_nonorm[ill]

        self.corr_nonorm = self.corr_nonorm[[ells.index(ell) for ell in self.ells]]


class CorrfuncDirectCorrEngine(BaseDirectCorrEngine):

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
        ells = sorted(set((0,) + self.ells))

        autocorr = self.autocorr and not self.with_mpi
        sum_sep = sum_poles = 0.

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
                weight_attrs = {'noffset': self.weight_attrs['noffset'], 'default_value': self.weight_attrs['default_value'] / self.weight_attrs['nrealizations']}
                correction = self.weight_attrs.get('correction', None)
                if correction is not None: weight_attrs['correction'] = correction

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
                      'isa': self.attrs.get('isa', 'fastest'), 'bin_type': self.bin_type}
            if 'rp' in self.selection_attrs:
                kwargs['attrs_selection'] = {'rp': self.selection_attrs['rp']}
            #kwargs['attrs_selection'] = self.selection_attrs  # theta-cut already included with rlimits

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
                poles = call_corrfunc(mocks.DDleg_mocks, autocorr, nthreads=self.nthreads,
                                      X1=limit_positions1[0], Y1=limit_positions1[1], Z1=limit_positions1[2], XP1=positions1[0], YP1=positions1[1], ZP1=positions1[2],
                                      X2=limit_positions2[0], Y2=limit_positions2[1], Z2=limit_positions2[2], XP2=positions2[0], YP2=positions2[1], ZP2=positions2[2],
                                      binfile=self.edges, ells=ells, rmin=self.rlimits[0], rmax=self.rlimits[1], mumax=1., los_type=los_type, **kwargs)
                sep, poles = poles['savg'], poles['poles']
            else:
                sep = poles = np.zeros((len(self.edges) - 1) * len(ells), dtype=self.dtype)
            sep = sep.reshape(len(self.edges) - 1, len(ells))[:, 0]
            poles = poles.reshape(len(self.edges) - 1, len(ells)).T

            sum_sep += self.mpicomm.allreduce(sep * poles[ells.index(0)])
            sum_poles += self.mpicomm.allreduce(poles)

        with_auto_pairs = self.rlimits[0] <= 0. and self.edges[0] <= 0. and all(limits[0] <= 0. for limits in self.selection_attrs.values())
        if self.autocorr and with_auto_pairs:  # remove auto-pairs
            weights = self._sum_auto_weights()
            for ill, ell in enumerate(ells):
                sum_poles[ill][0] -= weights / prefactor * (2 * ell + 1) * special.legendre(ell)(0.)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.sep = sum_sep / sum_poles[ells.index(0)]
        self.corr_nonorm = (sum_poles * prefactor).astype('f8')
        for ill, ell in enumerate(ells):
            # Note: in arXiv:1912.08803, eq. 26, should rather be sij = rj - ri
            self.corr_nonorm[ill] = (-1) ** (self.los_type == 'endpoint' and ell % 2) * self.corr_nonorm[ill]

        self.corr_nonorm = self.corr_nonorm[[ells.index(ell) for ell in self.ells]]
