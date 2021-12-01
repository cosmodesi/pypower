r"""Implementation of direct estimation of result spectrum multipoles, i.e. looping over particle pairs."""

import time

import numpy as np
from mpi4py import MPI

from . import mpi
from .mesh import _make_array, _format_positions


class BaseDirectPowerEngine(BaseClass):

    """Direct result spectrum measurement."""

    def __init__(self, modes, positions1, weights1=None, positions2=None, weights2=None, ells=(0, 2, 4), limits=(0, np.inf), limit_type='degree',
                 weight_type='auto', weight_attrs=None, los='endpoint', boxsize=None, mpicomm=mpi.COMM_WORLD, **kwargs):
        self.mpicomm = mpicomm
        self._set_modes(modes)
        self._set_ells(ells)
        self._set_los(los)
        self._set_limits(limits, limit_type=limit_type)
        self._set_boxsize(boxsize=boxsize)
        self._set_positions(positions1, positions2=positions2, position_type=position_type)
        self._set_weights(weights1, weights2=weights2, weight_type=weight_type, weight_attrs=weight_attrs)
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
        allowed_limit_types = ['degree', 'radians', 'theta', 's']
        if limit_type not in allowed_weight_types:
            raise ValueError('Limit should be in {}.'.format(allowed_limit_types))
        self.limits = tuple(limits)
        if limit_type == 'radians':
            self.limits = tuple(np.rad2deg(self.limits))
        self.limit_type = limit_type
        if limit_type in ['radians', 'degree']:
            self.limit_type = 'theta'

    def _set_boxsize(self, boxsize):
        self.boxsize = boxsize
        if self.periodic:
            self.boxsize = _make_array(boxsize, 3, dtype='f8')

    def _set_positions(self, positions1, positions2=None):
        self.positions1 = _format_positions(positions1)
        self.autocorr = positions2 is None
        if self.autocorr:
            self.positions2 = None
        else:
            self.positions2 = _format_positions(positions2)

    def _set_weights(self, weights1, weights2=None, weight_type='auto', weight_attrs=None):

        if weight_type is not None: weight_type = weight_type.lower()
        allowed_weight_types = [None, 'auto', 'product_individual', 'inverse_bitwise', 'inverse_bitwise_minus_individual']
        if weight_type not in allowed_weight_types:
            raise ValueError('weight_type should be one of {}'.format(allowed_weight_types))
        self.weight_type = weight_type

        if self.autocorr:
            if weights2 is not None:
                raise ValueError('weights2 are provided, but not positions2')

        if weights1 is None:
            if weights2 is not None:
                raise ValueError('weights2 are provided, but not weights1')
        else:
            if self.autocorr:
                if weights2 is not None:
                    raise ValueError('weights2 are provided, but not positions2')
            else:
                if weights2 is None:
                    raise ValueError('weights1 are provided, but not weights2')

        weight_attrs = weight_attrs or {}
        self.weight_attrs = {}

        self.n_bitwise_weights = 0
        if weight_type is None:
            self.weights1 = self.weights2 = []
        else:
            self.weight_attrs.update(nalways=weight_attrs.get('nalways', 0), nnever=weight_attrs.get('nnever', 0))
            noffset = weight_attrs.get('noffset', 1)
            default_value = weight_attrs.get('default_value', 0.)
            self.weight_attrs.update(noffset=noffset, default_value=default_value)

            def check_shape(weights, size):
                if weights is None or len(weights) == 0:
                    return None, 0
                if np.ndim(weights[0]) == 0:
                    weights = [weights]
                individual_weights = []
                bitwise_weights = []
                for w in weights:
                    if len(w) != size:
                        raise ValueError('All weight arrays should be of the same size as position arrays')
                    if np.issubdtype(w.dtype, np.integer):
                        if weight_type == 'product_individual':
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
                    weights += [np.prod(individual_weights, axis=0, dtype=self.dtype)]
                return weights, n_bitwise_weights

            self.weights1, n_bitwise_weights1 = check_shape(weights1, len(self.positions1[0]))

            def get_nrealizations(n_bitwise_weights):
                nrealizations = weight_attrs.get('nrealizations', None)
                if nrealizations is None:
                    nrealizations = n_bitwise_weights * 8 + 1
                return nrealizations
            #elif self.n_bitwise_weights and self.nrealizations - 1 > self.n_bitwise_weights * 8:
            #    raise ValueError('Provided number of realizations is {:d}, '
            #                           'more than actual number of bits in bitwise weights {:d} plus 1'.format(self.nrealizations, self.n_bitwise_weights * 8))
            self.weights2 = weights2

            if self.autocorr:

                self.weight_attrs['nrealizations'] = get_nrealizations(n_bitwise_weights1)
                self.n_bitwise_weights = n_bitwise_weights1

            else:
                self.weights2, n_bitwise_weights2 = check_shape(weights2, len(self.positions2[0]))

                if n_bitwise_weights2 == n_bitwise_weights1:

                    self.weight_attrs['nrealizations'] = get_nrealizations(n_bitwise_weights1)
                    self.n_bitwise_weights = n_bitwise_weights1

                else:

                    def wiip(weights, nrealizations):
                        denom = noffset + utils.popcount(*weights)
                        mask = denom == 0
                        denom[mask] = 1
                        toret = nrealizations/denom
                        toret[mask] = default_value
                        return toret

                    if n_bitwise_weights2 == 0:
                        indweights = self.weights1[n_bitwise_weights1:]
                        self.weight_attrs['nrealizations'] = get_nrealizations(n_bitwise_weights1)
                        self.weights1 = [self._get_wiip(self.weights1[:n_bitwise_weights1])*(self.weights1[n_bitwise_weights1:] or 1)]
                        self.n_bitwise_weights = 0
                        self.log_info('Setting IIP weights for first catalog.')
                    elif self.n_bitwise_weights == 0:
                        indweights = self.weights2[n_bitwise_weights2:]
                        self.weight_attrs['nrealizations'] = get_nrealizations(n_bitwise_weights2)
                        self.weights2 = [self._get_wiip(self.weights2[:n_bitwise_weights2])*(indweights or 1)]
                        self.n_bitwise_weights = 0
                        self.log_info('Setting IIP weights for second catalog.')
                    else:
                        raise ValueError('Incompatible length of bitwise weights: {:d} and {:d} bytes'.format(n_bitwise_weights1, n_bitwise_weights2))

    def _get_wiip(self, weights):
        denom = self.attrs['noffset'] + utils.popcount(*weights)
        mask = denom == 0
        denom[mask] = 1
        toret = self.attrs['nrealizations']/denom
        toret[mask] = self.attrs['default_value']
        return toret

    def _get_wpip(self, weights1, weights2):
        denom = self.weight_attrs['noffset'] + sum(utils.bincount(w1 & w2) for w1, w2 in zip(weights1, weights2))
        mask = denom == 0
        denom[mask] = 1
        toret = self.attrs['nrealizations'] / denom
        toret[mask] = self.attrs['default_value']
        return toret

    @property
    def with_mpi(self):
        """Whether to use MPI."""
        return self.mpicomm.size > 1

    def _mpi_decompose(self):
        positions1, positions2 = self.positions1, self.positions2
        weights1, weights2 = self.weights1, self.weights2
        if self.limit_type == 'angular': # we decompose on the unit sphere: normalize positions, and put original positions in weights for decomposition
            positions1 = self.positions1/utils.distance(self.positions1.T)
            weights1 = [self.positions1] + weights1
            if not self.autorr:
                positions2 = self.positions2/utils.distance(self.positions2.T)
                weights2 = [self.positions2] + weights2
        if self.with_mpi:
            smoothing = self.limits[1]
            if self.limit_type == 'theta':
                smoothing = 2 * np.sin(0.5 * np.deg2rad(smoothing))
            from . import mpi
            (positions1, weights1), (positions2, weights2) = mpi.domain_decompose(self.mpicomm, smoothing, positions1, weights1=weights1,
                                                                                  positions2=positions2, weights2=weights2, boxsize=self.boxsize)
        elif self.autocorr:
            positions2, weights2 = positions1, weights1
        if self.limit_type == 'angular': # we remove original positions from the list of weights
            (limit_positions1, positions1, weights1) = (positions1, weights1[:1], weights1[1:])
            (limit_positions2, positions2, weights2) = (positions2, weights2[:1], weights2[1:])
        else:
            (limit_positions1, positions1, weights1) = (positions1, positions1, weights1)
            (limit_positions2, positions2, weights2) = (positions2, positions2, weights2)


class KDTreeDirectPowerEngine(BaseDirectPowerEngine):

    def run(self):
        # FIXME: We may run out-of-memory when too many pairs...
        from scipy import spatial
        rank = self.mpicomm.rank
        start = time.time()
        ells = sorted(set(self.ells))
        (dlimit_positions1, dpositions1, dweights1), (dlimit_positions2, dpositions2, dweights2) = self._mpi_decompose()
        kwargs = {'leafsize': 16, 'compact_nodes': True, 'copy_data': False, 'balanced_tree': True}
        for name in kwargs:
            if name in self.attrs: kwargs[name] = self.attrs[name]
        autocorr = self.autocorr and not self.with_mpi
        dlimit_positions = dlimit_positions1
        # Very unfortunately, cKDTree.query_pairs does not handle cross-correlations...
        # But I feel this could be changed super easily here:
        # https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/spatial/ckdtree/src/query_pairs.cxx#L210
        if not autocorr:
            dlimit_positions = np.concatenate([dlimit_positions1, dlimit_positions2], axis=0)

        tree = spatial.cKDTree(dlimit_positions, **kwargs, boxsize=None)
        pairs = tree.query_pairs(self.limits[1], p=2.0, eps=0, output_type='ndarray')
        if not autorr: # Let us remove restrict to the pairs 1 <-> 2 (removing 1 <-> 1 and 2 <-> 2)
            pairs = pairs[(pairs[:,0] < dlimit_positions1.shape[0]) & (pairs[:,1] > dlimit_positions1.shape[0])]
            pairs[:,1] -= dlimit_positions1.shape[0]
        del tree
        del dlimit_positions
        distance = np.sum((dpositions1[pairs[0]] - dpositions2[pairs[1]])**2, axis=-1)
        mask = (distance > 0.) & (distance >= self.limits[0]) & (distance < self.limits[1])
        distance = distance[mask]
        pairs = pairs[mask]

        dweights1, dweights2 = [w[pairs[:,0]] for w in dweights1], [w[pairs[:,1]] for w in dweights2]
        weight = 1.
        if self.n_bitwise_weights:
            weight = self._get_wpip(dweights1[:self.n_bitwise_weights], dweights2[:self.n_bitwise_weights])
        if self.weight_type == 'inverse_bitwise_minus_individual':
            weight -= self._get_wiip(dweights1) * self._get_wiip(dweights2)
        if len(dweights1) > self.n_bitwise_weights:
            weight *= dweights1[-1] * dweights2[-1] # single individual weight, at the end

        def normalize(los): return los/np.sum(los**2, axis=-1)[:,None]

        if self.los_type == 'global':
            los = self.los
        else:
            if self.los_type in ['firstpoint', 'endpoint']:
                los1 = normalize(dpositions1[pairs[0]])
                los2 = normalize(dpositions2[pairs[1]])

        stop = time.time()
        if rank == 0:
            self.log_info('Pairs computed in elapsed time {:.2f} s.'.format(stop - start))
        start = stop

        result = [0.]*len(ells)
        legendre = [special.legendre(ell) for ell in ells]

        def power_slab(distance, mu, weight):
            for ill, ell in enumerate(self.ells):
                tmp = weight * special.spherical_jn(ell, self.modes[:,None]*distance, derivative=False) * legendre[ill](mu)
                result[ill] += np.sum(tmp, axis=-1)

        # To avoid memory issues when performing distance*modes product, work by slabs
        nslabs = len(ells) * len(self.modes)
        npairs = distance.size

        for islab in range(nslabs):
            sl = slice(islab*npairs/nslabs, (islab+1)*npairs/nslabs)
            d, w = distance[sl], 1. if np.ndim(weight) == 0 else weight[sl]
            diff = dpositions1[pairs[0][sl]] - dpositions2[pairs[1][sl]]
            if self.los_type == 'global':
                mu = np.sum(diff * los, axis=-1)/d
                power_slab(d, mu, 2.*w) # 2 factor as i < j
            elif self.los_type == 'midpoint':
                los = normalize(dpositions1[pairs[0][sl]] + dpositions2[pairs[1][sl]])
                mu = np.sum(diff * los, axis=-1)/d
                power_slab(d, mu, 2.*w) # 2 factor as i < j
            else: # firstpoint, endpoint
                mu = np.sum(diff * los1, axis=-1)/d
                power_slab(d, mu, w)
                mu = - np.sum(diff * los2, axis=-1)/d
                power_slab(d, mu, w)

        for ill, ell in enumerate(ells):
            result[ill] *= (-1j)**ell * (2 * ell + 1)
            if ell % 2 == 1 and self.los_type == 'firstpoint':
                result[ill] = result[ill].conj()

        self.power = np.asarray([result[ells.index(ell)] for ell in self.ells])

        stop = time.time()
        if rank == 0:
            self.log_info('Direct power computed in elapsed time {:.2f} s.'.format(stop - start))
