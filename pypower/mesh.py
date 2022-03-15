"""Implementation of methods to paint a catalog on mesh; workhorse is :class:`CatalogMesh`."""

import numpy as np
from mpi4py import MPI

from pmesh.pm import RealField, ComplexField, ParticleMesh
from pmesh.window import FindResampler, ResampleWindow
from .utils import BaseClass
from .direct_power import _make_array, _format_positions, _format_weights
from . import mpi, utils


def _get_real_dtype(dtype):
    # Return real-dtype equivalent
    return np.empty(0, dtype=dtype).real.dtype


def _get_resampler(resampler):
    # Return :class:`ResampleWindow` from string or :class:`ResampleWindow` instance
    if isinstance(resampler, ResampleWindow):
        return resampler
    conversions = {'ngp':'nnb', 'cic':'cic', 'tsc':'tsc', 'pcs':'pcs'}
    if resampler not in conversions:
        raise ValueError('Unknown resampler {}, choices are {}'.format(resampler, list(conversions.keys())))
    resampler = conversions[resampler]
    return FindResampler(resampler)


def _get_resampler_name(resampler):
    # Translate input :class:`ResampleWindow` instance to string
    conversions = {'nearest':'ngp', 'tunednnb':'ngp', 'tunedcic':'cic', 'tunedtsc':'tsc', 'tunedpcs':'pcs'}
    return conversions[resampler.kind]


def _get_compensation_window(resampler='cic', shotnoise=False):
    """
    Return the compensation function, which corrects for the particle-mesh assignment (resampler) kernel.

    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/source/mesh/catalog.py,
    following https://arxiv.org/abs/astro-ph/0409240.
    ("shotnoise" formula for pcs has been checked with WolframAlpha).

    Parameters
    ----------
    resampler : string, default='cic'
        Resampler used to assign particles to the mesh.
        Choices are ['ngp', 'cic', 'tcs', 'pcs'].

    shotnoise : bool, default=False
        If ``False``, return expression for eq. 18 in https://arxiv.org/abs/astro-ph/0409240.
        This the correct choice when applying interlacing, as aliased images (:math:`\mathbf{n} \neq (0,0,0)`) are suppressed in eq. 17.
        If ``True``, return expression for eq. 19.

    Returns
    -------
    window : callable
        Window function, taking as input :math:`\pi k_{i} / k_{N} = k / c`
        where :math:`k_{N}` is the Nyquist wavenumber and :math:`c` is the cell size,
        for each :math:`x`, :math:`y`, :math:`z`, axis.
    """
    resampler = resampler.lower()

    if shotnoise:

        if resampler == 'ngp':

            def window(*x):
                return 1.

        if resampler == 'cic':

            def window(*x):
                toret = 1.
                for xi in x:
                    toret = toret * (1 - 2. / 3 * np.sin(0.5 * xi) ** 2) ** 0.5
                return toret

        if resampler == 'tsc':

            def window(*x):
                toret = 1.
                for xi in x:
                    s = np.sin(0.5 * xi)**2
                    toret = toret * (1 - s + 2./15 * s**2) ** 0.5
                return toret

        if resampler == 'pcs':

            def window(*x):
                toret = 1.
                for xi in x:
                    s = np.sin(0.5 * xi)**2
                    toret = toret * (1 - 4./3. * s + 2./5. * s**2 - 4./315. * s**3) ** 0.5
                return toret

    else:
        p = {'ngp':1,'cic':2,'tsc':3,'pcs':4}[resampler]

        def window(*x):
            toret = 1.
            for xi in x:
                toret = toret * np.sinc(0.5 / np.pi * xi) ** p
            return toret

    return window


def _wrap_in_place(array, boxsize, offset=0.):
    array[:] = (array - offset) % boxsize + offset


def _get_box(nmesh=None, boxsize=None, boxcenter=None, cellsize=None, positions=None, boxpad=2., check=True, mpicomm=mpi.COMM_WORLD):
    """
    Compute enclosing box.

    Parameters
    ----------
    nmesh : array, int, default=None
        Mesh size, i.e. number of mesh nodes along each axis.
        If not provided, see ``value``.

    boxsize : float, default=None
        Physical size of the box.
        If not provided, see ``positions``.

    boxcenter : array, float, default=None
        Box center.
        If not provided, see ``positions``.

    cellsize : array, float, default=None
        Physical size of mesh cells.
        If not ``None``, and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` as ``nmesh * cellsize``.
        If ``nmesh`` is ``None``, it is set as (the nearest integer(s) to) ``boxsize/cellsize``.

    positions : (list of) (N, 3) arrays, default=None
        If ``boxsize`` and / or ``boxcenter`` is ``None``, use this (list of) position arrays
        to determine ``boxsize`` and / or ``boxcenter``.

    boxpad : float, default=2.
        When ``boxsize`` is determined from ``positions``, take ``boxpad`` times the smallest box enclosing ``positions`` as ``boxsize``.

    check : bool, default=True
        If ``True``, and input ``positions`` (if provided) are not contained in the box, raise a :class:`ValueError`.

    mpicomm : MPI communicator, default=MPI.COMM_WORLD
        The MPI communicator.

    Returns
    -------
    nmesh : array of shape (3,)
        Mesh size, i.e. number of mesh nodes along each axis.

    boxsize : array
        Physical size of the box.

    boxcenter : array
        Box center.
    """
    if boxsize is None or boxcenter is None or check:
        if positions is None:
            raise ValueError('positions must be provided if boxsize and boxcenter are not specified, or check is True')
        if not isinstance(positions, (tuple, list)):
            positions = [positions]
        # Find bounding coordinates
        pos_min, pos_max = np.min([pos.min(axis=0) for pos in positions],axis=0), np.max([pos.max(axis=0) for pos in positions], axis=0)
        pos_min, pos_max = np.min(mpicomm.allgather(pos_min), axis=0), np.max(mpicomm.allgather(pos_max), axis=0)
        delta = np.abs(pos_max - pos_min)
        if boxcenter is None: boxcenter = 0.5 * (pos_min + pos_max)
        if boxsize is None:
            if cellsize is not None and nmesh is not None:
                boxsize = nmesh * cellsize
            else:
                boxsize = delta.max() * boxpad
        if check and (boxsize < delta).any():
            raise ValueError('boxsize {} too small to contain all data (max {})'.format(boxsize, delta))

    if nmesh is None:
        if cellsize is not None:
            nmesh = np.rint(boxsize/cellsize).astype(int)
        else:
            raise ValueError('nmesh (or cellsize) must be specified')
    nmesh = _make_array(nmesh, 3, dtype='i4')
    boxsize = _make_array(boxsize, 3, dtype='f8')
    boxcenter = _make_array(boxcenter, 3, dtype='f8')
    return nmesh, boxsize, boxcenter


def ArrayMesh(array, boxsize, mpiroot=0, mpicomm=MPI.COMM_WORLD):
    """
    Turn numpy array into :class:`pmesh.pm.RealField`.

    Parameters
    ----------
    array : array
        Mesh numpy array gathered on ``mpiroot``.

    boxsize : array
        Physical box size.

    mpiroot : int, default=0
        MPI rank where input array is gathered.

    mpicomm : MPI communicator, default=MPI.COMM_WORLD
        The MPI communicator.

    Returns
    -------
    mesh : pmesh.pm.RealField
    """
    if mpicomm.rank == mpiroot:
        dtype, shape = array.dtype, array.shape
    else:
        dtype, shape, array = None, None, None

    dtype = mpicomm.bcast(dtype, root=mpiroot)
    shape = mpicomm.bcast(shape, root=mpiroot)
    boxsize = _make_array(boxsize, 3, dtype='f8')
    pm = ParticleMesh(BoxSize=boxsize, Nmesh=shape, dtype=dtype, comm=mpicomm)
    mesh = pm.create(type='real')
    if mpicomm.rank == mpiroot:
        array = array.ravel() # ignore data from other ranks
    else:
        array = np.empty((0,), dtype)
    mesh.unravel(array)
    return mesh


class CatalogMesh(BaseClass):

    """Class to paint catalog of positions and weights to mesh."""

    _slab_npoints_max = int(1024 * 1024 * 4)

    def __init__(self, data_positions, data_weights=None, randoms_positions=None, randoms_weights=None,
                 shifted_positions=None, shifted_weights=None,
                 nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=2., wrap=False, dtype='f8',
                 resampler='tsc', interlacing=2, position_type='xyz', copy=False, mpiroot=None, mpicomm=MPI.COMM_WORLD):
        """
        Initialize :class:`CatalogMesh`.

        Note
        ----
        When running with MPI, input positions and weights are assumed to be scatted on all MPI ranks of ``mpicomm``.
        If this is not the case, use :func:`mpi.scatter_array`.

        Parameters
        ----------
        data_positions : list, array
            Positions in the data catalog. Typically of shape (3, N) or (N, 3).

        data_weights : array of shape (N,), default=None
            Optionally, data weights.

        randoms_positions : list, array
            Positions in the randoms catalog. Typically of shape (3, N) or (N, 3).

        randoms_weights : array of shape (N,), default=None
            Randoms weights.

        shifted_positions : array, default=None
            Optionally, in case of BAO reconstruction, positions of the shifted catalog.

        shifted_weights : array, default=None
            Optionally, in case of BAO reconstruction, weigths of the shifted catalog.

        nmesh : array, int, default=None
            Mesh size, i.e. number of mesh nodes along each axis.
            If not provided, see ``value``.

        boxsize : float, default=None
            Physical size of the box.
            If not provided, see ``positions``.

        boxcenter : array, float, default=None
            Box center.
            If not provided, see ``positions``.

        cellsize : array, float, default=None
            Physical size of mesh cells.
            If not ``None``, and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` as ``nmesh * cellsize``.
            If ``nmesh`` is ``None``, it is set as (the nearest integer(s) to) ``boxsize/cellsize``.

        wrap : bool, default=False
            Whether to wrap input positions?
            If ``False`` and input positions do not fit in the the box size, raise a :class:`ValueError`.

        boxpad : float, default=2.
            When ``boxsize`` is determined from ``positions``, take ``boxpad`` times the smallest box enclosing ``positions`` as ``boxsize``.

        dtype : string, dtype, default='f8'
            The data type to use for the mesh.
            Input ``positions`` and ``weights`` are cast to the corresponding (real) precision.

        resampler : string, ResampleWindow, default='tsc'
            Resampler used to assign particles to the mesh.
            Choices are ['ngp', 'cic', 'tcs', 'pcs'].

        interlacing : bool, int, default=2
            Whether to use interlacing to reduce aliasing when painting the particles on the mesh.
            If positive int, the interlacing order (minimum: 2).

        position_type : string, default='xyz'
            Type of input positions, one of:

                - "pos": Cartesian positions of shape (N, 3)
                - "xyz": Cartesian positions of shape (3, N)
                - "rdd": RA/Dec in degree, distance of shape (3, N)

        copy : bool, default=False
            If ``False``, avoids copy of positions and weights if they are of (real) type ``dtype``, ``mpiroot`` is ``None``,
            and ``position_type`` is "pos" (for positions).
            Setting to ``True`` is only useful if one wants to modify positions or weights that have been passed as input
            while keeping those attached to the current mesh instance the same.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scatted across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=MPI.COMM_WORLD
            The MPI communicator.
        """
        self.mpicomm = mpicomm
        self.dtype = np.dtype(dtype)
        self.rdtype = _get_real_dtype(self.dtype)
        self._set_positions(data_positions=data_positions, randoms_positions=randoms_positions, shifted_positions=shifted_positions, position_type=position_type, copy=copy, mpiroot=mpiroot)
        self._set_weights(data_weights=data_weights, randoms_weights=randoms_weights, shifted_weights=shifted_weights, copy=copy, mpiroot=mpiroot)
        self._set_box(boxsize=boxsize, cellsize=cellsize, nmesh=nmesh, boxcenter=boxcenter, boxpad=boxpad, wrap=wrap)
        self._set_resampler(resampler)
        self._set_interlacing(interlacing)

    def __repr__(self):
        """String representation of current mesh."""
        info = ['{}={}'.format(name, getattr(self,name)) for name in ['nmesh', 'boxsize', 'boxcenter', 'dtype']]
        return '{}({})'.format(self.__class__.__name__,', '.join(info))

    @property
    def compensation(self):
        """Return dictionary specifying compensation scheme for particle-mesh resampling."""
        return {'resampler':_get_resampler_name(self.resampler), 'shotnoise': not bool(self.interlacing)}

    def clone(self, data_positions=None, data_weights=None, randoms_positions=None, randoms_weights=None,
              shifted_positions=None, shifted_weights=None,
              boxsize=None, cellsize=None, nmesh=None, boxcenter=None, dtype=None,
              resampler=None, interlacing=None, position_type='xyz', mpicomm=None):
        """
        Clone current instance, i.e. copy and set new positions and weights.
        Arguments 'boxsize', 'nmesh', 'boxcenter', 'dtype', 'resampler', 'interlacing', 'mpicomm', if ``None``,
        are overriden by those of the current instance.
        """
        new = self.__class__.__new__(self.__class__)
        kwargs = {}
        loc = locals()
        for name in ['boxsize', 'nmesh', 'boxcenter', 'dtype', 'resampler', 'interlacing', 'mpicomm']:
            kwargs[name] = loc[name] if loc[name] is not None else getattr(self, name)
        if cellsize is not None: # if cellsize is provided, remove default nmesh or boxsize value from current instance.
            kwargs['cellsize'] = cellsize
            if nmesh is None: kwargs.pop('nmesh')
            elif boxsize is None: kwargs.pop('boxsize')
        new.__init__(data_positions=data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights,
                     shifted_positions=shifted_positions, shifted_weights=shifted_weights, position_type=position_type, **kwargs)
        return new

    def _set_interlacing(self, interlacing):
        self.interlacing = int(interlacing)
        if self.interlacing != interlacing:
            raise ValueError('Interlacing must be either bool (False, 0) or an integer >= 2')
        if self.interlacing == 1:
            if self.mpicomm.rank == 0:
                self.log_warning('Provided interlacing is {}; setting it to 2.'.format(interlacing))
            self.interlacing = 2

    def _set_box(self, nmesh=None, boxsize=None, cellsize=None, boxcenter=None, boxpad=2., wrap=False):
        # Set :attr:`nmesh`, :attr:`boxsize` and :attr:`boxcenter`
        positions = [self.data_positions]
        if self.with_randoms: positions += [self.randoms_positions]
        if self.with_shifted: positions += [self.shifted_positions]
        self.nmesh, self.boxsize, self.boxcenter = _get_box(nmesh=nmesh, boxsize=boxsize, cellsize=cellsize, boxcenter=boxcenter,
                                                            positions=positions, boxpad=boxpad, check=not wrap, mpicomm=self.mpicomm)
        if wrap:
            for position in positions:
                _wrap_in_place(position, self.boxsize, self.boxcenter - self.boxsize/2.)

    def _set_positions(self, data_positions, randoms_positions=None, shifted_positions=None, position_type='xyz', copy=False, mpiroot=None):
        # Set data and optionally shifted and randoms positions, scattering on all ranks if not already
        if position_type is not None: position_type = position_type.lower()
        self.position_type = position_type

        self.data_positions = _format_positions(data_positions, position_type=self.position_type, dtype=self.rdtype, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)
        if self.data_positions is None:
            raise ValueError('Provide at least an array of data positions')
        self.randoms_positions = _format_positions(randoms_positions, position_type=self.position_type, dtype=self.rdtype, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.shifted_positions = _format_positions(shifted_positions, position_type=self.position_type, dtype=self.rdtype, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.data_size = len(self.data_positions)
        self.shifted_size, self.randoms_size = 0, 0
        if self.with_shifted:
            self.shifted_size = len(self.shifted_positions)
        if self.with_randoms:
            self.randoms_size = len(self.randoms_positions)

    def _set_weights(self, data_weights, randoms_weights=None, shifted_weights=None, copy=False, mpiroot=None):
        # Set data and optionally shifted and randoms weights and their sum, scattering on all ranks if not already

        def get_weights(weights):
            weights = _format_weights(weights, weight_type='product_individual', dtype=self.rdtype, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)[0]
            if weights:
                return weights[0]
            return None

        self.data_weights = get_weights(data_weights)
        self.randoms_weights = get_weights(randoms_weights)
        self.shifted_weights = get_weights(shifted_weights)

        if not self.with_randoms and self.randoms_weights is not None:
            raise ValueError('randoms_weights are provided, but not randoms_positions')

        if not self.with_shifted and self.shifted_weights is not None:
            raise ValueError('shifted_weights are provided, but not shifted_positions')

        def sum_weights(positions, weights):
            if weights is None:
                return self.mpicomm.allreduce(len(positions))
            return self.mpicomm.allreduce(sum(weights))

        self.sum_data_weights = sum_weights(self.data_positions, self.data_weights)
        self.sum_randoms_weights = self.sum_shifted_weights = 0.
        if self.with_shifted:
            self.sum_shifted_weights = sum_weights(self.shifted_positions, self.shifted_weights)
        if self.with_randoms:
            self.sum_randoms_weights = sum_weights(self.randoms_positions, self.randoms_weights)

    @property
    def with_randoms(self):
        """Whether randoms positions have been provided."""
        return self.randoms_positions is not None

    @property
    def with_shifted(self):
        """Whether "shifted" positions have been provided (e.g. for reconstruction)."""
        return self.shifted_positions is not None

    def _set_resampler(self, resampler='cic'):
        # Set :attr:`resampler`
        self.resampler = _get_resampler(resampler=resampler)

    def to_mesh(self, field=None, dtype=None, compensate=False):
        """
        Paint positions/weights to mesh.

        Parameters
        ----------
        field : string, default=None
            Field to paint to mesh, one of:

                - "data": data positions and weights
                - "shifted": shifted positions and weights (available only if shifted positions are provided)
                - "randoms": randoms positions and weights
                - "data-normalized_shifted": shifted positions and weights, renormalized (by alpha)
                   such that their sum is same as data weights
                - "data-normalized_randoms": randoms positions and weights, renormalized (by alpha)
                   such that their sum is same as data weights
                - "fkp": FKP field, i.e. data - alpha * (shifted if provided else randoms)
                - ``None``: defaults to "data" if no shifted/randoms, else "fkp"

        dtype : string, dtype, default='f8'
            The data type of the mesh when painting, to override current :attr:`dtype`.

        compensate : bool, default=False
            Wether to apply compensation for particle-mesh assignment scheme.

        Returns
        -------
        out : RealField
            Mesh, with values in "weights" units (not *normalized* as density).
        """
        if dtype is None: dtype = self.dtype

        if field is None:
            field = 'fkp' if (self.with_randoms or self.with_shifted) else 'data'
        field = field.lower()
        allowed_fields = set(['data', 'normalized_data'])
        if self.with_shifted: allowed_fields |= set(['shifted', 'data-normalized_shifted', 'fkp'])
        if self.with_randoms: allowed_fields |= set(['randoms', 'data-normalized_randoms', 'fkp'])
        if field not in list(set(allowed_fields)):
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, allowed_fields))
        positions, weights = [], []
        if field in ['data', 'fkp']:
            positions += [self.data_positions]
            weights += [(self.data_weights, None)]
        if field in ['normalized_data']:
            positions += [self.data_positions]
            weights += [(self.data_weights, self.nmesh.prod()/self.sum_data_weights)] # mean mesh is 1
        if field in ['fkp']:
            if self.with_shifted:
                positions += [self.shifted_positions]
                weights += [(self.shifted_weights, -self.sum_data_weights/self.sum_shifted_weights)]
            else:
                positions += [self.randoms_positions]
                weights += [(self.randoms_weights, -self.sum_data_weights/self.sum_randoms_weights)]
        if field in ['shifted', 'data-normalized_shifted']:
            positions += [self.shifted_positions]
            if field == 'data-normalized_shifted':
                weights += [(self.shifted_weights, self.sum_data_weights/self.sum_shifted_weights)]
            else:
                weights += [(self.shifted_weights, None)]
        if field in ['randoms', 'data-normalized_randoms']:
            positions += [self.randoms_positions]
            if field == 'data-normalized_randoms':
                weights += [(self.randoms_weights, self.sum_data_weights/self.sum_randoms_weights)]
            else:
                weights += [(self.randoms_weights, None)]

        pm = ParticleMesh(BoxSize=self.boxsize, Nmesh=self.nmesh, dtype=dtype, comm=self.mpicomm)
        offset = self.boxcenter - self.boxsize/2.
        #offset = self.boxcenter
        #offset = 0.

        def paint(positions, weights, scaling, out, transform=None):
            positions = positions - offset
            factor = bool(self.interlacing) + 0.5
            scalar_weights = weights is None

            if scaling is not None:
                if scalar_weights: weights = scaling
                else: weights = weights * scaling

            # We work by slab to limit memory footprint
            # Merely copy-pasted from https://github.com/bccp/nbodykit/blob/4aec168f176939be43f5f751c90363b39ec6cf3a/nbodykit/source/mesh/catalog.py#L300
            def paint_slab(sl):
                # Decompose positions such that they live in the same region as the mesh in the current process
                p = positions[sl]
                size = len(p)
                layout = pm.decompose(p, smoothing=factor * self.resampler.support)
                # If we are receiving too many particles, abort and retry with a smaller chunksize
                recvlengths = pm.comm.allgather(layout.recvlength)
                if any(recvlength > 2 * self._slab_npoints_max for recvlength in recvlengths):
                    if pm.comm.rank == 0:
                        self.log_info('Throttling slab size as some ranks will receive too many particles. ({:d} > {:d})'.format(max(recvlengths), self._slab_npoints_max * 2))
                    raise StopIteration
                p = layout.exchange(p)
                #w = layout.exchange(weights[sl])
                w = weights if scalar_weights else layout.exchange(weights[sl])
                # hold = True means no zeroing of out
                pm.paint(p, mass=w, resampler=self.resampler, transform=transform, hold=True, out=out)
                return size

            islab = 0
            slab_npoints = self._slab_npoints_max
            sizes = pm.comm.allgather(len(positions))
            csize = sum(sizes)
            local_size_max = max(sizes)
            painted_size = 0

            import gc
            while islab < local_size_max:

                sl = slice(islab, islab + slab_npoints)

                if pm.comm.rank == 0:
                    self.log_info('Slab {:d} ~ {:d} / {:d}.'.format(islab, islab + slab_npoints, local_size_max))
                try:
                    painted_size_slab = paint_slab(sl)
                except StopIteration:
                    slab_npoints = slab_npoints // 2
                    if slab_npoints < 1:
                        raise RuntimeError('Cannot find a slab size that fits into memory.')
                    continue
                finally:
                    # collect unfreed items
                    gc.collect()

                painted_size += pm.comm.allreduce(painted_size_slab)

                if pm.comm.rank == 0:
                    self.log_info('Painted {:d} out of {:d} objects to mesh.'.format(painted_size, csize))

                islab += slab_npoints
                slab_npoints = min(self._slab_npoints_max, int(slab_npoints * 1.2))

        out = pm.create(type='real', value=0.)
        for p, w in zip(positions, weights): paint(p, *w, out)

        if self.interlacing:
            if self.mpicomm.rank == 0:
                self.log_info('Running interlacing at order {:d}.'.format(self.interlacing))
            cellsize = self.boxsize/self.nmesh
            shifts = np.arange(self.interlacing)*1./self.interlacing
            # remove 0 shift, already computed
            shifts = shifts[1:]
            out = out.r2c()
            for shift in shifts:
                transform = pm.affine.shift(shift) # this shifts particle positions by ``shift`` before painting to mesh
                # paint to two shifted meshes
                mesh_shifted = pm.create(type='real', value=0.)
                for p, w in zip(positions, weights): paint(p, *w, mesh_shifted, transform=transform)
                mesh_shifted = mesh_shifted.r2c()
                for k, s1, s2 in zip(out.slabs.x, out.slabs, mesh_shifted.slabs):
                    kc = sum(k[i] * cellsize[i] for i in range(3))
                    # pmesh convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r)
                    # shifting by "shift * cellsize" we compute F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r - shift * cellsize)
                    # i.e. F(k) = e^{- i shift * kc} 1/N^3 e^{-ikr} F(r)
                    # Hence compensation below
                    s1[...] = s1[...] + s2[...] * np.exp(shift * 1j * kc)
            if compensate:
                self._compensate(out)
            out = out.c2r()
            out[:] /= self.interlacing
        elif compensate:
            out = out.r2c()
            self._compensate(out)
            out = out.c2r()
        return out

    def _compensate(self, cfield):
        if self.mpicomm.rank == 0:
            self.log_info('Applying compensation {}.'.format(self.compensation))
        # Apply compensation window for particle-assignment scheme
        window = _get_compensation_window(**compensation)

        cellsize = self.boxsize/self.nmesh
        for k, slab in zip(cfield.slabs.x, cfield.slabs):
            kc = tuple(ki * ci for ki, ci in zip(k, cellsize))
            slab[...] /= window(*kc)

    def unnormalized_shotnoise(self):
        """
        Return unnormalized shotnoise, as:

        .. math::

            \sum_{i=1}^{N_{g}} w_{i,g}^{2} + \alpha^{2} \sum_{i=1}^{N_{r}} w_{i,r}^{2}

        Where the sum runs over data (and optionally) shifted/randoms weights.
        """
        def sum_weights2(positions, weights=None):
            if weights is None:
                return self.mpicomm.allreduce(len(positions))
            return self.mpicomm.allreduce(sum(weights**2))

        shotnoise = sum_weights2(self.data_positions, self.data_weights)
        if self.with_shifted:
            alpha = self.sum_data_weights/self.sum_shifted_weights
            shotnoise += alpha**2 * sum_weights2(self.shifted_positions, self.shifted_weights)
        elif self.with_randoms:
            alpha = self.sum_data_weights/self.sum_randoms_weights
            shotnoise += alpha**2 * sum_weights2(self.randoms_positions, self.randoms_weights)
        return shotnoise
