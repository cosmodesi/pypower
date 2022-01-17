"""
Implementation of (approximate) window function estimation and convolution.
Typically, the window function will be estimated through :class:`CatalogFFTWindowMultipole`,
and window function matrices using :class:`PowerSpectrumOddWideAngleMatrix`,
following https://arxiv.org/abs/2106.06324.
"""

import math
from fractions import Fraction
import logging

import numpy as np
from scipy import special

from .utils import BaseClass
from .fftlog import CorrelationToPower
from .fft_power import BasePowerSpectrumStatistic, MeshFFTPower, CatalogMesh,\
                       _get_real_dtype, _make_array, _format_positions, _format_weights,\
                       get_default_nrealizations, get_inverse_probability_weight, _get_box
from .wide_angle import Projection, BaseMatrix, CorrelationFunctionOddWideAngleMatrix, PowerSpectrumOddWideAngleMatrix
from . import mpi, utils


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return np.concatenate([[x[1]-x[0]],x[2:]-x[:-2],[x[-1]-x[-2]]])/2.


class PowerSpectrumWindowMultipole(BasePowerSpectrumStatistic):

    """Power spectrum window function multipoles."""

    name = 'window'
    _attrs = BasePowerSpectrumStatistic._attrs + ['projs']

    def __init__(self, edges, modes, power_nonorm, nmodes, projs, wnorm=1., shotnoise_nonorm=0., **kwargs):
        r"""
        Initialize :class:`PowerSpectrumWindowMultipole`.

        Parameters
        ----------
        edges : tuple of ndim arrays
            Edges used to bin window function measurement.

        modes : array
            Mean "wavenumber" (:math:`k`) in each bin.

        power_nonorm : array
            Power spectrum in each bin, *without* normalization.

        nmodes : array
            Number of modes in each bin.

        projs : list
            List of :class:`Projection` instances or (multipole, wide-angle order) tuples.

        wnorm : float, default=1.
            Window function normalization.

        shotnoise_nonorm : float, default=0.
            Shot noise, *without* normalization.

        kwargs : dict
            Other arguments for :attr:`BasePowerSpectrumStatistic`.
        """
        self.projs = [Projection(proj) for proj in projs]
        super(PowerSpectrumWindowMultipole, self).__init__(edges, modes, power_nonorm, nmodes, wnorm=wnorm, shotnoise_nonorm=shotnoise_nonorm, **kwargs)
        if np.ndim(self.shotnoise_nonorm) == 0:
            self.shotnoise_nonorm = _make_array(0., len(self.power_nonorm), dtype='f8')
            for iproj, proj in enumerate(self.projs):
                if proj.ell == 0: self.shotnoise_nonorm[iproj] = shotnoise_nonorm
        self.wnorm = _make_array(wnorm, len(self.power_nonorm), dtype='f8')

    @property
    def power(self):
        """Power spectrum, normalized and with shot noise removed from monopole."""
        return (self.power_nonorm + self.power_direct_nonorm) / self.wnorm[:,None] - self.shotnoise[:,None]

    @property
    def kavg(self):
        """Mode-weighted average wavenumber = :attr:`k`."""
        return self.k

    def __call__(self, proj, k=None, complex=True, default_zero=False):
        r"""
        Return :attr:`power`, optionally performing linear interpolation over :math:`k`.

        Parameters
        ----------
        proj : tuple, Projection
            Projection, i.e. (multipole, wide-angle order) tuple.

        k : float, array, default=None
            :math:`k` where to interpolate the window function.
            Defaults to :attr:`kavg` (no interpolation performed).

        complex : bool, default=True
            Whether (``True``) to return the complex power spectrum,
            or (``False``) return its real part only if ``ell`` is even, imaginary part if ``ell`` is odd.

        default_zero : bool, default=False
            If input ``proj`` is not in :attr:`projs` (not computed), and ``default_zero`` is ``True``, return 0.
            If ``default_zero`` is ``False``, raise an :class:`IndexError`.

        Returns
        -------
        toret : array
            (Optionally interpolated) window function.
        """
        if k is None: k = self.k
        proj = Projection(proj)
        if proj not in self.projs:
            if default_zero:
                self.log_info('No window provided for projection {}, defaulting to 0.'.format(proj))
                return np.zeros_like(k)
            raise IndexError('No window provided for projection {}. If you want to ignore this error (set the corresponding window to zero), pass defaut_zero = True'.format(proj))
        tmp = self.power[self.projs.index(proj)]
        if not complex: tmp = tmp.real if proj.ell % 2 == 0 else tmp.imag
        return np.interp(k, self.k, tmp)

    @classmethod
    def from_power(cls, power, wa_order=0):
        """
        Build window function from input :class:`PowerSpectrumMultipole`.

        Parameters
        ----------
        power : PowerSpectrumMultipole
            Power spectrum measurement to convert into :class:`PowerSpectrumWindowMultipole`.

        wa_order : int, default=0
            Wide-angle order used for input power spectrum measurement.

        Returns
        -------
        window : PowerSpectrumWindowMultipole
        """
        state = power.__getstate__()
        state.pop('name', None)
        state['projs'] = [Projection(ell=ell, wa_order=wa_order) for ell in state.pop('ells')]
        return cls(**state)

    @classmethod
    def concatenate_x(cls, *others, select='nmodes'):
        """
        Concatenate input window functions, along k-coordinates.
        If several input windows have value for a given k-bin, choose the one with largest number of modes.

        Parameters
        ----------
        others : list of PowerSpectrumWindowMultipole
            List of window functions to be concatenated.

        select : string, default='nmodes'
            How to select input windows for each k (if several);
            'nmodes': select window with highest number of modes.

        Returns
        -------
        new : PowerSpectrumWindowMultipole
        """
        new = others[0].deepcopy()
        names = ['power_nonorm', 'power_direct_nonorm', 'nmodes']
        # First set common edges
        for other in others[1:]:
            mid = (other.edges[0][:-1] + other.edges[0][1:])/2.
            mask_low, mask_high = np.flatnonzero(mid < new.edges[0][0]), np.flatnonzero(mid > new.edges[0][-1])
            new.edges[0] = np.concatenate([other.edges[0][mask_low], new.edges[0], other.edges[0][mask_high + 1]], axis=0)
            for name in names:
                setattr(new, name, np.concatenate([getattr(other, name)[...,mask_low], getattr(new, name), getattr(other, name)[...,mask_high]], axis=-1))
            new.modes[0] = np.concatenate([other.modes[0][...,mask_low], new.modes[0], other.modes[0][...,mask_high]], axis=-1)

        tedges = list(zip(new.edges[0][:-1], new.edges[0][1:]))
        for other in others[1:]:
            for iother, tedge in enumerate(zip(other.edges[0][:-1], other.edges[0][1:])):
                if tedge in tedges: # search for k-bin in other
                    inew = tedges.index(tedge)
                    if other.nmodes[iother] > new.nmodes[inew]:
                        for name in names:
                            getattr(new, name)[...,inew] = getattr(other, name)[...,iother] # replace by value in window with highest number of modes
                        new.modes[0][...,inew] = other.modes[0][inew]
        return new

    @classmethod
    def concatenate_proj(cls, *others):
        """
        Concatenate input window functions, along projections.

        Parameters
        ----------
        others : list of PowerSpectrumWindowMultipole
            List of window functions to be concatenated.

        Returns
        -------
        new : PowerSpectrumWindowMultipole
        """
        new = others[0].deepcopy()
        new.projs = []
        for other in others: new.projs += other.projs
        names = ['power_nonorm', 'power_direct_nonorm', 'wnorm', 'shotnoise_nonorm']
        for name in names:
            tmp = [getattr(other, name) for other in others]
            setattr(new, name, np.concatenate(tmp, axis=0))
        return new

    def to_real(self, sep=None):
        """
        Transform current instance to configuration space.

        Parameters
        ----------
        sep : array, default=None
            Separations to compute window function at.

        Returns
        -------
        window : CorrelationFunctionWindowMultipole
        """
        return power_to_correlation_window(self, sep=sep)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = super(PowerSpectrumWindowMultipole, self).__getstate__()
        state['projs'] = [proj.__getstate__() for proj in self.projs]
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(PowerSpectrumWindowMultipole, self).__setstate__(state)
        self.projs = [Projection.from_state(state) for state in self.projs]


class CorrelationFunctionWindowMultipole(BaseClass):

    """Correlation window function multipoles."""

    _attrs = ['sep', 'corr']

    def __init__(self, sep, corr, projs):
        r"""
        Initialize :class:`CorrelationFunctionWindowMultipole`.

        Parameters
        ----------
        modes : array
            Mean separation.

        corr : array
            Mean correlation.

        projs : list
            List of :class:`Projection` instances or (multipole, wide-angle order) tuples.
        """
        self.sep = np.asarray(sep)
        self.projs = [Projection(proj) for proj in projs]
        self.corr = np.asarray(corr)

    def __call__(self, proj, sep=None, default_zero=False):
        r"""
        Return :attr:`corr`, optionally performing linear interpolation over :math:`s`.

        Parameters
        ----------
        proj : tuple, Projection
            Projection, i.e. (multipole, wide-angle order) tuple.

        sep : float, array, default=None
            :math:`s` where to interpolate the window function.
            Defaults to :attr:`sep` (no interpolation performed).

        default_zero : bool, default=False
            If input ``proj`` is not in :attr:`projs` (not computed), and ``default_zero`` is ``True``, return 0.
            If ``default_zero`` is ``False``, raise an :class:`IndexError`.

        Returns
        -------
        toret : array
            (Optionally interpolated) window function.
        """
        if sep is None:
            sep = self.sep
        if proj not in self.projs:
            if default_zero:
                self.log_info('No window provided for projection {}, defaulting to 0.'.format(proj))
                return np.zeros_like(sep)
            raise IndexError('No window provided for projection {}. If you want to ignore this error (set the corresponding window to zero), pass defaut_zero = True'.format(proj))
        return np.interp(sep, self.sep, self.corr[self.projs.index(proj)])

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        state['projs'] = [proj.__getstate__() for proj in self.projs]
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(CorrelationFunctionWindowMultipole, self).__setstate__(state)
        self.projs = [Projection.from_state(state) for state in self.projs]


def power_to_correlation_window(fourier_window, sep=None):
    """
    Compute correlation window function by taking Hankel transforms of input power spectrum window function.

    Parameters
    ----------
    fourier_window : PowerSpectrumWindowMultipole
        Power spectrum window function.

    sep : array, default=None
        Separations :math:`s` where to compute Hankel transform; defaults to inverse of ``fourier_window`` wavenumbers.

    Returns
    -------
    window : CorrelationFunctionWindowMultipole
        Correlation window function.
    """
    k = fourier_window.k
    mask = k > 0
    k = k[mask]
    if sep is None:
        sep = 1./k[::-1]
    else:
        sep = np.asarray(sep)
    window = []
    for proj in fourier_window.projs:
        wk = fourier_window(proj=proj)[mask]
        wk = wk.real if proj.ell % 2 == 0 else wk.imag
        volume = (2.*np.pi)**3 / np.prod(fourier_window.attrs['boxsize']) * fourier_window.nmodes[mask]
        kk, ss = np.meshgrid(k, sep, indexing='ij')
        ks = kk*ss
        integrand = wk[:,None] * 1. / (2.*np.pi)**3 * special.spherical_jn(proj.ell, ks)
        # prefactor is i^ell, but we provide the imaginary part of odd power spectra, so (-1)^(ell//2)
        prefactor = (-1) ** (proj.ell // 2)
        window.append(prefactor * np.sum(volume[:,None]*integrand, axis=0))

    return CorrelationFunctionWindowMultipole(sep, window, fourier_window.projs.copy())


class CatalogFFTWindowMultipole(MeshFFTPower):

    """Wrapper on :class:`MeshFFTPower` to estimate window function from input random positions and weigths."""

    def __init__(self, randoms_positions1=None, randoms_positions2=None,
                randoms_weights1=None, randoms_weights2=None,
                edges=None, projs=None, power_ref=None,
                los=None, nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=2., dtype=None,
                resampler=None, interlacing=None, position_type='xyz', weight_type='auto', weight_attrs=None,
                wnorm=None, shotnoise=None, mpiroot=None, mpicomm=mpi.COMM_WORLD):
        r"""
        Initialize :class:`CatalogFFTWindowMultipole`, i.e. estimate power spectrum window.

        Parameters
        ----------
        randoms_positions1 : list, array, default=None
            Positions in the first randoms catalog. Typically of shape (3, N) or (N, 3).

        randoms_positions2 : list, array, default=None
            Optionally (for cross-correlation), positions in the second randoms catalog. See ``randoms_positions1``.

        randoms_weights1 : array of shape (N,), default=None
            Optionally, weights in the first randoms catalog.

        randoms_weights2 : array of shape (N,), default=None
            Optionally (for cross-correlation), weights in the second randoms catalog.

        edges : tuple, array, default=None
            If ``los`` is local (``None``), :math:`k`-edges for :attr:`poles`.
            Else, one can also provide :math:`\mu-edges` (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            If ``kedges`` is ``None``, defaults to edges containing unique :math:`k` (norm) values, see :func:`find_unique_edges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :amth:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'dk' (in which case :func:`find_unique_edges` is used to find unique :math:`k` (norm) values).

        projs : list, default=None
            List of :class:`Projection` instances or (multipole, wide-angle order) tuples.
            If ``None``, and ``power_ref`` is provided, the list of projections is set
            to be able to compute window convolution of theory power spectrum multipoles of orders ``power_ref.ells``.

        power_ref : PowerSpectrumMultipole, default=None
            "Reference" power spectrum estimation, e.g. of the actual data.
            It is used to set default values for ``projs``, ``los``, ``boxsize``, ``boxcenter``, ``nmesh``,
            ``interlacing``, ``resampler`` and ``wnorm`` if those are ``None``.

        los : string, array, default=None
            If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.
            If ``None``, defaults to line-of-sight used in estimation of ``power_ref``.

        nmesh : array, int, default=None
            Mesh size, i.e. number of mesh nodes along each axis.
            If ``None``, defaults to the value used in estimation of ``power_ref``.

        boxsize : float, default=None
            Physical size of the box, defaults to maximum extent taken by all input positions, times ``boxpad``.
            If ``None``, defaults to the value used in estimation of ``power_ref``.

        boxcenter : array, float, default=None
            Box center, defaults to center of the Cartesian box enclosing all input positions.
            If ``None``, defaults to the value used in estimation of ``power_ref``.

        cellsize : array, float, default=None
            Physical size of mesh cells.
            If not ``None``, and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` as ``nmesh * cellsize``.
            If ``nmesh`` is ``None``, it is set as (the nearest integer(s) to) ``boxsize/cellsize``.

        boxpad : float, default=2.
            When ``boxsize`` is determined from input positions, take ``boxpad`` times the smallest box enclosing positions as ``boxsize``.

        dtype : string, dtype, default=None
            The data type to use for input positions and weights and the mesh.
            If ``None``, defaults to the value used in estimation of ``power_ref``.

        resampler : string, ResampleWindow, default='cic'
            Resampler used to assign particles to the mesh.
            Choices are ['ngp', 'cic', 'tcs', 'pcs'].
            If ``None``, defaults to the value used in estimation of ``power_ref``.

        interlacing : bool, int, default=2
            Whether to use interlacing to reduce aliasing when painting the particles on the mesh.
            If positive int, the interlacing order (minimum: 2).
            If ``None``, defaults to the value used in estimation of ``power_ref``.

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
            and "default_value", the default value of weights if the denominator is zero (defaulting to 0).

        wnorm : float, default=None
            Window function normalization.
            If ``None``, defaults to the value used in estimation of ``power_ref``,
            rescaled to the input random weights --- which yields a correct normalization of the window function
            for the power spectrum estimation ``power_ref``.
            If ``power_ref`` provided, use internal estimate obtained with :func:`normalization` --- which is wrong
            (the normalization :attr:`poles.wnorm` can be reset a posteriori using the above recipe).

        shotnoise : float, default=None
            Power spectrum shot noise, to use instead of internal estimate, which is 0 in case of cross-correlation
            and in case of auto-correlation is obtained by dividing :meth:`CatalogMesh.unnormalized_shotnoise by power spectrum normalization.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scatted across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=MPI.COMM_WORLD
            The MPI communicator.
        """
        rdtype = _get_real_dtype(dtype)
        mesh_names = ['nmesh', 'boxsize', 'boxcenter']
        loc = locals()
        mesh_attrs = {name: loc[name] for name in mesh_names if loc[name] is not None}
        if power_ref is not None:
            for name in mesh_names:
                mesh_attrs.setdefault(name, power_ref.attrs[name])
            if los is None:
                los_type = power_ref.attrs['los_type']
                los = power_ref.attrs['los']
                if los_type != 'global': los = los_type
            if interlacing is None:
                interlacing = tuple(power_ref.attrs['interlacing{:d}'.format(i+1)] for i in range(2))
            if resampler is None:
                resampler = tuple(power_ref.attrs['resampler{:d}'.format(i+1)] for i in range(2))
            if projs is None:
                ellmax = max(power_ref.ells)
                with_odd = int(any(ell % 2 for ell in power_ref.ells))
                projs = [(ell, 0) for ell in range(0, 2*ellmax + 1, 2 - with_odd)]
                if los is None or isinstance(los, str) and los in ['firstpoint', 'endpoint']:
                    projs += [(ell, 1) for ell in range(1 - with_odd, 2*ellmax + 1, 2 - with_odd)]

        if projs is None:
            raise ValueError('If no reference power spectrum "power_ref" provided, provide list of projections "projs".')
        projs = [Projection(proj) for proj in projs]
        ells_for_wa_order = {proj.wa_order:[] for proj in projs}
        for proj in projs:
            ells_for_wa_order[proj.wa_order].append(proj.ell)

        if cellsize is not None: # if cellsize is provided, remove default nmesh or boxsize value from old_matrix instance.
            mesh_attrs['cellsize'] = cellsize
            if nmesh is None: mesh_attrs.pop('nmesh')
            elif boxsize is None: mesh_attrs.pop('boxsize')

        bpositions, positions = [], {}
        for name in ['randoms_positions1', 'randoms_positions2']:
            tmp = _format_positions(locals()[name], position_type=position_type, dtype=rdtype, mpicomm=mpicomm, mpiroot=mpiroot)
            if tmp is not None: bpositions.append(tmp)
            label = name.replace('randoms_positions','R')
            positions[label] = tmp

        weight_attrs = (weight_attrs or {}).copy()
        noffset = weight_attrs.get('noffset', 1)
        default_value = weight_attrs.get('default_value', 0)
        weight_attrs.update(noffset=noffset, default_value=default_value)

        def get_nrealizations(weights):
            nrealizations = weight_attrs.get('nrealizations', None)
            if nrealizations is None: nrealizations = get_default_nrealizations(weights)
            return nrealizations

        weights = {}
        for name in ['randoms_weights1', 'randoms_weights2']:
            label = name.replace('data_weights','D').replace('randoms_weights','R').replace('shifted_weights','S')
            weight, n_bitwise_weights = _format_weights(locals()[name], weight_type=weight_type, dtype=rdtype, mpicomm=mpicomm, mpiroot=mpiroot)

            if n_bitwise_weights:
                bitwise_weight = weight[:n_bitwise_weights]
                nrealizations = get_nrealizations(bitwise_weight)
                weights[label] = get_inverse_probability_weight(bitwise_weight, noffset=noffset, nrealizations=nrealizations, default_value=default_value)
                if len(weight) > n_bitwise_weights:
                    weights[label] *= weight[n_bitwise_weights] # individual weights
            elif len(weight):
                weights[label] = weight[0] # individual weights
            else:
                weights[label] = None

        autocorr = positions['R2'] is None

        # Get box encompassing all catalogs
        nmesh, boxsize, boxcenter = _get_box(**mesh_attrs, positions=bpositions, boxpad=boxpad, mpicomm=mpicomm)
        if not isinstance(resampler, tuple):
            resampler = (resampler,)*2
        if not isinstance(interlacing, tuple):
            interlacing = (interlacing,)*2

        if wnorm is None and power_ref is not None:
            wsum = [mpicomm.allreduce(sum(weights['R1']))]*2
            if not autocorr: wsum[1] = mpicomm.allreduce(sum(weights['R2']))
            ialpha2 = np.prod([wsum[ii]/power_ref.attrs[name] for ii, name in enumerate(['sum_data_weights1', 'sum_data_weights2'])])
            wnorm = ialpha2 * power_ref.wnorm

        def get_mesh(data_positions, data_weights=None, **kwargs):
            return CatalogMesh(data_positions, data_weights=data_weights,
                               nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter,
                               position_type='pos', dtype=dtype, mpicomm=mpicomm, **kwargs)

        # Get catalog meshes
        mesh1 = get_mesh(positions['R1'], data_weights=weights['R1'], resampler=resampler[0], interlacing=interlacing[0])

        poles = []
        for wa_order, ells in ells_for_wa_order.items():
            mesh2 = None
            if wa_order == 0:
                if not autocorr:
                    mesh2 = get_mesh(positions['R2'], data_weights=weights['R2'], resampler=resampler[1], interlacing=interlacing[1])
            else:
                label = 'R1' if autocorr else 'R2'
                weights2 = np.ones_like(positions[label], shape=len(positions[label])) if weights[label] is None else weights[label]
                d = utils.distance(positions[label].T)
                mesh2 = get_mesh(positions[label], data_weights=weights2/d**wa_order, resampler=resampler[1], interlacing=interlacing[1])
            # Now, run power spectrum estimation
            super(CatalogFFTWindowMultipole, self).__init__(mesh1=mesh1, mesh2=mesh2, edges=edges, ells=ells, los=los, wnorm=wnorm, shotnoise=shotnoise)
            poles.append(PowerSpectrumWindowMultipole.from_power(self.poles, wa_order=wa_order))

        self.poles = PowerSpectrumWindowMultipole.concatenate_proj(*poles)
        del self.ells

    @classmethod
    def concatenate(cls, *others, axis=0):
        new = others[0].copy()
        new.poles = others[0].concatenate(*[other.poles for other in others], axis=axis)
        return new


class CorrelationFunctionWindowMultipoleMatrix(BaseMatrix):
    """
    Class computing matrix for window product in configuration space.

    Attributes
    ----------
    projmatrix : array
        Array of shape ``(len(self.projsout), len(self.projsin), len(self.x))``.
    """
    def __init__(self, sep, projsin, projsout=None, window=None, sum_wa=True, default_zero=False, attrs=None):
        """
        Initialize :class:`CorrelationFunctionWindowMultipoleMatrix`.

        Parameters
        ----------
        sep : array
            Input (and ouput) separations.

        projsin : list
            Input projections.

        projsout : list, default=None
            Output projections. Defaults to ``propose_out(projsin, sum_wa=sum_wa)``.

        window : CorrelationFunctionWindowMultipole, PowerSpectrumWindowMultipole
            Window function to convolve power spectrum with.
            If a :class:`PowerSpectrumWindowMultipole` instance is provided, it is transformed to configuration space.

        sum_wa : bool, default=True
            Whether to perform summation over output wide-angle orders.
            Always set to ``True`` except for debugging purposes.

        default_zero : bool, default=False
            If a given projection is not provided in window function, set to 0.
            Else an :class:`IndexError` is raised.

        attrs : dict, default=None
            Dictionary of other attributes.
        """
        self.window = window
        self.attrs = {}
        if hasattr(window, 'attrs'): self.attrs.update(window.attrs)
        self.sum_wa = sum_wa
        self.default_zero = default_zero

        self.projsin = [Projection(proj) for proj in projsin]
        if projsout is None:
            self.projsout = self.propose_out(projsin, sum_wa=self.sum_wa)
        else:
            self.projsout = [Projection(proj, default_wa_order=None if self.sum_wa else 0) for proj in projsout]

        self._set_xw(xin=sep, xout=sep)
        self.attrs = attrs or {}
        self.setup()

    def setup(self):
        r"""
        Set up transform, i.e. compute matrix:

        .. math::

            W_{\ell,\ell^{\prime}}^{(n,n^{\prime})}(s) = \delta_{n n^{\prime}} \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}^{(n)}(s)

        with :math:`\ell` multipole order corresponding to ``projout.ell`` and :math:`\ell^{\prime}` to ``projin.ell``,
        :math:`n` wide angle order corresponding to ``projout.wa_order`` and :math:`n^{\prime}` to ``projin.wa_order``.
        If :attr:`sum_wa` is ``True``, or output ``projout.wa_order`` is ``None``, sum over :math:`n` (always the case except for debugging purposes).
        For example, see q. D5 and D6 of arXiv:1810.05051.
        """
        ellsin, ellsout = [proj.ell for proj in self.projsin], [proj.ell for proj in self.projsout]
        window = self.window
        sep = self.xin[0]
        if isinstance(window, PowerSpectrumWindowMultipole):
            window = window.to_real(sep=sep)

        self.projvalue = []
        for projin in self.projsin:
            line = []
            for projout in self.projsout:
                block = np.zeros_like(sep)
                if not self.sum_wa and (projin.wa_order is None or projout.wa_order is None):
                    raise ValueError('Input and output projections should both have wide-angle order wa_order specified')
                sum_wa = self.sum_wa and projout.wa_order is None
                if sum_wa or projout.wa_order == projin.wa_order:
                    ellsw, coeffs = wigner3j_square(projout.ell, projin.ell)
                    # sum over L = ell, coeff is C_{\ell \ell^{\prime} L}, window is Q_{L}
                    for ell, coeff in zip(ellsw, coeffs):
                        proj = projin.clone(ell=ell)
                        block += coeff*window(proj, sep, default_zero=self.default_zero)
                line.append(block)
            self.projvalue.append(line)
        self.projvalue = np.array(self.projvalue) # (in, out)

    @property
    def value(self):
        if getattr(self, '_value', None) is None:
            self._value = np.bmat([[np.diag(tmp) for tmp in line] for line in self.projvalue]).A
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @staticmethod
    def propose_out(projsin, sum_wa=True):
        """
        Propose output projections given proposed input projections ``projsin``.
        If ``sum_wa`` is ``True`` (typically always the case), return projections with :attr:`Projection.wa_order` set to ``None``
        (all wide-angle orders have been summed).
        """
        projsout = [Projection(proj) for proj in projsin]
        if sum_wa:
            ellsout = np.unique([proj.ell for proj in projsout])
            projsout = [Projection(ell=ell, wa_order=None) for ell in ellsout]
        return projsout

    def resum_input_odd_wide_angle(self, **kwargs):
        """
        Resum odd wide-angle orders. By default, line-of-sight is chosen as that save in :attr:`attrs` (``attrs['los_type']``).
        To override, use input ``kwargs`` which will be passed to :attr:`CorrelationFunctionOddWideAngleMatrix`.
        """
        projsin = [proj for proj in self.projsin if proj.wa_order == 0.]
        if projsin == self.projsin: return
        from .wide_angle import CorrelationFunctionOddWideAngleMatrix
        if 'los' not in kwargs and 'los_type' in self.attrs: kwargs['los'] = self.attrs['los_type']
        matrix = CorrelationFunctionOddWideAngleMatrix([0.], projsin, projsout=self.projsin, **kwargs).value
        self.prod_proj(matrix, axes=('in', -1), projs=projsin)


class PowerSpectrumWindowMultipoleMatrix(BaseMatrix):

    """Class computing matrix for window convolution in Fourier space."""

    _slab_npoints_max = 100 * 1000

    def __init__(self, kout, projsin, projsout=None, k=None, kin_rebin=1, kin_lim=(1e-4, 1.), sep=None, window=None, xy=1, q=0, sum_wa=True, default_zero=False, attrs=None):
        """
        Initialize :class:`PowerSpectrumWindowMultipoleMatrix`.

        Parameters
        ----------
        kout : array
            Output wavenumbers.

        projsin : list
            Input projections.

        projsout : list, default=None
            Output projections. Defaults to ``propose_out(projsin, sum_wa=sum_wa)``.

        k : array, default=None
            Wavenumber for Hankel transforms; must be log-spaced.
            If ``None``, use ``sep`` and ``xy`` instead to determine :attr:`k`.

        kin_rebin : tuple, default=1
            To save some memory, rebin along input k-coordinates by this factor.

        kin_lim : tuple, default=(1e-4, 1.)
            To save some memory, pre-cut input k-coordinates to these limits.

        sep : array, default=None
            Separations for Hankel transforms; must be log-spaced.
            If ``None``, use ``k`` and ``xy`` instead to determine :attr:`sep`.

        window : CorrelationFunctionWindowMultipole, PowerSpectrumWindowMultipole
            Window function to convolve power spectrum with.
            If a :class:`PowerSpectrumWindowMultipole` instance is provided, it is transformed to configuration space.

        xy : float, default=1
            If one of ``k`` or ``sep`` is ``None``, set it following e.g. ``xy/sep[::-1]``.

        q : int, default=0
            Power-law tilt to regularize Hankel transforms.

        sum_wa : bool, default=True
            Whether to perform summation over output wide-angle orders.
            Always set to ``True`` except for debugging purposes.

        default_zero : bool, default=False
            If a given projection is not provided in window function, set to 0.
            Else an :class:`IndexError` is raised.

        attrs : dict, default=None
            Dictionary of other attributes.
        """
        self.xy = 1.

        if k is None:
            if sep is None: raise ValueError('k or sep must be provided')
            self.sep = np.asarray(sep)
            self.k = xy / self.sep[::-1]
        elif sep is None:
            self.k = np.asarray(k)
            self.sep = xy/self.kin[::-1]
        else:
            self.sep = np.asarray(sep)
            self.k = np.asarray(k)
            xy = self.k * self.sep[::-1]
            self.xy = xy[0]
            if not np.allclose(self.xy, xy): raise ValueError('kin and sep must be related by kin * sep[::-1] = cste')

        self.kin_rebin = kin_rebin
        if len(self.k) % self.kin_rebin:
            raise ValueError('Rebinning factor kin_rebin must divide len(k) or len(sep)')
        self.kin_lim = kin_lim
        self.q = q

        self.window = window
        self.attrs = {}
        if hasattr(window, 'attrs'): self.attrs.update(window.attrs)
        self.sum_wa = sum_wa
        self.default_zero = default_zero

        self.projsin = [Projection(proj) for proj in projsin]
        if projsout is None:
            self.projsout = self.propose_out(projsin)
        else:
            self.projsout = [Projection(proj, default_wa_order=None if self.sum_wa else 0) for proj in projsout]

        self._set_xw(xin=self.k, xout=kout)
        self.attrs = attrs or {}
        self.setup()

    def setup(self):
        r"""
        Set up transform. Provided arXiv:2106.06324 eq. 2.5:

        .. math::

            W_{\ell\ell^{\prime}}^{(n)}(k) = \frac{2}{\pi} (-i)^{\ell} i^{\ell^{\prime}} \int ds s^{2} j_{\ell}(ks) j_{\ell^{\prime}}(k^{\prime}s)
            \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}^{(n)}(s)

        with :math:`\ell` corresponding to ``projout.ell`` and :math:`\ell^{\prime}` to ``projin.ell``, :math:`k` to ``kout`` and :math:`k^{\prime}` to ``kin``.
        :math:`n` is the wide-angle order ``proj.wa_order``.
        Yet, to avoid bothering with complex values, we only work with the imaginary part of odd power spectra.
        In addition, we include the :math:`dk^{\prime} k^{\prime 2}` volume element (arXiv:2106.06324 eq. 2.7). Hence we actually implement:

        .. math::

            W_{\ell,\ell^{\prime}}^{(n)}(k) = dk^{\prime} k^{\prime 2} \frac{2}{\pi} (-1)^{\ell} (-1)^{\ell^{\prime}} \int ds s^{2} j_{\ell}(ks) j_{\ell^{\prime}}(k^{\prime}s)
            \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}^{(n)}(s)

        Note that we do not include :math:`k^{-n}` as this factor is included in :class:`PowerSpectrumOddWideAngleMatrix`.
        """
        self.corrmatrix = CorrelationFunctionWindowMultipoleMatrix(self.sep, self.projsin, projsout=self.projsout, window=self.window, sum_wa=self.sum_wa, default_zero=self.default_zero).projvalue

        self.value = []

        krebin = utils.rebin(self.k, len(self.k)//self.kin_rebin, statistic=np.mean)
        maskin = np.ones(len(krebin), dtype='?')
        if self.kin_lim is not None:
            maskin &= (krebin >= self.kin_lim[0]) & (krebin <= self.kin_lim[-1])
        nin = maskin.sum()

        for iin, projin in enumerate(self.projsin):
            self.xin[iin] = krebin[maskin]
            line = []
            for iout, projout in enumerate(self.projsout):
                nout = len(self.xout[iout])
                block = np.zeros((nin, nout), dtype=self.corrmatrix.dtype)
                nslabs = min(max(len(self.sep) * nout // self._slab_npoints_max, 1), nout)
                for islab in range(nslabs): # proceed by slab to save memory (if nin << len(self.sep))
                    slout = slice(islab*nout//nslabs, (islab+1)*nout//nslabs)
                    xout = self.xout[iout][slout]
                    # tmp is j_{\ell}(ks) \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}(s)
                    tmp = special.spherical_jn(projout.ell, xout[:,None]*self.sep) * self.corrmatrix[iin, iout] # matrix has dimensions (kout,s)
                    #from hankl import P2xi, xi2P
                    fftlog = CorrelationToPower(self.sep, ell=projin.ell, q=self.q, xy=self.xy, lowring=False) # prefactor is 4 \pi (-i)^{\ell^{\prime}}
                    # tmp is 4 \pi (-i)^{\ell^{\prime}} \int ds s^{2} j_{\ell}(ks) j_{\ell^{\prime}}(k^{\prime}s) \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}(s)
                    xin, tmp = fftlog(tmp) # matrix has dimensions (kout, k)
                    assert np.allclose(xin, self.k)
                    prefactor = 1./(2.*np.pi**2) * (-1j)**projout.ell * (-1)**projin.ell # now prefactor 2 / \pi (-i)^{\ell} i^{\ell^{\prime}}
                    if projout.ell % 2 == 1: prefactor *= -1j # we provide the imaginary part of odd power spectra, so let's multiply by (-i)^{\ell}
                    if projin.ell % 2 == 1: prefactor *= 1j # we take in the imaginary part of odd power spectra, so let's multiply by i^{\ell^{\prime}}
                    # tmp is dk^{\prime} k^{\prime 2} \frac{2}{\pi} (-1)^{\ell} (-1)^{\ell^{\prime}} \int ds s^{2} j_{\ell}(ks) j_{\ell^{\prime}}(k^{\prime}s) \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}(s)
                    tmp = np.real(prefactor * tmp) * weights_trapz(xin**3) / 3. # everything should be real now
                    block[:,slout] = utils.rebin(tmp.T, (len(krebin), len(xout)), statistic=np.sum)[maskin,:] # matrix has dimensions (k, kout)
                line.append(block)
            self.value.append(line)
        self.value = np.bmat(self.value).A # (in, out)

    propose_out = CorrelationFunctionWindowMultipoleMatrix.propose_out

    def resum_input_odd_wide_angle(self, **kwargs):
        """
        Resum odd wide-angle orders. By default, line-of-sight is chosen as that save in :attr:`attrs` (``attrs['los_type']``).
        To override, use input ``kwargs`` which will be passed to :attr:`PowerSpectrumOddWideAngleMatrix`.
        """
        projsin = [proj for proj in self.projsin if proj.wa_order == 0.]
        if projsin == self.projsin: return
        from .wide_angle import CorrelationFunctionOddWideAngleMatrix
        if 'los' not in kwargs and 'los_type' in self.attrs: kwargs['los'] = self.attrs['los_type']
        matrix = PowerSpectrumOddWideAngleMatrix(self.xin[0], projsin=projsin, projsout=self.projsin, **kwargs)
        self.__dict__.update(self.join(matrix, self).__dict__)


def wigner3j_square(ellout, ellin, prefactor=True):
    r"""
    Return the coefficients corresponding to the product of two Legendre polynomials, corresponding to :math:`C_{\ell \ell^{\prime} L}`
    of e.g. arXiv:2106.06324 eq. 2.2, with :math:`\ell` corresponding to ``projout.ell`` and :math:`\ell^{\prime}` to ``projin.ell``.

    Parameters
    ----------
    ellout : int
        Output order.

    ellin : int
        Input order.

    prefactor : bool, default=True
        Whether to include prefactor :math:`(2 \ell + 1)/(2 \ell^{\prime} + 1)` for window convolution.

    Returns
    -------
    ells : list
        List of mulipole orders.

    coeffs : list
        List of corresponding window coefficients.
    """
    qvals, coeffs = [], []

    def G(p):
        """
        Return the function G(p), as defined in Wilson et al 2015.
        See also: WA Al-Salam 1953
        Taken from https://github.com/nickhand/pyRSD.

        Parameters
        ----------
        p : int
            Multipole order.

        Returns
        -------
        numer, denom: int
            The numerator and denominator.
        """
        toret = 1
        for p in range(1, p+1): toret *= (2*p - 1)
        return toret, math.factorial(p)

    for p in range(min(ellin,ellout)+1):

        numer, denom = [], []

        # numerator of product of G(x)
        for r in [G(ellout-p), G(p), G(ellin-p)]:
            numer.append(r[0])
            denom.append(r[1])

        # divide by this
        a,b = G(ellin+ellout-p)
        numer.append(b)
        denom.append(a)

        numer.append(2*(ellin+ellout) - 4*p + 1)
        denom.append(2*(ellin+ellout) - 2*p + 1)

        q = ellin + ellout - 2*p
        if prefactor:
            numer.append(2*ellout + 1)
            denom.append(2*q + 1)

        numer = Fraction(np.prod(numer))
        denom = Fraction(np.prod(denom))
        coeffs.append(numer*1./denom)
        qvals.append(q)

    return qvals[::-1], coeffs[::-1]
