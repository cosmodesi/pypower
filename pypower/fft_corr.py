r"""Implementation of correlation function estimator."""


import os
import time

import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy import special

from .utils import BaseClass
from . import mpi, utils
from .mesh import CatalogMesh, _get_real_dtype, _get_mesh_attrs, _wrap_positions
from .fft_power import MeshFFTBase, get_real_Ylm, _transform_rslab, _format_positions, _format_all_weights, project_to_basis, _nan_to_zero, find_unique_edges, unnormalized_shotnoise


class BaseCorrelationFunctionStatistics(BaseClass):
    """
    Base template correlation function statistic class.
    Specific correlation function statistic should extend this class.

    We recommend accessing correlation function measurements through :meth:`get_corr`,
    or :meth:`__call__` (accessed through ``my_corr_statistic_instance()``).
    """
    name = 'base'
    _attrs = ['name', 'edges', 'modes', 'corr_nonorm', 'corr_zero_nonorm', 'corr_direct_nonorm', 'nmodes', 'wnorm', 'shotnoise_nonorm', 'attrs']
    _tosum = ['nmodes']
    _toaverage = ['modes', 'corr_nonorm', 'corr_zero_nonorm', 'corr_direct_nonorm']
    _coords_names = ['s']
    _corr_names = ['xi(s)']

    def __init__(self, edges, modes, corr_nonorm, nmodes, wnorm=1., shotnoise_nonorm=0., corr_zero_nonorm=None, corr_direct_nonorm=None, attrs=None, mpicomm=None):
        r"""
        Initialize :class:`BaseCorrelationFunctionStatistics`.

        Parameters
        ----------
        edges : tuple of ndim arrays
            Edges used to bin correlation function measurement.

        modes : array
            Mean "wavevector" (e.g. :math:`(s, \mu)`) in each bin.

        corr_nonorm : array
            Correlation function in each bin, *without* normalization.

        nmodes : array
            Number of modes in each bin.

        wnorm : float, default=1.
            Correlation function normalization.

        shotnoise_nonorm : float, default=0.
            Shot noise, *without* normalization.

        corr_zero_nonorm : float, array, default=0.
            Value of correlation function at :math:`s = 0`.

        corr_direct_nonorm : array, default=0.
            Value of pair-count-based 'direct' correlation function estimation,
            (e.g. PIP and angular upweights correction, eq. 26 of https://arxiv.org/abs/1912.08803),
            to be added to :attr:`corr_nonorm`.

        attrs : dict, default=None
            Dictionary of other attributes.

        mpicomm : MPI communicator, default=None
            The MPI communicator, only used when saving (:meth:`save` and :meth:`save_txt`) statistics.
        """
        if np.ndim(edges[0]) == 0: edges = (edges,)
        if np.ndim(modes[0]) == 0: modes = (modes,)
        self.edges = list(np.asarray(edge) for edge in edges)
        self.modes = list(np.asarray(mode) for mode in modes)
        self.corr_nonorm = np.asarray(corr_nonorm)
        self.corr_zero_nonorm = corr_zero_nonorm
        if corr_zero_nonorm is None:
            self.corr_zero_nonorm = np.zeros_like(self.corr_nonorm)
        else:
            self.corr_zero_nonorm = np.asarray(corr_zero_nonorm)
        self.corr_direct_nonorm = corr_direct_nonorm
        if corr_direct_nonorm is None:
            self.corr_direct_nonorm = np.zeros_like(self.corr_nonorm)
        else:
            self.corr_direct_nonorm = np.asarray(corr_direct_nonorm)
        self.nmodes = np.asarray(nmodes)
        self.wnorm = wnorm
        self.shotnoise_nonorm = shotnoise_nonorm
        self.mpicomm = mpicomm
        self.attrs = attrs or {}

    def get_corr(self, add_direct=True, remove_shotnoise=True, null_zero_mode=True, divide_wnorm=True, complex=True):
        """
        Return correlation function, computed using various options.

        Parameters
        ----------
        add_direct : bool, default=True
            Add pair-count-based 'direct' correlation function measurement.

        remove_shotnoise : bool, default=True
            Remove estimated shot noise at :math:`s = 0`  (if within :attr:`edges`).

        null_zero_mode : bool, default=True
            Remove power spectrum at :math:`k = 0`.

        divide_wnorm : bool, default=True
            Divide by estimated correlation function normalization.

        complex : bool, default=True
            Whether (``True``) to return the complex correlation function,
            or (``False``) return its real part only.

        Results
        -------
        corr : array
        """
        toret = self.corr_nonorm.copy()
        if add_direct:
            toret += self.corr_direct_nonorm
        if remove_shotnoise:
            dig_zero = tuple(np.digitize(0., edges, right=False) - 1 for edges in self.edges)
            if all(0 <= dig_zero[ii] < self.shape[ii] for ii in range(self.ndim)):
                with np.errstate(divide='ignore', invalid='ignore'):
                    toret[(Ellipsis,) * (toret.ndim - self.ndim) + dig_zero] -= self.shotnoise_nonorm / self.nmodes[dig_zero]
        if null_zero_mode:
            toret -= self.corr_zero_nonorm
        if divide_wnorm:
            toret /= self.wnorm
        if not complex and np.iscomplexobj(toret):
            toret = toret.real
        return toret

    @property
    def corr(self):
        """Correlation function, normalized and with shot noise removed."""
        return self.get_corr()

    @property
    def shotnoise(self):
        """Normalized shot noise."""
        return self.shotnoise_nonorm / self.wnorm

    @property
    def s(self):
        """Separations."""
        return self.modes[0]

    @property
    def sedges(self):
        """Separation edges."""
        return self.edges[0]

    @property
    def shape(self):
        """Return shape of binned correlation function :attr:`corr`."""
        return tuple(len(edges) - 1 for edges in self.edges)

    @property
    def ndim(self):
        """Return binning dimensionality."""
        return len(self.edges)

    def modeavg(self, axis=0, method=None):
        r"""
        Return average of modes for input axis.

        Parameters
        ----------
        axis : int, default=0
            Axis.

        method : str, default=None
            If ``None``, return average separation from :attr:`modes`.
            If 'mid', return bin mid-points.

        Returns
        -------
        modeavg : array
            1D array of size :attr:`shape[axis]`.
        """
        axis = axis % self.ndim
        if method is None:
            axes_to_sum_over = tuple(ii for ii in range(self.ndim) if ii != axis)
            with np.errstate(divide='ignore', invalid='ignore'):
                toret = np.sum(_nan_to_zero(self.modes[axis]) * self.nmodes, axis=axes_to_sum_over) / np.sum(self.nmodes, axis=axes_to_sum_over)
        elif isinstance(method, str):
            allowed_methods = ['mid']
            method = method.lower()
            if method not in allowed_methods:
                raise ValueError('method should be one of {}'.format(allowed_methods))
            elif method == 'mid':
                toret = (self.edges[axis][:-1] + self.edges[axis][1:]) / 2.
        return toret

    sepavg = modeavg

    def __call__(self):
        """Method that interpolates correlation function measurement at any point."""
        raise NotImplementedError('Implement method "__call__" in your {}'.format(self.__class__.__name__))

    def __getitem__(self, slices):
        """Call :meth:`slice`."""
        new = self.copy()
        if isinstance(slices, tuple):
            new.slice(*slices)
        else:
            new.slice(slices)
        return new

    def select(self, *xlims):
        """
        Restrict statistic to provided coordinate limits in place.

        For example:

        .. code-block:: python

            statistic.select((0, 30))  # restrict first axis to (0, 30)
            statistic.select(None, (0, 20))  # restrict second axis to (0, 20)
            statistic.select((0, 30, 4))   # rebin to match step size of 4 and restrict to (0, 30)

        """
        if len(xlims) > self.ndim:
            raise IndexError('Too many limits: statistics is {:d}-dimensional, but {:d} were indexed'.format(self.ndim, len(xlims)))
        slices = []
        for iaxis, xlim in enumerate(xlims):
            if xlim is None:
                slices.append(slice(None))
            elif len(xlim) == 3:
                factor = int(xlim[2] / np.diff(self.edges[iaxis]).mean() + 0.5)
                if not np.allclose(np.diff(self.edges[iaxis][::factor]), xlim[2]):
                    import warnings
                    with np.printoptions(threshold=40):
                        warnings.warn('Unable to match step {} with edges {}'.format(xlim[2], self.edges[iaxis]))
                slices.append(slice(0, (self.shape[iaxis] // factor) * factor, factor))
            elif len(xlim) != 2:
                raise ValueError('Input limits must be a tuple (min, max) or (min, max, step)')
        self.slice(*slices)
        slices = []
        for iaxis, xlim in enumerate(xlims):
            if xlim is None:
                slices.append(slice(None))
            else:
                x = self.modeavg(axis=iaxis, method='mid')
                indices = np.flatnonzero((x >= xlim[0]) & (x <= xlim[1]))
                if indices.size:
                    slices.append(slice(indices[0], indices[-1] + 1, 1))
                else:
                    slices.append(slice(0))
        self.slice(*slices)

    def slice(self, *slices):
        """
        Slice statistics in place. If slice step is not 1, use :meth:`rebin`.
        For example:

        .. code-block:: python

            statistic.slice(slice(0, 10, 2), slice(0, 6, 3)) # rebin by factor 2 (resp. 3) along axis 0 (resp. 1), up to index 10 (resp. 6)
            statistic[:10:2, :6:3] # same as above, but return new instance.

        """
        inslices = list(slices) + [slice(None)] * (self.ndim - len(slices))
        if len(inslices) > self.ndim:
            raise IndexError('Too many indices: statistics is {:d}-dimensional, but {:d} were indexed'.format(self.ndim, len(slices)))
        slices, eslices, factor = [], [], []
        for iaxis, sl in enumerate(inslices):
            start, stop, step = sl.indices(self.nmodes.shape[iaxis])
            if step < 0:
                raise IndexError('Positive slicing step only supported')
            slices.append(slice(start, stop, 1))
            eslices.append(slice(start, stop + 1, 1))
            factor.append(step)
        slices = tuple(slices)
        for name in self._tosum + self._toaverage:
            array = getattr(self, name, None)
            if array is None: continue
            if isinstance(array, list):
                setattr(self, name, [arr[(Ellipsis,) * (arr.ndim - self.ndim) + slices] for arr in array])
            else:
                setattr(self, name, array[(Ellipsis,) * (array.ndim - self.ndim) + slices])
        self.edges = [edges[eslice] for edges, eslice in zip(self.edges, eslices)]
        if not all(f == 1 for f in factor):
            self.rebin(factor=factor)

    def rebin(self, factor=1):
        """
        Rebin correlation function estimation in place, by factor(s) ``factor``.
        Input factors must divide :attr:`shape`.
        """
        if np.ndim(factor) == 0:
            factor = (factor,)
        factor = list(factor) + [1] * (self.ndim - len(factor))
        if len(factor) > self.ndim:
            raise ValueError('Too many rebinning factors: statistics is {:d}-dimensional, but got {:d} factors'.format(self.ndim, len(factor)))
        if any(s % f for s, f in zip(self.shape, factor)):
            raise ValueError('Rebinning factor must divide shape')
        new_shape = tuple(s // f for s, f in zip(self.shape, factor))
        nmodes = self.nmodes
        for name in self._tosum:
            array = getattr(self, name, None)
            if array is None: continue
            if isinstance(array, list):
                array = [utils.rebin(arr, new_shape, statistic=np.sum) for arr in array]
            else:
                array = utils.rebin(array, new_shape, statistic=np.sum)
            setattr(self, name, array)
        for name in self._toaverage:
            array = getattr(self, name, None)
            if array is None: continue
            if isinstance(array, list):
                with np.errstate(divide='ignore', invalid='ignore'):
                    array = [utils.rebin(_nan_to_zero(arr) * nmodes, new_shape, statistic=np.sum) / self.nmodes for arr in array]
            else:
                extradim = array.ndim > self.ndim
                with np.errstate(divide='ignore', invalid='ignore'):
                    array = np.asarray([utils.rebin(_nan_to_zero(arr) * nmodes, new_shape, statistic=np.sum) / self.nmodes for arr in array.reshape((-1,) + self.shape)])
                    array.shape = (-1,) * extradim + new_shape
            setattr(self, name, array)
        self.edges = [edges[::f] for edges, f in zip(self.edges, factor)]

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        name = {'wedge': 'wedges'}.get(self.name, 'poles')
        if name in state:  # actually a MeshFFTPower object, which has wedges or poles
            state = state[name]
        super(BaseCorrelationFunctionStatistics, self).__setstate__(state)
        if self.corr_zero_nonorm.ndim < self.corr_nonorm.ndim:  # for backward-compatibility; to be removed soon!
            self.corr_zero_nonorm = np.zeros_like(self.corr_nonorm)

    def __copy__(self):
        new = super(BaseCorrelationFunctionStatistics, self).__copy__()
        for name in ['edges', 'modes', 'attrs']:
            setattr(new, name, getattr(new, name).copy())
        return new

    def deepcopy(self):
        import copy
        new = copy.deepcopy(self)
        if hasattr(self, 'mpicomm'):
            new.mpicomm = self.mpicomm
        return new

    @classmethod
    def average(cls, *others, weights=None):
        """
        Average input correlation functions.

        Warning
        -------
        Input orrelation function have same edges / number of modes for this operation to make sense
        (no checks performed).
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = others[0].deepcopy()
        if weights is None:
            weights = [1. for other in others]
        if len(weights) != len(others):
            raise ValueError('Provide as many weights as instances to average')
        weights = [np.array(weight) for weight in weights]
        for name in cls._attrs:
            if name.endswith('nonorm') and hasattr(new, name):
                value = 0.
                for other, weight in zip(others, weights):
                    tmp = getattr(other, name)
                    tmp = np.asarray(tmp)
                    weight = weight[(Ellipsis,) * weight.ndim + (None,) * (tmp.ndim - weight.ndim)]
                    value += tmp * weight
                setattr(new, name, value)
            if 'wnorm' in name and hasattr(new, name):
                value = sum(getattr(other, 'wnorm') * weight for other, weight in zip(others, weights))
                setattr(new, name, value)
        return new

    @classmethod
    def sum(cls, *others):
        """
        Sum input correlation function, weighted by their :attr:`wnorm`.

        Warning
        -------
        Input orrelation function have same edges / number of modes for this operation to make sense
        (no checks performed).
        """
        return cls.average(others)

    def __add__(self, other):
        return self.sum(self, other)

    def __radd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)

    def save_txt(self, filename, fmt='%.12e', delimiter=' ', header=None, comments='# ', **kwargs):
        """
        Save correlation function as txt file.

        Warning
        -------
        Attributes are not all saved, hence there is :meth:`load_txt` method.

        Parameters
        ----------
        filename : str
            File name.

        fmt : str, default='%.12e'
            Format for floating types.

        delimiter : str, default=' '
            String or character separating columns.

        header : str, list, default=None
            String that will be written at the beginning of the file.
            If multiple lines, provide a list of one-line strings.

        comments : str, default=' #'
            String that will be prepended to the header string.

        kwargs : dict
            Arguments for :meth:`get_corr`.
        """
        if not self.with_mpi or self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            formatter = {'int_sind': lambda x: '%d' % x, 'float_sind': lambda x: fmt % x}

            def complex_sind(x):
                imag = fmt % x.imag
                if imag[0] not in ['+', '-']: imag = '+' + imag
                return '{}{}j'.format(fmt % x.real, imag)

            formatter['complex_sind'] = complex_sind
            if header is None: header = []
            elif isinstance(header, str): header = [header]
            else: header = list(header)
            for name in ['autocorr', 'data_size1', 'data_size2', 'sum_data_weights1', 'sum_data_weights2',
                         'randoms_size1', 'randoms_size2', 'sum_randoms_weights1', 'sum_randoms_weights2',
                         'shifted_size1', 'shifted_size2', 'sum_shifted_weights1', 'sum_shifted_weights2',
                         'los_type', 'los', 'nmesh', 'boxsize', 'boxcenter', 'resampler1', 'resampler2',
                         'interlacing1', 'interlacing2', 'shotnoise', 'wnorm']:
                value = self.attrs.get(name, getattr(self, name, None))
                if value is None:
                    value = 'None'
                elif any(name.startswith(key) for key in ['los_type', 'resampler']):
                    value = str(value)
                else:
                    value = np.array2string(np.array(value), separator=delimiter, formatter=formatter).replace('\n', '')
                header.append('{} = {}'.format(name, value))
                # value = value.split('\n')
                # header.append('{} = {}'.format(name, value[0]))
                # for arr in value[1:]: header.append(' '*(len(name) + 3) + arr)
            labels = ['nmodes']
            assert len(self._coords_names) == self.ndim
            for name in self._coords_names:
                labels += ['{}mid'.format(name), '{}avg'.format(name)]
            labels += self._corr_names
            corr = self.get_corr(**kwargs)
            columns = [self.nmodes.flat]
            mids = np.meshgrid(*(self.modeavg(idim, method='mid') for idim in range(self.ndim)), indexing='ij')
            for idim in range(self.ndim):
                columns += [mids[idim].flat, self.modes[idim].flat]
            for column in corr.reshape((-1,) * (corr.ndim == self.ndim) + corr.shape):
                columns += [column.flat]
            columns = [[np.array2string(value, formatter=formatter) for value in column] for column in columns]
            widths = [max(max(map(len, column)) - len(comments) * (icol == 0), len(label)) for icol, (column, label) in enumerate(zip(columns, labels))]
            widths[-1] = 0  # no need to leave a space
            header.append((' ' * len(delimiter)).join(['{:<{width}}'.format(label, width=width) for label, width in zip(labels, widths)]))
            widths[0] += len(comments)
            with open(filename, 'w') as file:
                for line in header:
                    file.write(comments + line + '\n')
                for irow in range(len(columns[0])):
                    file.write(delimiter.join(['{:<{width}}'.format(column[irow], width=width) for column, width in zip(columns, widths)]) + '\n')

        # if self.with_mpi:
        #     self.mpicomm.Barrier()


def get_corr_statistic(statistic='wedge'):
    """Return :class:`BaseCorrelationFunctionStatistics` subclass corresponding to ``statistic`` (either 'wedge' or 'multipole')."""
    if statistic == 'wedge':
        return CorrelationFunctionWedges
    if statistic == 'multipole':
        return CorrelationFunctionMultipoles
    return BaseCorrelationFunctionStatistics


class MetaCorrelationFunctionStatistics(type(BaseClass)):

    """Metaclass to return correct correlation function statistic."""

    def __call__(cls, *args, statistic='wedge', **kwargs):
        return get_corr_statistic(statistic=statistic)(*args, **kwargs)


class CorrelationFunctionStatistics(BaseClass, metaclass=MetaCorrelationFunctionStatistics):

    """Entry point to correlation function statistics."""

    @classmethod
    def from_state(cls, state):
        state = state.copy()
        name = state.pop('name')
        return get_corr_statistic(statistic=name).from_state(state)


class CorrelationFunctionWedges(BaseCorrelationFunctionStatistics):

    r"""Correlation function binned in :math:`(s, \mu)`."""

    name = 'wedge'
    _coords_names = ['s', 'mu']
    _corr_names = ['xi(s, mu)']

    @property
    def savg(self):
        """Mode-weighted average wavenumber."""
        return self.modeavg(axis=0)

    @property
    def mu(self):
        """Cosine angle to line-of-sight."""
        return self.modes[1]

    @property
    def muavg(self):
        r"""Mode-weighted average :math:`\mu`."""
        return self.modeavg(axis=1)

    @property
    def muedges(self):
        r""":math:`\mu`-edges."""
        return self.edges[1]

    def __call__(self, s=None, mu=None, return_s=False, return_mu=False, complex=True, **kwargs):
        r"""
        Return correlation function, optionally performing linear interpolation over :math:`s` and :math:`\mu`.

        Parameters
        ----------
        s : float, array, default=None
            :math:`s` where to interpolate the correlation function.
            Values outside :attr:`savg` are set to the first/last correlation function value;
            outside :attr:`edges[0]` to nan.
            Defaults to :attr:`savg`.

        mu : float, array, default=None
            :math:`\mu` where to interpolate the correlation function.
            Values outside :attr:`muavg` are set to the first/last correlation function value;
            outside :attr:`edges[1]` to nan.
            Defaults to :attr:`muavg`.

        return_s : bool, default=False
            Whether (``True``) to return :math:`s`-modes (see ``s``).
            If ``None``, return :math:`s`-modes if ``s`` is ``None``.

        return_mu : bool, default=False
            Whether (``True``) to return :math:`\mu`-modes (see ``mu``).
            If ``None``, return :math:`\mu`-modes if ``mu`` is ``None``.

        complex : bool, default=True
            Whether (``True``) to return the complex correlation function,
            or (``False``) return its real part only.

        kwargs : dict
            Other arguments for :meth:`get_corr`.

        Returns
        -------
        s : array
            Optionally, :math:`s`-modes.

        mu : array
            Optionally, :math:`\mu`-modes.

        corr : array
            (Optionally interpolated) correlation function.
        """
        corr = self.get_corr(complex=complex, **kwargs)
        savg, muavg = self.savg, self.muavg
        if return_s is None:
            return_s = s is None
        if return_mu is None:
            return_mu = mu is None
        if s is None and mu is None:
            if return_s:
                if return_mu:
                    return savg, muavg, corr
                return savg, corr
            return corr
        if s is None: s = savg
        if mu is None: mu = muavg
        mask_finite_s, mask_finite_mu = ~np.isnan(savg), ~np.isnan(muavg)
        savg, muavg, corr = savg[mask_finite_s], muavg[mask_finite_mu], corr[np.ix_(mask_finite_s, mask_finite_mu)]
        s, mu = np.asarray(s), np.asarray(mu)
        toret_shape = s.shape + mu.shape
        s, mu = s.ravel(), mu.ravel()
        toret = np.nan * np.zeros((s.size, mu.size), dtype=corr.dtype)
        mask_s = (s >= self.edges[0][0]) & (s <= self.edges[0][-1])
        mask_mu = (mu >= self.edges[1][0]) & (mu <= self.edges[1][-1])
        s_masked, mu_masked = s[mask_s], mu[mask_mu]
        if s_masked.size and mu_masked.size:
            if muavg.size == 1:

                def interp(array):
                    return UnivariateSpline(savg, array, s=0, k=1, ext='const')(s_masked)[:, None]

            else:
                i_s = np.argsort(s_masked); ii_s = np.argsort(i_s)
                i_mu = np.argsort(mu_masked); ii_mu = np.argsort(i_mu)

                def interp(array):
                    return RectBivariateSpline(savg, muavg, array, kx=1, ky=1, s=0)(s_masked[i_s], mu_masked[i_mu], grid=True)[np.ix_(ii_s, ii_mu)]

            toret[np.ix_(mask_s, mask_mu)] = interp(corr.real)
            if complex and np.iscomplexobj(corr):
                toret[np.ix_(mask_s, mask_mu)] += 1j * interp(corr.imag)

        toret.shape = toret_shape
        if return_s:
            if return_mu:
                return s, mu, toret
            return s, toret
        return toret

    def plot(self, ax=None, fn=None, kw_save=None, show=False):
        r"""
        Plot correlation function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes where to plot samples. If ``None``, takes current axes.

        fn : string, default=None
            If not ``None``, file name where to save figure.

        kw_save : dict, default=None
            Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            Whether to show figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from matplotlib import pyplot as plt
        fig = None
        if ax is None: fig, ax = plt.subplots()
        s, corr = self.s, self(complex=False)
        wedges = self.edges[1]
        for iwedge, wedge in enumerate(zip(wedges[:-1], wedges[1:])):
            ax.plot(s[:, iwedge], s[:, iwedge]**2 * corr[:, iwedge], label=r'${:.2f} < \mu < {:.2f}$'.format(*wedge))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        ax.set_ylabel(r'$s^{2} \xi(s, \mu)$ [$(\mathrm{Mpc}/h)^{2}$]')
        if not self.with_mpi or self.mpicomm.rank == 0:
            if fn is not None:
                utils.savefig(fn, fig=fig, **(kw_save or {}))
            if show:
                plt.show()
        return ax


class CorrelationFunctionMultipoles(BaseCorrelationFunctionStatistics):

    """Correlation function multipoles binned in :math:`s`."""

    name = 'multipole'
    _attrs = BaseCorrelationFunctionStatistics._attrs + ['ells']

    def __init__(self, edges, modes, corr_nonorm, nmodes, ells, **kwargs):
        r"""
        Initialize :class:`CorrelationFunctionMultipoles`.

        Parameters
        ----------
        edges : tuple of ndim arrays
            Edges used to bin correlation function measurement.

        modes : array
            Mean "wavevector" (e.g. :math:`(s, \mu)`) in each bin.

        corr_nonorm : array
            Correlation function in each bin, *without* normalization.

        nmodes : array
            Number of modes in each bin.

        ells : tuple, list.
            Multipole orders.

        kwargs : dict
            Other arguments for :attr:`BaseCorrelationFunctionStatistics`.
        """
        self.ells = tuple(ells)
        super(CorrelationFunctionMultipoles, self).__init__(edges, modes, corr_nonorm, nmodes, **kwargs)

    @property
    def _corr_names(self):
        return ['xi{:d}(s)'.format(ell) for ell in self.ells]

    @property
    def savg(self):
        """Mode-weighted average wavenumber = :attr:`s`."""
        return self.s

    def to_wedges(self, muedges, ells=None):
        r"""Transform poles to wedges, with input :math:`\mu`-edges.

        Parameters
        ----------
        muedges : array
            :math:`\mu`-edges.

        ells : tuple, list, default=None
            Multipole orders to use in the Legendre expansion.
            If ``None``, all poles are used.

        Returns
        -------
        wedges : CorrelationFunctionWedges
            Correlation function wedges.
        """
        if ells is None:
            ells = self.ells
        elif np.ndim(ells) == 0:
            ells = [ells]
        muedges = np.array(muedges)
        mu = (muedges[:-1] + muedges[1:]) / 2.
        dmu = np.diff(muedges)
        edges = (self.sedges.copy(), muedges)
        modes = (np.repeat(self.modes[0][:, None], len(dmu), axis=-1), np.repeat(mu[None, :], len(self.s), axis=0))
        corr_nonorm, corr_zero_nonorm, corr_direct_nonorm = 0, 0, 0
        for ell in ells:
            poly = np.diff(special.legendre(ell).integ()(muedges)) / dmu
            ill = self.ells.index(ell)
            corr_nonorm += self.corr_nonorm[ill, ..., None] * poly
            corr_zero_nonorm += self.corr_zero_nonorm[ill, ..., None] * poly
            corr_direct_nonorm += self.corr_direct_nonorm[ill, ..., None] * poly
        nmodes = self.nmodes[:, None] / dmu
        return CorrelationFunctionWedges(edges, modes, corr_nonorm, nmodes, wnorm=self.wnorm, shotnoise_nonorm=self.shotnoise_nonorm,
                                         corr_zero_nonorm=corr_zero_nonorm, corr_direct_nonorm=corr_direct_nonorm,
                                         attrs=self.attrs, mpicomm=getattr(self, 'mpicomm', None))

    def get_corr(self, add_direct=True, remove_shotnoise=True, null_zero_mode=True, divide_wnorm=True, complex=True):
        """
        Return correlation function, computed using various options.

        Parameters
        ----------
        add_direct : bool, default=True
            Add direct correlation function measurement.

        remove_shotnoise : bool, default=True
            Remove estimated shot noise.

        remove_shotnoise : bool, default=True
            Remove estimated shot noise at :math:`s = 0`  (if within :attr:`edges`).

        null_zero_mode : bool, default=True
            Remove power spectrum at :math:`k = 0`.

        divide_wnorm : bool, default=True
            Divide by estimated correlation function normalization.

        complex : bool, default=True
            Whether (``True``) to return the complex correlation function,
            or (``False``) return its real part if even multipoles, imaginary part if odd multipole.

        Results
        -------
        corr : array
        """
        toret = super(CorrelationFunctionMultipoles, self).get_corr(add_direct=add_direct, remove_shotnoise=False, null_zero_mode=null_zero_mode, divide_wnorm=False, complex=True)
        if remove_shotnoise and 0 in self.ells:
            dig_zero = np.digitize(0., self.edges[0], right=False) - 1
            if 0 <= dig_zero < self.shape[0]:
                with np.errstate(divide='ignore', invalid='ignore'):
                    toret[self.ells.index(0), dig_zero] -= self.shotnoise_nonorm / self.nmodes[dig_zero]
        if divide_wnorm:
            toret /= self.wnorm
        if not complex and np.iscomplexobj(toret):
            toret = np.array([toret[ill].real if ell % 2 == 0 else toret[ill].imag for ill, ell in enumerate(self.ells)], dtype=toret.real.dtype)
        return toret

    def __call__(self, ell=None, s=None, return_s=False, complex=True, **kwargs):
        r"""
        Return correlation function, optionally performing linear interpolation over :math:`s`.

        Parameters
        ----------
        ell : int, list, default=None
            Multipole(s). Defaults to all multipoles.

        s : float, array, default=None
            :math:`s` where to interpolate the correlation function.
            Values outside :attr:`savg` are set to the first/last correlation function value;
            outside :attr:`edges[0]` to nan.
            Defaults to :attr:`savg` (no interpolation performed).

        return_s : bool, default=False
            Whether (``True``) to return :math:`s`-modes (see ``s``).
            If ``None``, return :math:`s`-modes if ``s`` is ``None``.

        complex : bool, default=True
            Whether (``True``) to return the complex correlation function,
            or (``False``) return its real part if ``ell`` is even, imaginary part if ``ell`` is odd.

        kwargs : dict
            Other arguments for :meth:`get_corr`.

        Returns
        -------
        s : array
            Optionally, :math:`s`-modes.

        corr : array
            (Optionally interpolated) correlation function.
        """
        if ell is None:
            isscalar = False
            ell = self.ells
        else:
            isscalar = np.ndim(ell) == 0
            if isscalar: ell = [ell]
        ells = ell
        corr = self.get_corr(complex=complex, **kwargs)
        corr = corr[[self.ells.index(ell) for ell in ells]]
        savg = self.s.copy()
        if return_s is None:
            return_s = s is None
        if s is None:
            toret = corr
            if isscalar:
                toret = corr[0]
            if return_s:
                return savg, toret
            return toret
        mask_finite_s = ~np.isnan(savg) & ~np.isnan(corr).any(axis=0)
        savg, corr = savg[mask_finite_s], corr[:, mask_finite_s]
        s = np.asarray(s)
        toret = np.nan * np.zeros((len(ells),) + s.shape, dtype=corr.dtype)
        mask_s = (s >= self.edges[0][0]) & (s <= self.edges[0][-1])
        s_masked = s[mask_s]
        if s_masked.size:

            def interp(array):
                return np.array([UnivariateSpline(savg, arr, k=1, s=0, ext='const')(s_masked) for arr in array], dtype=array.dtype)

            toret[..., mask_s] = interp(corr.real)
            if complex and np.iscomplexobj(corr):
                toret[..., mask_s] = toret[..., mask_s] + 1j * interp(corr.imag)
        if isscalar:
            toret = toret[0]
        if return_s:
            return s, toret
        return toret

    def plot(self, ax=None, fn=None, kw_save=None, show=False):
        r"""
        Plot correlation function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes where to plot samples. If ``None``, takes current axes.

        fn : string, default=None
            If not ``None``, file name where to save figure.

        kw_save : dict, default=None
            Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            Whether to show figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from matplotlib import pyplot as plt
        fig = None
        if ax is None: fig, ax = plt.subplots()
        for ill, ell in enumerate(self.ells):
            s, corr = self(ell=ell, return_s=True, complex=False)
            ax.plot(s, s**2 * corr, label=r'$\ell = {:d}$'.format(ell))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        if not self.with_mpi or self.mpicomm.rank == 0:
            if fn is not None:
                utils.savefig(fn, fig=fig, **(kw_save or {}))
            if show:
                plt.show()
        return ax

    def select(self, *xlims, ells=None):
        """
        Restrict statistic to provided coordinate limits and multipoles ``ells`` in place.

        For example:

        .. code-block:: python

            statistic.select((0, 30))  # restrict first axis to (0, 30)
            statistic.select((0, 20), ells=(0, 2))  # restrict first axis to (0, 20), and select monopole and quadrupole

        """
        if ells is not None:
            indices = [self.ells.index(ell) for ell in ells]
            self.ells = tuple(self.ells[index] for index in indices)
            names = ['corr_nonorm', 'corr_zero_nonorm', 'corr_direct_nonorm']
            for name in names:
                setattr(self, name, getattr(self, name)[indices])
        super(CorrelationFunctionMultipoles, self).select(*xlims)
        return self

    @classmethod
    def concatenate_proj(cls, *others):
        """
        Concatenate input correlation functions, along poles.

        Parameters
        ----------
        others : list of CorrelationFunctionMultipoles
            List of correlation function multipoles to be concatenated.

        Returns
        -------
        new : CorrelationFunctionMultipoles
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = others[0].deepcopy()
        new.ells = []
        arrays = {name: [] for name in ['corr_nonorm', 'corr_zero_nonorm', 'corr_direct_nonorm']}
        for other in others:
            indices = [ill for ill, ell in enumerate(other.ells) if ell not in new.ells]
            new.ells += [other.ells[index] for index in indices]
            for name, array in arrays.items():
                arrays[name].append(getattr(other, name)[indices] * new.wnorm / other.wnorm)
        new.ells = tuple(new.ells)
        for name, array in arrays.items():
            setattr(new, name, np.concatenate(array, axis=0))
        return new


class MeshFFTCorr(MeshFFTBase):
    """
    Class that computes correlation functions from input mesh(es), using global or local line-of-sight, following https://arxiv.org/abs/1704.02357.

    Attributes
    ----------
    poles : CorrelationFunctionMultipoles
        Estimated correlation function multipoles.

    wedges : CorrelationFunctionWedges
        Estimated correlation function wedges (if relevant).
    """

    def __init__(self, mesh1, mesh2=None, edges=None, ells=(0, 2, 4), los=None, boxcenter=None, compensations=None, wnorm=None, shotnoise=None, shotnoise_nonorm=None):
        r"""
        Initialize :class:`MeshFFTCorr`, i.e. estimate correlation function.

        Warning
        -------
        In case line-of-sight is not local, one can provide :math:`\mu`-edges. In this case, integration over Legendre polynomials for multipoles
        is performed between the first and last :math:`\mu`-edges.
        For example, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, integration is performed between :math:`\mu = 0.2` and :math:`\mu = 0.8`.
        In all other cases, integration is performed between :math:`\mu = -1.0` and :math:`\mu = 1.0`.

        Parameters
        ----------
        mesh1 : CatalogMesh, RealField, ComplexField
            First mesh.
            If ``RealField``, assumed to be :math:`1 + \delta` or :math:`\bar{n} (1 + \delta)`.
            In case of :class:`ComplexField`, assumed to be the FFT of :math:`\delta` (or :math:`1 + \delta`), i.e. unit density.

        mesh2 : CatalogMesh, RealField, ComplexField, default=None
            In case of cross-correlation, second mesh, with same size and physical extent (``boxsize`` and ``boxcenter``) that ``mesh1``.

        edges : tuple, array, default=None
            If ``los`` is local (``None``), :math:`s`-edges for :attr:`poles`.
            Else, one can also provide :math:`\mu`-edges (hence a tuple ``(sedges, muedges)``) for :attr:`wedges`.
            If ``sedges`` is ``None``, defaults to edges containing unique :math:`s` (norm) values, see :func:`find_unique_edges`.
            ``sedges`` may be a dictionary, with keys 'min' (minimum :math:`s`, defaults to 0), 'max' (maximum :math:`s`, defaults to ``boxsize/2``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`s` (norm) values between 'min' and 'max').
            For both :math:`s` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end, i.e. ``edges[i] <= x < edges[i+1]``.
            However, last :math:`\mu`-bin is inclusive on both ends: ``edges[-2] <= mu <= edges[-1]``.
            Therefore, with e.g. :math:`\mu`-edges ``[0.2, 0.4, 1.0]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 1.0`.
            Similarly, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 0.8`.

        ells : list, tuple, default=(0, 2, 4)
            Multipole orders.

        los : string, array, default=None
            If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        boxcenter : float, array, default=None
            Box center; defaults to 0.
            Used only if provided ``mesh1`` and ``mesh2`` are not ``CatalogMesh``.

        compensations : list, tuple, string, default=None
            Compensations to apply to mesh to (optionally) correct for particle-mesh assignment scheme;
            e.g. 'cic' (resp. 'cic-sn') for cic assignment scheme, with (resp. without) interlacing.
            In case ``mesh2`` is not ``None`` (cross-correlation), provide a list (or tuple) of two such strings
            (for ``mesh1`` and ``mesh2``, respectively).
            Used only if provided ``mesh1`` or ``mesh2`` are not ``CatalogMesh``.

        wnorm : float, default=None
            Correlation function normalization, to use instead of internal estimate obtained with :func:`normalization`.

        shotnoise : float, default=None
            Correlation function shot noise, to use instead of internal estimate, which is 0 in case of cross-correlation
            or both ``mesh1`` and ``mesh2`` are :class:`pmesh.pm.RealField`,
            and in case of auto-correlation is obtained by dividing :meth:`unnormalized_shotnoise`
            of ``mesh1`` by correlation function normalization.
        """
        t0 = time.time()
        self._set_compensations(compensations)
        self._set_los(los)
        self._set_ells(ells)
        self._set_mesh(mesh1, mesh2=mesh2, boxcenter=boxcenter)
        self._set_edges(edges)
        self._set_normalization(wnorm, mesh1=mesh1, mesh2=mesh2)
        self._set_shotnoise(shotnoise, shotnoise_nonorm=shotnoise_nonorm, mesh1=mesh1, mesh2=mesh2)
        self.attrs.update(self._get_attrs())
        t1 = time.time()
        if self.mpicomm.rank == 0:
            self.log_info('Meshes prepared in elapsed time {:.2f} s.'.format(t1 - t0))
            self.log_info('Running mesh calculation.')
        self.run()
        t2 = time.time()
        if self.mpicomm.rank == 0:
            self.log_info('Mesh calculations performed in elapsed time {:.2f} s.'.format(t2 - t1))
            self.log_info('Correlation function computed in elapsed time {:.2f} s.'.format(t2 - t0))

    def _set_edges(self, edges):
        # Set :attr:`edges`
        if edges is None or isinstance(edges, dict) or (not isinstance(edges[0], dict) and np.ndim(edges[0]) == 0):
            edges = (edges,)
        if len(edges) == 1:
            sedges, muedges = edges[0], None
        else:
            sedges, muedges = edges
        if sedges is None:
            sedges = {}
        if isinstance(sedges, dict):
            smin = sedges.get('min', 0.)
            smax = sedges.get('max', self.boxsize.max() / 2.)
            ds = sedges.get('step', None)
            if ds is None:
                # Find unique edges
                s = [s.real.astype('f8') for s in self.pm.create_coords('real')]
                ds = np.min(self.boxsize / self.nmesh)
                sedges = find_unique_edges(s, x0=ds, xmin=smin, xmax=smax + 1e-5 * ds, mpicomm=self.mpicomm)
            else:
                sedges = np.arange(smin, smax + 1e-5 * ds, ds)
        if self.mpicomm.rank == 0:
            self.log_info('Using {:d} s-bins between {:.3f} and {:.3f}.'.format(len(sedges) - 1, sedges[0], sedges[-1]))
        if muedges is None:
            muedges = np.linspace(-1., 1., 2, endpoint=True)  # single :math:`\mu`-wedge
        elif self.los_type != 'global' and muedges.size > 2:
            raise ValueError('Cannot compute wedges with local {} line-of-sight'.format(self.los_type))
        self.edges = (np.asarray(sedges, dtype='f8'), np.asarray(muedges, dtype='f8'))
        for name, edges in zip(['s', 'mu'], self.edges):
            if len(edges) < 2:
                raise ValueError('{}-edges are of size {:d} < 2'.format(name, len(edges)))

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {'attrs': self.attrs}
        for name in ['wedges', 'poles']:
            if hasattr(self, name):
                state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state."""
        super(MeshFFTCorr, self).__setstate__(state)
        for name in ['wedges', 'poles']:
            if name in state:
                setattr(self, name, get_corr_statistic(statistic=state[name].pop('name')).from_state(state[name]))

    def run(self):
        if self.los_type == 'global':  # global (fixed) line-of-sight
            self._run_global_los()
        else:  # local (varying) line-of-sight
            self._run_local_los()

    def _run_global_los(self):

        # Calculate the 3d correlation function, slab-by-slab to save memory
        # FFT 1st density field and apply the resampler transfer kernel
        cfield2 = cfield1 = self._to_complex(self.mesh1, copy=True)  # copy because will be modified in-place
        del self.mesh1
        # We will apply all compensation transfer functions to cfield1
        compensations = [self.compensations[0]] if self.autocorr else self.compensations
        self._compensate(cfield1, *compensations)

        if not self.autocorr:
            cfield2 = self._to_complex(self.mesh2, copy=False)
        del self.mesh2
        # cfield1.conj() * cfield2
        for c1, c2 in zip(cfield1.slabs, cfield2.slabs):
            c1[...] = c1.conj() * c2

        # for i, c1 in zip(cfield1.slabs.i, cfield1.slabs):
        #     mask_zero = True
        #     for ii in i: mask_zero = mask_zero & (ii == 0)
        #     # if mask_zero.any(): corr_zero = c1[mask_zero].item().real / self.nmesh.prod(dtype='f8')
        #     c1[mask_zero] = 0.

        rfield = cfield1.c2r()
        del cfield1, cfield2
        result, result_poles = project_to_basis(rfield, self.edges, ells=self.ells, los=self.los, exclude_zero=False)
        rfield[...] = rfield.cmean()
        result_zero, result_zero_poles = project_to_basis(rfield, self.edges, ells=self.ells, los=self.los, exclude_zero=False)

        kwargs = {'wnorm': self.wnorm, 'shotnoise_nonorm': self.shotnoise * self.wnorm * self.nmesh.prod(dtype='f8') / self.boxsize.prod(dtype='f8'),
                  'attrs': self.attrs, 'mpicomm': self.mpicomm}
        s, mu, corr, nmodes = result[:4]
        corr_zero = result_zero[2]
        # pmesh convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here
        corr, corr_zero = (self.nmesh.prod(dtype='f8')**2 / self.boxsize.prod(dtype='f8') * tmp for tmp in (corr, corr_zero))
        # Format the corr results into :class:`CorrelationFunctionWedges` instance
        self.wedges = CorrelationFunctionWedges(modes=(s, mu), edges=self.edges, corr_nonorm=corr, corr_zero_nonorm=corr_zero, nmodes=nmodes, **kwargs)

        if result_poles:
            # Format the corr results into :class:`CorrelationFunctionMultipoles` instance
            s, corr, nmodes = result_poles[:3]
            corr_zero = result_zero_poles[1]
            # pmesh convention is F(s) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here
            corr, corr_zero = (self.nmesh.prod(dtype='f8')**2 / self.boxsize.prod(dtype='f8') * tmp for tmp in (corr, corr_zero))
            self.poles = CorrelationFunctionMultipoles(modes=s, edges=self.edges[0], corr_nonorm=corr, corr_zero_nonorm=corr_zero, nmodes=nmodes, ells=self.ells, **kwargs)

    def _run_local_los(self):

        swap = self.los_type == 'endpoint'
        if swap: self.mesh1, self.mesh2 = self.mesh2, self.mesh1  # swap meshes + complex conjugaison at the end of run()

        rank = self.mpicomm.rank

        nonzeroells = ells = sorted(set(self.ells))
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]

        if nonzeroells:
            rfield1 = self._to_real(self.mesh1)

        # FFT 1st density field and apply the resampler transfer kernel
        A0 = self._to_complex(self.mesh2, copy=True)  # pmesh r2c convention is 1/N^3 e^{-ikr}
        # Set mean value or real field to 0
        # for i, c in zip(A0.slabs.i, A0.slabs):
        #     mask_zero = True
        #     for ii in i: mask_zero = mask_zero & (ii == 0)
        #     c[mask_zero] = 0.

        # We will apply all compensation transfer functions to A0_1 (faster than applying to each Aell)
        compensations = [self.compensations[0]] * 2 if self.autocorr else self.compensations

        result, result_zero = [], []
        # Loop over the higher order multipoles (ell > 0)

        if self.autocorr:
            if nonzeroells:
                # Higher-order multipole requested
                # If monopole requested, copy A0 without window in Aell
                if 0 in self.ells: Aell = A0.copy()
                self._compensate(A0, *compensations)
            else:
                # In case of autocorrelation, and only monopole requested, no A0_1 copy need be made
                # Apply a single window, which will be squared by the autocorrelation
                if 0 in self.ells: Aell = A0.copy()
                self._compensate(A0, compensations[0])
        else:
            # Cross-correlation, all windows on A0
            if 0 in self.ells: Aell = self._to_complex(self.mesh1, copy=True)  # mesh1 != mesh2!
            self._compensate(A0, *compensations)

        del self.mesh2, self.mesh1

        if 0 in self.ells:

            for islab in range(A0.shape[0]):
                Aell[islab, ...] = Aell[islab] * A0[islab].conj()

            # the 1D monopole
            # from nbodykit.algorithms.fftcorr import project_to_basis
            rfield = Aell.c2r()
            del Aell
            proj_result = project_to_basis(rfield, self.edges, exclude_zero=False)[0]
            result.append(np.ravel(proj_result[2]))
            result_zero.append(np.ones_like(result[-1]) * rfield.cmean())
            s, nmodes = proj_result[0], proj_result[3]

            if rank == 0:
                self.log_info('ell = {:d} done; {:d} r2c completed'.format(0, 1))

        if nonzeroells:
            # Initialize the memory holding the Aell terms for
            # higher multipoles (this holds sum of m for fixed ell)
            # NOTE: this will hold FFTs of density field #1
            from pmesh.pm import RealField, ComplexField
            rfield = RealField(self.pm)
            cfield = ComplexField(self.pm)
            Aell = RealField(self.pm)
            Aell_zero = RealField(self.pm)

            # Spherical harmonic kernels (for ell > 0)
            Ylms = [[get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in nonzeroells]

            offset = self.boxcenter - self.boxsize / 2.
            # NOTE: we do not apply half cell shift as in nbodykit below
            # offset = self.boxcenter - self.boxsize/2. + 0.5*self.boxsize / self.nmesh # in nbodykit
            # offset = self.boxcenter + 0.5*self.boxsize / self.nmesh # in nbodykit

            def _wrap_rslab(rslab):
                # We do not use the same conventions as pmesh:
                # rslab < 0 is sent back to [boxsize/2, boxsize]
                toret = []
                for ii, rr in enumerate(rslab):
                    mask = rr > self.boxsize[ii] / 2.
                    rr[mask] -= self.boxsize[ii]
                    toret.append(rr)
                return toret

            def _safe_divide(num, denom):
                with np.errstate(divide='ignore', invalid='ignore'):
                    toret = num / denom
                toret[denom == 0.] = 0.
                return toret

            # The real-space grid
            xhat = [xx.real.astype('f8') + offset[ii] for ii, xx in enumerate(_transform_rslab(rfield1.slabs.optx, self.boxsize))]
            xnorm = np.sqrt(sum(xx**2 for xx in xhat))
            xhat = [_safe_divide(xx, xnorm) for xx in xhat]
            del xnorm

            # The separation-space grid
            shat = [ss.real.astype('f8') for ss in _wrap_rslab(_transform_rslab(rfield1.slabs.optx, self.boxsize))]
            snorm = np.sqrt(sum(ss**2 for ss in shat))
            shat = [_safe_divide(ss, snorm) for ss in shat]
            del snorm

        for ill, ell in enumerate(nonzeroells):

            Aell_zero[:] = Aell[:] = 0.
            # Iterate from m=-ell to m=ell and apply Ylm
            t0 = time.time()
            for Ylm in Ylms[ill]:
                # Reset the real-space mesh to the original density #1
                rfield[:] = rfield1[:]

                # Apply the config-space Ylm
                for islab, slab in enumerate(rfield.slabs):
                    slab[:] *= Ylm(xhat[0][islab], xhat[1][islab], xhat[2][islab])

                # Real to complex of field #2
                rfield.r2c(out=cfield)

                for islab in range(A0.shape[0]):
                    tmp = cfield[islab] * A0[islab].conj()
                    if swap: tmp = tmp.conj()
                    cfield[islab, ...] = tmp

                cfield.c2r(out=rfield)
                zero = rfield.cmean()

                # Apply the separation-space Ylm
                for islab, slab in enumerate(rfield.slabs):
                    slab[:] *= Ylm(shat[0][islab], shat[1][islab], shat[2][islab])
                Aell[:] += rfield[:]

                # Apply the separation-space Ylm
                for islab, slab in enumerate(rfield.slabs):
                    slab[:] = zero * Ylm(shat[0][islab], shat[1][islab], shat[2][islab])
                Aell_zero[:] += rfield[:]

                # And this contribution to the total sum
                t1 = time.time()
                if rank == 0:
                    self.log_debug('Done term for Y(l={:d}, m={:d}) in {:.2f} s.'.format(Ylm.l, Ylm.m, t1 - t0))

            if rank == 0:
                self.log_info('ell = {:d} done; {:d} r2c completed'.format(ell, len(Ylms[ill])))

            # Project on to 1d s-basis (averaging over mu=[-1, 1])
            proj_result = project_to_basis(Aell, self.edges, antisymmetric=bool(ell % 2), exclude_zero=False)[0]
            result.append(4 * np.pi * np.ravel(proj_result[2]))
            proj_result = project_to_basis(Aell_zero, self.edges, antisymmetric=bool(ell % 2), exclude_zero=False)[0]
            result_zero.append(4 * np.pi * np.ravel(proj_result[2]))
            s, nmodes = proj_result[0], proj_result[3]

        # pmesh convention is F(s) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here
        corr, corr_zero = (np.array([r[ells.index(ell)] for ell in self.ells]) for r in (result, result_zero))
        corr, corr_zero = (self.nmesh.prod(dtype='f8')**2 / self.boxsize.prod(dtype='f8') * tmp for tmp in (corr, corr_zero))
        # Format the corr results into :class:`CorrelationFunctionMultipoles` instance
        s, nmodes = np.ravel(s), np.ravel(nmodes)
        kwargs = {'wnorm': self.wnorm, 'shotnoise_nonorm': self.shotnoise * self.wnorm * self.nmesh.prod(dtype='f8') / self.boxsize.prod(dtype='f8'),
                  'attrs': self.attrs, 'mpicomm': self.mpicomm}
        self.poles = CorrelationFunctionMultipoles(modes=s, edges=self.edges[0], corr_nonorm=corr, corr_zero_nonorm=corr_zero, nmodes=nmodes, ells=self.ells, **kwargs)


class CatalogFFTCorr(MeshFFTCorr):

    """Wrapper on :class:`MeshFFTCorr` to estimate correlation function directly from positions and weights."""

    def __init__(self, data_positions1, data_positions2=None, randoms_positions1=None, randoms_positions2=None,
                 shifted_positions1=None, shifted_positions2=None,
                 data_weights1=None, data_weights2=None, randoms_weights1=None, randoms_weights2=None,
                 shifted_weights1=None, shifted_weights2=None,
                 edges=None, ells=(0, 2, 4), los=None,
                 nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=2., wrap=False, dtype='f8',
                 resampler='tsc', interlacing=2, position_type='xyz', weight_type='auto', weight_attrs=None,
                 wnorm=None, shotnoise=None, shotnoise_nonorm=None, mpiroot=None, mpicomm=mpi.COMM_WORLD):
        r"""
        Initialize :class:`CatalogFFTCorr`, i.e. estimate correlation function.

        Note
        ----
        To compute the cross-correlation of samples 1 and 2, provide ``data_positions2``
        (and optionally ``randoms_positions2``, ``shifted_positions2`` for the selection function / shifted random catalogs of population 2).
        To compute (with the correct shot noise estimate) the auto-correlation of sample 1, but with 2 weights, provide ``data_positions1``
        (but no ``data_positions2``, nor ``randoms_positions2`` and ``shifted_positions2``), ``data_weights1`` and ``data_weights2``;
        ``randoms_weights2`` and ``shited_weights2`` default to ``randoms_weights1`` and ``shited_weights1``, resp.

        Warning
        -------
        In case line-of-sight is not local, one can provide :math:`\mu`-edges. In this case, integration over Legendre polynomials for multipoles
        is performed between the first and last :math:`\mu`-edges.
        For example, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, integration is performed between :math:`\mu = 0.2` and :math:`\mu = 0.8`.
        In all other cases, integration is performed between :math:`\mu = -1.0` and :math:`\mu = 1.0`.

        Parameters
        ----------
        data_positions1 : list, array
            Positions in the first data catalog. Typically of shape (3, N) or (N, 3).

        data_positions2 : list, array, default=None
            Optionally (for cross-correlation), positions in the second data catalog. See ``data_positions1``.

        randoms_positions1 : list, array, default=None
            Optionally, positions of the random catalog representing the first selection function.
            If no randoms are provided, selection function will be assumed uniform.

        randoms_positions2 : list, array, default=None
            Optionally (for cross-correlation), positions in the second randoms catalog. See ``randoms_positions1``.

        shifted_positions1 : array, default=None
            Optionally, in case of BAO reconstruction, positions of the first shifted catalog.

        shifted_positions2 : array, default=None
            Optionally, in case of BAO reconstruction, positions of the second shifted catalog.

        data_weights1 : array of shape (N,), default=None
            Optionally, weights in the first data catalog.

        data_weights2 : array of shape (N,), default=None
            Optionally (for cross-correlation), weights in the second data catalog.

        randoms_weights1 : array of shape (N,), default=None
            Optionally, weights in the first randoms catalog.

        randoms_weights2 : array of shape (N,), default=None
            Optionally (for cross-correlation), weights in the second randoms catalog.

        shifted_weights1 : array, default=None
            Optionally, weights of the first shifted catalog. See ``data_weights1``.

        shifted_weights2 : array, default=None
            Optionally, weights of the second shifted catalog.
            See ``shifted_weights1``.

        edges : tuple, array, default=None
            If ``los`` is local (``None``), :math:`s`-edges for :attr:`poles`.
            Else, one can also provide :math:`\mu`-edges (hence a tuple ``(sedges, muedges)``) for :attr:`wedges`.
            If ``sedges`` is ``None``, defaults to edges containing unique :math:`s` (norm) values, see :func:`find_unique_edges`.
            ``sedges`` may be a dictionary, with keys 'min' (minimum :math:`s`, defaults to 0), 'max' (maximum :math:`s`, defaults to ``boxsize/2``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`s` (norm) values between 'min' and 'max').
            For both :math:`s` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end, i.e. ``edges[i] <= x < edges[i+1]``.
            However, last :math:`\mu`-bin is inclusive on both ends: ``edges[-2] <= mu <= edges[-1]``.
            Therefore, with e.g. :math:`\mu`-edges ``[0.2, 0.4, 1.0]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 1.0`.
            Similarly, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 0.8`.

        ells : list, tuple, default=(0, 2, 4)
            Multipole orders.

        los : string, array, default='firstpoint'
            If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        nmesh : array, int, default=None
            Mesh size, i.e. number of mesh nodes along each axis.

        boxsize : array, float, default=None
            Physical size of the box along each axis, defaults to maximum extent taken by all input positions, times ``boxpad``.

        boxcenter : array, float, default=None
            Box center, defaults to center of the Cartesian box enclosing all input positions.

        cellsize : array, float, default=None
            Physical size of mesh cells.
            If not ``None``, and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` as ``nmesh * cellsize``.
            If ``nmesh`` is ``None``, it is set as (the nearest integer(s) to) ``boxsize / cellsize``.

        boxpad : float, default=2.
            When ``boxsize`` is determined from input positions, take ``boxpad`` times the smallest box enclosing positions as ``boxsize``.

        wrap : bool, default=False
            Whether to wrap input positions in [0, boxsize[.
            If ``False`` and input positions do not fit in the the box size, raise a :class:`ValueError`.

        dtype : string, dtype, default='f8'
            The data type to use for input positions and weights and the mesh.

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

            In addition, angular upweights can be provided with ``D1D2_twopoint_weights``, ``D1R2_twopoint_weights``, etc.
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

        wnorm : float, default=None
            Power spectrum normalization, to use instead of internal estimate obtained with :func:`normalization`.

        shotnoise : float, default=None
            Power spectrum shot noise, to use instead of internal estimate, which is 0 in case of cross-correlation
            and in case of auto-correlation is obtained by dividing :meth:`unnormalized_shotnoise` by correlation function normalization.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=mpi.COMM_WORLD
            The MPI communicator.
        """
        rdtype = _get_real_dtype(dtype)
        loc = locals()
        bpositions, positions = [], {}
        for name in ['data_positions1', 'data_positions2', 'randoms_positions1', 'randoms_positions2', 'shifted_positions1', 'shifted_positions2']:
            tmp = _format_positions(loc[name], position_type=position_type, dtype=rdtype, mpicomm=mpicomm, mpiroot=mpiroot)
            if tmp is not None: bpositions.append(tmp)
            label = name.replace('data_positions', 'D').replace('randoms_positions', 'R').replace('shifted_positions', 'S')
            positions[label] = tmp

        with_shifted = positions['S1'] is not None
        with_randoms = positions['R1'] is not None
        autocorr = positions['D2'] is None
        if autocorr and (positions['R2'] is not None or positions['S2'] is not None):
            raise ValueError('randoms_positions2 or shifted_positions2 are provided, but not data_positions2')

        weights = {name: loc[name] for name in ['data_weights1', 'data_weights2', 'randoms_weights1', 'randoms_weights2', 'shifted_weights1', 'shifted_weights2']}
        weights, bweights, n_bitwise_weights, weight_attrs = _format_all_weights(dtype=rdtype, weight_type=weight_type, weight_attrs=weight_attrs, mpicomm=mpicomm, mpiroot=mpiroot, **weights)

        self.same_shotnoise = autocorr and (weights['D2'] is not None)
        autocorr &= not self.same_shotnoise

        # Get box encompassing all catalogs
        nmesh, boxsize, boxcenter = _get_mesh_attrs(boxsize=boxsize, cellsize=cellsize, nmesh=nmesh, boxcenter=boxcenter, positions=bpositions, boxpad=boxpad, check=not wrap, mpicomm=mpicomm)
        if not isinstance(resampler, tuple):
            resampler = (resampler,) * 2
        if not isinstance(interlacing, tuple):
            interlacing = (interlacing,) * 2

        if wrap:
            for name, position in positions.items():
                if position is not None:
                    positions[name] = _wrap_positions(position, boxsize, boxcenter - boxsize / 2.)

        # Get catalog meshes
        def get_mesh(data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, shifted_positions=None, shifted_weights=None, **kwargs):
            return CatalogMesh(data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights,
                               shifted_positions=shifted_positions, shifted_weights=shifted_weights,
                               nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, position_type='pos', dtype=dtype, mpicomm=mpicomm, **kwargs)

        mesh1 = get_mesh(positions['D1'], data_weights=weights['D1'], randoms_positions=positions['R1'], randoms_weights=weights['R1'],
                         shifted_positions=positions['S1'], shifted_weights=weights['S1'], resampler=resampler[0], interlacing=interlacing[0], wrap=wrap)

        mesh2 = None
        if not autocorr:
            if self.same_shotnoise:
                for name in ['D', 'R', 'S']:
                    positions[name + '2'] = positions[name + '1']
                    if weights[name + '2'] is None: weights[name + '2'] = weights[name + '1']
            mesh2 = get_mesh(positions['D2'], data_weights=weights['D2'], randoms_positions=positions['R2'], randoms_weights=weights['R2'],
                             shifted_positions=positions['S2'], shifted_weights=weights['S2'], resampler=resampler[1], interlacing=interlacing[1], wrap=wrap)

        # Now, run correlation function estimation
        super(CatalogFFTCorr, self).__init__(mesh1=mesh1, mesh2=mesh2, edges=edges, ells=ells, los=los, wnorm=wnorm, shotnoise=shotnoise, shotnoise_nonorm=shotnoise_nonorm)
