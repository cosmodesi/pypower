"""
Implementation of (approximate) window function estimation and convolution.
Typically, the window function will be estimated through :class:`CatalogSmoothWindow`,
and window function matrices using :class:`PowerSpectrumSmoothWindowMatrix`,
following https://arxiv.org/abs/2106.06324.
"""

import math

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import special

from .utils import BaseClass, _make_array
from .fftlog import CorrelationToPower
from .fft_power import (BasePowerSpectrumStatistics, MeshFFTPower, CatalogMesh,
                        _get_real_dtype, _format_positions, _format_all_weights, _get_mesh_attrs, _wrap_positions)
from .wide_angle import Projection, BaseMatrix, CorrelationFunctionOddWideAngleMatrix, PowerSpectrumOddWideAngleMatrix
from . import mpi, utils


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]]) / 2.


class PowerSpectrumSmoothWindow(BasePowerSpectrumStatistics):

    """Power spectrum window function multipoles."""

    name = 'window'
    _attrs = BasePowerSpectrumStatistics._attrs + ['volume', 'projs', 'wnorm_ref']
    _tosum = ['nmodes', 'volume']

    def __init__(self, edges, modes, power_nonorm, nmodes, projs, wnorm_ref=None, **kwargs):
        r"""
        Initialize :class:`PowerSpectrumSmoothWindow`.

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

        wnorm_ref : float, array, default=None
            Normalization of the reference data power spectrum; defaults to ``wnorm``.

        kwargs : dict
            Other arguments for :attr:`BasePowerSpectrumStatistics`.
        """
        self.projs = [Projection(proj) for proj in projs]
        super(PowerSpectrumSmoothWindow, self).__init__(edges, modes, power_nonorm, nmodes, **kwargs)
        if np.size(self.shotnoise_nonorm) <= 1:
            shotnoise_nonorm = self.shotnoise_nonorm
            self.shotnoise_nonorm = _make_array(0., len(self.power_nonorm), dtype=self.power_nonorm.dtype)
            for iproj, proj in enumerate(self.projs):
                if proj.ell == 0: self.shotnoise_nonorm[iproj] = shotnoise_nonorm
        self.wnorm = _make_array(self.wnorm, len(self.power_nonorm), dtype=self.power_nonorm.dtype)
        if wnorm_ref is None:
            self.wnorm_ref = self.wnorm.copy()
        else:
            self.wnorm_ref = _make_array(wnorm_ref, len(self.power_nonorm), dtype=self.power_nonorm.dtype)
        self.volume = None
        if 'boxsize' in self.attrs:
            self.volume = (2. * np.pi)**3 / np.prod(self.attrs['boxsize']) * self.nmodes

    @property
    def _power_names(self):
        return ['W{:d},{:d}(k)'.format(proj.ell, proj.wa_order) for proj in self.projs]

    @property
    def kavg(self):
        """Mode-weighted average wavenumber = :attr:`k`."""
        return self.k

    def get_power(self, add_direct=True, remove_shotnoise=True, null_zero_mode=False, divide_wnorm=True, complex=True):
        """
        Return power spectrum, computed using various options.

        Parameters
        ----------
        add_direct : bool, default=True
            Add direct power spectrum measurement.

        remove_shotnoise : bool, default=True
            Remove estimated shot noise.

        null_zero_mode : bool, default=True
            Remove power spectrum at :math:`k = 0` (if within :attr:`edges`).

        divide_wnorm : bool, default=True
            Divide by estimated power spectrum normalization.

        complex : bool, default=True
            Whether (``True``) to return the complex power spectrum,
            or (``False``) return its real part if even multipoles, imaginary part if odd multipole.

        Results
        -------
        power : array
        """
        toret = super(PowerSpectrumSmoothWindow, self).get_power(add_direct=add_direct, remove_shotnoise=False, null_zero_mode=null_zero_mode, divide_wnorm=False, complex=True)
        if remove_shotnoise:
            toret -= self.shotnoise_nonorm[:, None]
        if divide_wnorm:
            toret /= self.wnorm[:, None]
        if not complex and np.iscomplexobj(toret):
            toret = np.array([toret[iproj].real if proj.ell % 2 == 0 else toret[iproj].imag for iproj, proj in enumerate(self.projs)], dtype=toret.real.dtype)
        return toret

    def __call__(self, proj=None, k=None, return_k=False, complex=True, default_zero=False, **kwargs):
        r"""
        Return window function, optionally performing linear interpolation over :math:`k`.

        Parameters
        ----------
        proj : tuple, Projection
            Projection, i.e. (multipole, wide-angle order) tuple.
            Defaults to all projections.

        k : float, array, default=None
            :math:`k` where to interpolate the window function.
            Defaults to :attr:`kavg` (no interpolation performed).
            Values outside :attr:`k` are set to the first/last window value.

        return_k : bool, default=False
            Whether (``True``) to return :math:`k`-modes (see ``k``).
            If ``None``, return :math:`k`-modes if ``k`` is ``None``.

        complex : bool, default=True
            Whether (``True``) to return the complex power spectrum,
            or (``False``) return its real part if ``proj.ell`` is even, imaginary part if ``proj.ell`` is odd.

        kwargs : dict
            Other arguments for :meth:`get_power`.

        Returns
        -------
        k : array
            Optionally, :math:`k`-modes.

        power : array
            (Optionally interpolated) window function.
        """
        isscalar = True
        if proj is None:
            isscalar = False
            proj = self.projs
        else:
            proj = [Projection(proj)]
        projs = proj
        power = self.get_power(complex=complex, **kwargs)
        tmp = []
        for proj in projs:
            if proj not in self.projs:
                if default_zero:
                    self.log_info('No window provided for projection {}, defaulting to 0.'.format(proj))
                    tmp.append(np.zeros_like(power[0]))
                else:
                    raise IndexError('No window provided for projection {}. If you want to ignore this error (set the corresponding window to zero), pass defaut_zero = True'.format(proj))
            else:
                tmp.append(power[self.projs.index(proj)])
        power = np.asarray(tmp)
        kavg = self.k.copy()
        if k is None:
            toret = power
            if isscalar:
                toret = power[0]
            if return_k:
                return kavg, toret
            return toret
        mask_finite_k = ~np.isnan(kavg) & ~np.isnan(power).any(axis=0)
        kavg, power = kavg[mask_finite_k], power[:, mask_finite_k]
        k = np.asarray(k)

        def interp(array):
            return np.array([UnivariateSpline(kavg, arr, k=1, s=0, ext='const')(k) for arr in array], dtype=array.dtype)

        toret = interp(power.real)
        if complex and np.iscomplexobj(power): toret = toret + 1j * interp(power.imag)
        if isscalar:
            toret = toret[0]
        if return_k:
            return k, toret
        return toret

    @classmethod
    def from_power(cls, power, wa_order=0, **kwargs):
        """
        Build window function from input :class:`PowerSpectrumMultipoless`.

        Parameters
        ----------
        power : PowerSpectrumMultipoless
            Power spectrum measurement to convert into :class:`PowerSpectrumSmoothWindow`.

        wa_order : int, default=0
            Wide-angle order used for input power spectrum measurement.

        Returns
        -------
        window : PowerSpectrumSmoothWindow
        """
        state = power.__getstate__()
        state.pop('name', None)
        state['projs'] = [Projection(ell=ell, wa_order=wa_order) for ell in state.pop('ells')]
        return cls(**state, **kwargs)

    @classmethod
    def concatenate_x(cls, *others, select='nmodes', frac_nyq=None):
        """
        Concatenate input window functions, along k-coordinates.
        k-edges and k-coordinates are taken from the first provided window;
        then expanded by those of other provided windows, if they cover a wider range.
        Eventually, for each bin (low, high), loop through all windows and select that with the largest number of modes
        in the given bin, if exists (bins are declared equal when exact floating point matching of (low, high)).
        In case two windows have the same number of modes in the same bin, the first provided one is selected.
        Therefore, different results may be obtained when changing the order of input windows.

        Note
        ----
        Typically, you will want to input windows with decreasing box size (largest box size first).

        Parameters
        ----------
        others : list of PowerSpectrumSmoothWindow
            List of window functions to be concatenated.

        select : string, default='nmodes'
            How to select input windows for each k (if several);
            'nmodes': select window with highest number of modes.

        frac_nyq : float, tuple, default=None
            Optionally, fraction of Nyquist frequency where to cut input windows (e.g. 0.8).
            If a float, the same for all input windows; else a tuple or a list of such fraction for each input window.

        Returns
        -------
        new : PowerSpectrumSmoothWindow
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = others[0].deepcopy()
        names = ['power_nonorm', 'power_direct_nonorm', 'nmodes', 'volume']

        if frac_nyq is None or np.ndim(frac_nyq) == 0:
            frac_nyq = (frac_nyq,) * len(others)
        else:
            frac_nyq = list(frac_nyq)
            frac_nyq += [None] * (len(others) - len(frac_nyq))  # complete with None

        def get_mask_nqy(ind):
            self = others[ind]
            if frac_nyq[ind] is None:
                return np.ones(self.shape[0], dtype='?')
            return self.edges[0][1:] <= frac_nyq[ind] * np.pi * np.min(self.attrs['nmesh'] / self.attrs['boxsize'])

        # Start with edges/modes of the first window
        mask_nyq = np.flatnonzero(get_mask_nqy(0))
        for name in names:
            setattr(new, name, getattr(new, name)[..., mask_nyq])
        new.edges[0] = new.edges[0][np.append(mask_nyq, mask_nyq[-1] + 1)]
        new.modes[0] = new.modes[0][mask_nyq]
        # Then expand with edges/modes of other windows, if those have a wider range
        for iother, other in enumerate(others[1:]):
            mid = (other.edges[0][:-1] + other.edges[0][1:]) / 2.
            mask_nyq = get_mask_nqy(iother + 1)
            mask_low, mask_high = np.flatnonzero((mid < new.edges[0][0]) & mask_nyq), np.flatnonzero((mid > new.edges[0][-1]) & mask_nyq)
            new.edges[0] = np.concatenate([other.edges[0][mask_low], new.edges[0], other.edges[0][mask_high + 1]], axis=0)
            for name in names:
                setattr(new, name, np.concatenate([getattr(other, name)[..., mask_low], getattr(new, name), getattr(other, name)[..., mask_high]], axis=-1))
            new.modes[0] = np.concatenate([other.modes[0][..., mask_low], new.modes[0], other.modes[0][..., mask_high]], axis=-1)

        # For each bin, loop through all windows and select that with the largest number of modes in the given bin, if exists
        tedges = list(zip(new.edges[0][:-1], new.edges[0][1:]))
        for other in others[1:]:
            for iother, tedge in enumerate(zip(other.edges[0][:-1], other.edges[0][1:])):
                if tedge in tedges:  # Search for k-bin in other window, requiring *exact* matching of floating point numbers...
                    inew = tedges.index(tedge)
                    if other.nmodes[iother] > new.nmodes[inew]:
                        for name in names:
                            getattr(new, name)[..., inew] = getattr(other, name)[..., iother]  # replace by value in window with highest number of modes
                        new.modes[0][..., inew] = other.modes[0][..., iother]

        return new

    @classmethod
    def concatenate_proj(cls, *others):
        """
        Concatenate input window functions, along projections.

        Parameters
        ----------
        others : list of PowerSpectrumSmoothWindow
            List of window functions to be concatenated.

        Returns
        -------
        new : PowerSpectrumSmoothWindow
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = others[0].deepcopy()
        new.projs = []
        for other in others: new.projs += other.projs
        names = ['power_nonorm', 'power_zero_nonorm', 'power_direct_nonorm', 'wnorm', 'wnorm_ref', 'shotnoise_nonorm']
        for name in names:
            array = [getattr(other, name) for other in others]
            setattr(new, name, np.concatenate(array, axis=0))
        return new

    def to_real(self, **kwargs):
        """
        Transform power spectrum window function to configuration space.

        Parameters
        ----------
        kwargs : dict
            Arguments for :func:`power_to_correlation_window`.

        Returns
        -------
        window : CorrelationFunctionSmoothWindow
        """
        return power_to_correlation_window(self, **kwargs)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = super(PowerSpectrumSmoothWindow, self).__getstate__()
        state['projs'] = [proj.__getstate__() for proj in self.projs]
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(PowerSpectrumSmoothWindow, self).__setstate__(state)
        if not hasattr(self, 'wnorm_ref'):
            self.wnorm_ref = self.wnorm.copy()  # backward-compatibility
        self.projs = [Projection.from_state(state) for state in self.projs]

    @classmethod
    def average(cls, *others, weights=None):
        """
        Average input window functions.

        Warning
        -------
        Input power spectra have same edges / number of modes for this operation to make sense
        (no checks performed).
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        if weights is None:
            wnorm_ref = [other.wnorm_ref for other in others]
            wnorm = [other.wnorm for other in others]
            weights = [wd / sum(wnorm_ref) / (w / sum(wnorm)) for wd, w in zip(wnorm_ref, wnorm)]
        new = super(PowerSpectrumSmoothWindow, cls).average(*others, weights=weights)
        new.wnorm_ref = sum(other.wnorm_ref for other in others)
        return new


class CorrelationFunctionSmoothWindow(BaseClass):

    """Correlation window function multipoles."""

    _attrs = ['sep', 'corr', 'wnorm_ref']

    def __init__(self, sep, corr, projs, wnorm_ref=None):
        r"""
        Initialize :class:`CorrelationFunctionSmoothWindow`.

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
        self.wnorm_ref = wnorm_ref

    def __call__(self, proj=None, sep=None, return_sep=False, default_zero=False):
        r"""
        Return window function, optionally performing linear interpolation over :math:`s`.

        Parameters
        ----------
        proj : tuple, Projection, default=None
            Projection, i.e. (multipole, wide-angle order) tuple.
            Defaults to all projections.

        sep : float, array, default=None
            :math:`s` where to interpolate the window function.
            Values outside :attr:`sep` are set to the first/last window value.
            Defaults to :attr:`sep` (no interpolation performed).

        return_sep : bool, default=False
            Whether (``True``) to return :math:`s` (see ``sep``).
            If ``None``, return :math:`s` if ``sep`` is ``None``.

        default_zero : bool, default=False
            If input ``proj`` is not in :attr:`projs` (not computed), and ``default_zero`` is ``True``, return 0.
            If ``default_zero`` is ``False``, raise an :class:`IndexError`.

        Returns
        -------
        sep : array
            Optionally, :math:`s`.

        corr : array
            (Optionally interpolated) window function.
        """
        isscalar = True
        if proj is None:
            isscalar = False
            proj = self.projs
        else:
            proj = [Projection(proj)]
        projs = proj
        tmp = []
        for proj in projs:
            if proj not in self.projs:
                if default_zero:
                    self.log_info('No window provided for projection {}, defaulting to 0.'.format(proj))
                    tmp.append(np.zeros_like(self.corr[0]))
                else:
                    raise IndexError('No window provided for projection {}. If you want to ignore this error (set the corresponding window to zero), pass defaut_zero = True'.format(proj))
            else:
                tmp.append(self.corr[self.projs.index(proj)])
        corr = np.asarray(tmp)
        sepavg = self.sep.copy()
        if return_sep is None:
            return_sep = sep is None
        if sep is None:
            toret = corr
            if isscalar:
                toret = corr[0]
            if return_sep:
                return sep, toret
            return toret
        mask_finite_sep = ~np.isnan(sepavg) & ~np.isnan(corr).any(axis=0)
        sepavg, corr = sepavg[mask_finite_sep], corr[:, mask_finite_sep]
        toret = np.array([UnivariateSpline(sepavg, arr, k=1, s=0, ext='const')(sep) for arr in corr], dtype=corr.dtype)
        if isscalar:
            toret = toret[0]
        if return_sep:
            return sep, toret
        return toret

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
        super(CorrelationFunctionSmoothWindow, self).__setstate__(state)
        self.projs = [Projection.from_state(state) for state in self.projs]


def power_to_correlation_window(fourier_window, sep=None, k=None, smooth=None):
    r"""
    Compute correlation window function by taking Hankel transforms of input power spectrum window function.

    Parameters
    ----------
    fourier_window : PowerSpectrumSmoothWindow
        Power spectrum window function.

    sep : array, default=None
        Separations :math:`s` where to compute Hankel transform; defaults to inverse of ``fourier_window`` wavenumbers.

    k : array, default=None
        Wavenumbers where to interpolate the window function.
        If provided, :math:`k`-space volume element will be computed as :math:`4 \pi dk k^{2}`.
        Else, defaults to :attr:`fourier_window.k` and :attr:`fourier_window.volume`.

    smooth : float, array, default=None
        If not ``None``, if float, radius of Gaussian smoothing.
        Else, smoothing kernel, should be the same size as used ``k`` (see above).

    Returns
    -------
    window : CorrelationFunctionSmoothWindow
        Correlation window function.
    """
    sum_on_window = k is None
    if sum_on_window:
        k = fourier_window.k
        mask_finite = ~np.isnan(k) & ~np.isnan(fourier_window.power).any(axis=0)
        k = k[mask_finite]
        volume = fourier_window.volume[mask_finite]
    else:
        dk = np.diff(k)
        volume = 4. * np.pi * np.append(dk, dk[-1]) * k**2
    if smooth is None:
        smoothing = 1.
    elif np.ndim(smooth) == 0:
        smoothing = np.exp(-(smooth * k)**2)
    else:
        smoothing = np.asarray(smooth)
        if smoothing.size != k.size:
            raise ValueError('smoothing kernel must be of the same size as k coordinates i.e. {:d}'.format(k.size))
    # mask = k > 0
    # k = k[mask]; volume = volume[mask]
    if sep is None:
        sep = 1. / k[k > 0][::-1]
    else:
        sep = np.asarray(sep)
    window = []
    _slab_npoints_max = 10 * 1000
    for proj in fourier_window.projs:
        wk = fourier_window(proj=proj, k=k, complex=False, null_zero_mode=False) * smoothing
        block = np.empty_like(sep)
        nslabs = min(max(len(k) * len(sep) // _slab_npoints_max, 1), len(sep))
        for islab in range(nslabs):  # proceed by slab to save memory
            sl = slice(islab * len(sep) // nslabs, (islab + 1) * len(sep) // nslabs)
            ks = k[:, None] * sep[sl]
            integrand = wk[:, None] * 1. / (2. * np.pi)**3 * special.spherical_jn(proj.ell, ks)
            # Prefactor is (-i)^ell, but we take in the imaginary part of odd power spectra, hence:
            # (-i)^ell = (-1)^(ell/2) if ell is even
            # (-i)^ell i = (-1)^(ell//2) if ell is odd
            prefactor = (-1) ** (proj.ell // 2)
            block[sl] = prefactor * np.sum(volume[:, None] * integrand, axis=0)
        window.append(block)

    return CorrelationFunctionSmoothWindow(sep, window, fourier_window.projs.copy(), wnorm_ref=fourier_window.wnorm_ref)


class CatalogSmoothWindow(MeshFFTPower):

    """Wrapper on :class:`MeshFFTPower` to estimate window function from input random positions and weigths."""

    def __init__(self, randoms_positions1=None, randoms_positions2=None,
                 randoms_weights1=None, randoms_weights2=None,
                 edges=None, projs=None, power_ref=None,
                 los=None, nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=2., wrap=False, dtype=None,
                 resampler=None, interlacing=None, position_type='xyz', weight_type='auto', weight_attrs=None,
                 wnorm=None, shotnoise=None, mpiroot=None, mpicomm=mpi.COMM_WORLD):
        r"""
        Initialize :class:`CatalogSmoothWindow`, i.e. estimate power spectrum window.

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
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').

        projs : list, default=None
            List of :class:`Projection` instances or (multipole, wide-angle order) tuples.
            If ``None``, and ``power_ref`` is provided, the list of projections is set
            to be able to compute window convolution of theory power spectrum multipoles of orders ``power_ref.ells``.
            Namely, for maximum theory and output multipole :math:`\ell_{\mathrm{max}}`,
            window function multipoles will be computed at wide-angle order 0 up to :math:`2 \ell_{\mathrm{max}}`
            (maximum order yielded by the product of any theory and output Legendre polynomial,
            see e.g. eq. C8 of https://arxiv.org/pdf/2106.06324.pdf).
            In addition, if chosen line-of-sight is local (either 'firstpoint' or 'endpoint'),
            odd poles of the window function will be computed at wide-angle order 1, up to :math:`2 \ell_{\mathrm{max}} + 1`
            (maximum non-zero odd pole of wide angle order 1 generated by even poles up to :math:`\ell_{\mathrm{max}}` is :math:`\ell_{\mathrm{max}} + 1`).
            Finally, if any of ``power_ref.ells`` is odd, all (even and odd) poles will be computed at wide-angle orders 0 and 1,
            up to :math:`2 \ell_{\mathrm{max}}` and :math:`2 \ell_{\mathrm{max}} + 1`, respectively.

        power_ref : PowerSpectrumMultipoless, default=None
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

        boxsize : array, float, default=None
            Physical size of the box along each axis.
            If ``None``, defaults to the value used in estimation of ``power_ref``.

        boxcenter : array, float, default=None
            Box center, defaults to center of the Cartesian box enclosing all input positions.
            If ``None``, defaults to the value used in estimation of ``power_ref``.

        cellsize : array, float, default=None
            Physical size of mesh cells.
            If not ``None``, and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` as ``nmesh * cellsize``.
            If ``nmesh`` is ``None``, it is set as (the nearest integer(s) to) ``boxsize / cellsize``.

        boxpad : float, default=2.
            When ``boxsize`` is determined from input positions, take ``boxpad`` times the smallest box enclosing positions as ``boxsize``.

        wrap : bool, default=False
            Whether to wrap input positions in [0, boxsize[?
            If ``False`` and input positions do not fit in the the box size, raise a :class:`ValueError`.

        dtype : string, dtype, default=None
            The data type to use for input positions and weights and the mesh.
            If ``None``, defaults to the value used in estimation of ``power_ref`` if provided, else 'f8'.

        resampler : string, ResampleWindow, default=None
            Resampler used to assign particles to the mesh.
            Choices are ['ngp', 'cic', 'tcs', 'pcs'].
            If ``None``, defaults to the value used in estimation of ``power_ref``.

        interlacing : bool, int, default=None
            Whether to use interlacing to reduce aliasing when painting the particles on the mesh.
            If positive int, the interlacing order (minimum: 2).
            If ``None``, defaults to the value used in estimation of ``power_ref``.

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
                - "auto": automatically choose weighting based on input ``weights1`` and ``weights2``,
                   i.e. ``None`` when ``weights1`` and ``weights2`` are ``None``,
                   else "product_individual".

            If floating weights are of (real) type ``dtype`` and ``mpiroot`` is ``None``,
            no internal copy of weights will be made, hence saving some memory.

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
            Window function shot noise, to use instead of internal estimate, which is 0 in case of cross-correlation
            and in case of auto-correlation is obtained by dividing :meth:`CatalogMesh.unnormalized_shotnoise` by window function normalization.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scatted across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=mpi.COMM_WORLD
            The MPI communicator.
        """
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
                interlacing = tuple(power_ref.attrs['interlacing{:d}'.format(i + 1)] for i in range(2))
            if resampler is None:
                resampler = tuple(power_ref.attrs['resampler{:d}'.format(i + 1)] for i in range(2))
            if projs is None:
                ellmax = max(power_ref.ells)
                with_odd = int(any(ell % 2 for ell in power_ref.ells))
                projs = [(ell, 0) for ell in range(0, 2 * ellmax + 1, 2 - with_odd)]
                if los is None or isinstance(los, str) and los in ['firstpoint', 'endpoint']:
                    projs += [(ell, 1) for ell in range(1 - with_odd, 2 * ellmax + 2, 2 - with_odd)]  # e.g. P_5^{(1)} contribution to P_4 => ell = 9
            if dtype is None: dtype = power_ref.attrs.get('dtype', 'f8')

        if dtype is None: dtype = 'f8'
        rdtype = _get_real_dtype(dtype)
        if projs is None:
            raise ValueError('If no reference power spectrum "power_ref" provided, provide list of projections "projs".')
        projs = [Projection(proj) for proj in projs]
        ells_for_wa_order = {proj.wa_order: [] for proj in projs}
        for proj in projs:
            ells_for_wa_order[proj.wa_order].append(proj.ell)

        if cellsize is not None:  # if cellsize is provided, remove default nmesh or boxsize value from old_matrix instance.
            mesh_attrs['cellsize'] = cellsize
            if nmesh is None: mesh_attrs.pop('nmesh')
            elif boxsize is None: mesh_attrs.pop('boxsize')

        loc = locals()
        bpositions, positions = [], {}
        for name in ['randoms_positions1', 'randoms_positions2']:
            tmp = _format_positions(loc[name], position_type=position_type, dtype=rdtype, mpicomm=mpicomm, mpiroot=mpiroot)
            if tmp is not None: bpositions.append(tmp)
            label = name.replace('randoms_positions', 'R')
            positions[label] = tmp

        autocorr = positions['R2'] is None

        weights = {name: loc[name] for name in ['randoms_weights1', 'randoms_weights2']}
        weights, bweights, n_bitwise_weights, weight_attrs = _format_all_weights(dtype=rdtype, weight_type=weight_type, weight_attrs=weight_attrs, mpicomm=mpicomm, mpiroot=mpiroot, **weights)

        # Get box encompassing all catalogs
        nmesh, boxsize, boxcenter = _get_mesh_attrs(**mesh_attrs, positions=bpositions, boxpad=boxpad, check=not wrap, mpicomm=mpicomm)
        if resampler is None: resampler = 'tsc'
        if interlacing is None: interlacing = 2
        if not isinstance(resampler, tuple):
            resampler = (resampler,) * 2
        if not isinstance(interlacing, tuple):
            interlacing = (interlacing,) * 2

        if wrap:
            for name, position in positions.items():
                if position is not None:
                    positions[name] = _wrap_positions(position, boxsize, boxcenter - boxsize / 2.)

        ialpha2 = 1.
        wnorm_ref = None
        if power_ref is not None:
            wnorm_ref = power_ref.wnorm
            if wnorm is None:
                wsum = [mpicomm.allreduce(sum(weights['R1']) if weights['R1'] is not None else len(positions['R1']))] * 2
                if not autocorr: wsum[1] = mpicomm.allreduce(sum(weights['R2']) if weights['R2'] is not None else len(positions['R2']))
                ialpha2 = np.prod([wsum[ii] / power_ref.attrs[name] for ii, name in enumerate(['sum_data_weights1', 'sum_data_weights2'])])
                wnorm = ialpha2 * wnorm_ref

        def get_mesh(data_positions, data_weights=None, **kwargs):
            return CatalogMesh(data_positions, data_weights=data_weights,
                               nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter,
                               position_type='pos', dtype=dtype, mpicomm=mpicomm, **kwargs)

        # Get catalog meshes
        mesh1 = get_mesh(positions['R1'], data_weights=weights['R1'], resampler=resampler[0], interlacing=interlacing[0])
        if autocorr:
            mesh2 = None
        else:
            mesh2 = get_mesh(positions['R2'], data_weights=weights['R2'], resampler=resampler[1], interlacing=interlacing[1])

        poles = []
        for wa_order, ells in ells_for_wa_order.items():
            mesh2_wa = mesh2
            if wa_order == 0:
                mesh1_wa = mesh1
            else:
                label = 'R1'
                weights1 = np.ones_like(positions[label], shape=len(positions[label])) if weights[label] is None else weights[label]
                d = utils.distance(positions[label].T)
                mesh1_wa = get_mesh(positions[label], data_weights=weights1 / d**wa_order, resampler=resampler[1], interlacing=interlacing[1])
                if autocorr:
                    mesh2_wa = mesh1
                else:
                    mesh2_wa = mesh2
            # Now, run power spectrum estimation
            super(CatalogSmoothWindow, self).__init__(mesh1=mesh1_wa, mesh2=mesh2_wa, edges=edges, ells=ells, los=los, wnorm=wnorm, shotnoise=shotnoise)
            if autocorr and shotnoise is None and wa_order != 0:  # when providing 2 meshes, shot noise estimate is 0; correct this here
                weights2 = mesh1_wa.data_weights * mesh2_wa.data_weights if mesh2_wa.data_weights is not None else mesh1_wa.data_weights
                self.poles.shotnoise_nonorm = mesh1_wa.mpicomm.allreduce(sum(weights2))
            poles.append(PowerSpectrumSmoothWindow.from_power(self.poles, wa_order=wa_order, wnorm_ref=wnorm_ref, mpicomm=self.mpicomm))

        self.poles = PowerSpectrumSmoothWindow.concatenate_proj(*poles)
        del self.ells

    @classmethod
    def concatenate_x(cls, *others, **kwargs):
        """
        Concatenate :attr:`poles`.
        Same argument as :meth:`PowerSpectrumSmoothWindow.concatenate_x`.
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = others[0].copy()
        for name in ['poles']:
            if hasattr(others[0], name):
                setattr(new, name, PowerSpectrumSmoothWindow.concatenate_x(*[getattr(other, name) for other in others]))
        return new


class CorrelationFunctionSmoothWindowMatrix(BaseMatrix):
    """
    Class computing matrix for window product in configuration space.

    Attributes
    ----------
    projmatrix : array
        Array of shape ``(len(self.projsout), len(self.projsin), len(self.x))``.
    """
    def __init__(self, sep, projsin, projsout=None, window=None, sum_wa=True, default_zero=False, attrs=None):
        """
        Initialize :class:`CorrelationFunctionSmoothWindowMatrix`.

        Parameters
        ----------
        sep : array
            Input (and ouput) separations.

        projsin : list
            Input projections.

        projsout : list, default=None
            Output projections. Defaults to ``propose_out(projsin, sum_wa=sum_wa)``.

        window : CorrelationFunctionSmoothWindow, PowerSpectrumSmoothWindow
            Window function to convolve power spectrum with.
            If a :class:`PowerSpectrumSmoothWindow` instance is provided, it is transformed to configuration space.

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
        self.attrs.update(attrs or {})
        self.sum_wa = sum_wa
        self.default_zero = default_zero

        self.projsin = [Projection(proj) for proj in projsin]
        if projsout is None:
            self.projsout = self.propose_out(projsin, sum_wa=self.sum_wa)
        else:
            self.projsout = [Projection(proj, default_wa_order=None if self.sum_wa else 0) for proj in projsout]

        self._set_xw(xin=sep, xout=sep, weight=getattr(window, 'wnorm_ref', [1.])[0])
        self.run()

    def run(self):
        r"""
        Set up transform, i.e. compute matrix:

        .. math::

            W_{\ell,\ell^{\prime}}^{(n,n^{\prime})}(s) = \delta_{n n^{\prime}} \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}^{(n)}(s)

        with :math:`\ell` multipole order corresponding to ``projout.ell`` and :math:`\ell^{\prime}` to ``projin.ell``,
        :math:`n` wide angle order corresponding to ``projout.wa_order`` and :math:`n^{\prime}` to ``projin.wa_order``.
        If :attr:`sum_wa` is ``True``, or output ``projout.wa_order`` is ``None``, sum over :math:`n` (always the case except for debugging purposes).
        For example, see q. D5 and D6 of arXiv:1810.05051.
        """
        window = self.window
        sep = self.xin[0]
        if isinstance(window, PowerSpectrumSmoothWindow):
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
                        block += coeff * window(proj=proj, sep=sep, default_zero=self.default_zero)
                line.append(block)
            self.projvalue.append(line)
        self.projvalue = np.array(self.projvalue)  # (in, out)

    @property
    def value(self):
        if getattr(self, '_value', None) is None:
            self._value = np.bmat([[np.diag(block) for block in line] for line in self.projvalue]).A
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
        projsin = [proj for proj in self.projsin if proj.wa_order == 0]
        if projsin == self.projsin: return
        if 'los' not in kwargs and 'los_type' in self.attrs: kwargs['los'] = self.attrs['los_type']
        matrix = CorrelationFunctionOddWideAngleMatrix([0.], projsin, projsout=self.projsin, **kwargs).value
        self.prod_proj(matrix, axes=('in', -1), projs=projsin)


class PowerSpectrumSmoothWindowMatrix(BaseMatrix):

    """Class computing matrix for window convolution in Fourier space."""

    _slab_npoints_max = 10 * 1000

    def __init__(self, kout, projsin, projsout=None, weightsout=None, k=None, kin_rebin=1, kin_lim=(1e-4, 1.), sep=None, window=None, xy=1, q=0, sum_wa=True, default_zero=False, attrs=None):
        """
        Initialize :class:`PowerSpectrumSmoothWindowMatrix`.

        Parameters
        ----------
        kout : array
            Output wavenumbers.

        projsin : list
            Input projections.

        projsout : list, default=None
            Output projections. Defaults to ``propose_out(projsin, sum_wa=sum_wa)``.

        weightsout : array, list, default=None
            Optionally, list of weights to apply when rebinning output "observed" coordinates.
            Typically, one would use for ``kout`` and ``weightsout`` the modes :attr:`BasePowerSpectrumStatistics.k`
            and number of modes :attr:`BasePowerSpectrumStatistics.nmodes` of the observed power spectrum,
            such that output k-modes of power spectrum and window matrix match after rebinning.

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

        window : CorrelationFunctionSmoothWindow, PowerSpectrumSmoothWindow
            Window function to convolve power spectrum with.
            If a :class:`PowerSpectrumSmoothWindow` instance is provided, it is transformed to configuration space.

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
            self.sep = xy / self.kin[::-1]
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
        self.attrs.update(attrs or {})
        self.sum_wa = sum_wa
        self.default_zero = default_zero

        self.projsin = [Projection(proj) for proj in projsin]
        if projsout is None:
            self.projsout = self.propose_out(projsin)
        else:
            self.projsout = [Projection(proj, default_wa_order=None if self.sum_wa else 0) for proj in projsout]

        self._set_xw(xin=self.k, xout=kout, weightsout=weightsout, weight=getattr(window, 'wnorm_ref', [1.])[0])
        self.run()

    def run(self):
        r"""
        Set matrix. Provided arXiv:2106.06324 eq. 2.5:

        .. math::

            W_{\ell\ell^{\prime}}^{(n)}(k) = \frac{2}{\pi} i^{\ell} (-i)^{\ell^{\prime}} \int ds s^{2} j_{\ell}(ks) j_{\ell^{\prime}}(k^{\prime}s)
            \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}^{(n)}(s)

        with :math:`\ell` corresponding to ``projout.ell`` and :math:`\ell^{\prime}` to ``projin.ell``, :math:`k` to ``kout`` and :math:`k^{\prime}` to ``kin``.
        :math:`n` is the wide-angle order ``proj.wa_order``.
        Yet, to avoid bothering with complex values, we only work with the imaginary part of odd power spectra (input and output).
        In addition, we include the :math:`dk^{\prime} k^{\prime 2}` volume element (arXiv:2106.06324 eq. 2.7). Hence we actually implement:

        .. math::

            W_{\ell,\ell^{\prime}}^{(n)}(k) = dk^{\prime} k^{\prime 2} \frac{2}{\pi} (-1)^{\ell/2} (-1)^{\ell^{\prime}/2} \int ds s^{2} j_{\ell}(ks) j_{\ell^{\prime}}(k^{\prime}s)
            \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}^{(n)}(s)

        Note that we do not include :math:`k^{-n}` as this factor is included in :class:`PowerSpectrumOddWideAngleMatrix`.
        """
        self.corrmatrix = CorrelationFunctionSmoothWindowMatrix(self.sep, self.projsin, projsout=self.projsout, window=self.window, sum_wa=self.sum_wa, default_zero=self.default_zero).projvalue

        self.value = []

        krebin = utils.rebin(self.k, len(self.k) // self.kin_rebin, statistic=np.mean)
        maskin = np.ones(len(krebin), dtype='?')
        if self.kin_lim is not None:
            maskin &= (krebin >= self.kin_lim[0]) & (krebin <= self.kin_lim[-1])
        nin = maskin.sum()

        for iin, projin in enumerate(self.projsin):
            self.xin[iin] = krebin[maskin]
            line = []
            for iout, projout in enumerate(self.projsout):
                nout = len(self.xout[iout])
                block = np.empty((nin, nout), dtype=self.corrmatrix.dtype)
                nslabs = min(max(len(self.sep) * nout // self._slab_npoints_max, 1), nout)
                for islab in range(nslabs):  # proceed by slab to save memory (if nin << len(self.sep))
                    slout = slice(islab * nout // nslabs, (islab + 1) * nout // nslabs)
                    xout = self.xout[iout][slout]
                    # tmp is j_{\ell}(ks) \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}(s)
                    tmp = special.spherical_jn(projout.ell, xout[:, None] * self.sep) * self.corrmatrix[iin, iout]  # matrix has dimensions (kout,s)
                    # from hankl import P2xi, xi2P
                    fftlog = CorrelationToPower(self.sep, ell=projin.ell, q=self.q, xy=self.xy, lowring=False, complex=False)
                    xin, tmp = fftlog(tmp)  # matrix has dimensions (kout, k)
                    assert np.allclose(xin, self.k)
                    # Current prefactor is 4 \pi i^{\ell^{\prime}}, real part for even poles, imag part for odd poles = 4 \pi (-1)^{\ell^{\prime}/2}
                    # Desired prefactor is 2/\pi i^{\ell} (-i)^{\ell^{\prime}}, but:
                    # - we provide the imag part for odd poles: i^{\ell} => (-1)^{\ell/2}
                    # - we take in the imag part for odd poles: (-i)^{\ell^{\prime}} => (-i)^{\ell^{\prime}} i = (-1)^{\ell^{\prime}/2} when \ell^{\prime} is odd
                    prefactor = 1 / (2 * np.pi**2) * (-1)**(projout.ell // 2)  # * (-1)**(projin.ell // 2) / (-1)**(projin.ell // 2)
                    # tmp is dk^{\prime} k^{\prime 2} \frac{2}{\pi} (-1)^{\ell} \int ds s^{2} j_{\ell}(ks) j_{\ell^{\prime}}(k^{\prime}s) \sum_{L} C_{\ell \ell^{\prime} L} Q_{L}(s)
                    tmp = np.real(prefactor * tmp) * weights_trapz(xin**3) / 3.  # everything should be real now
                    block[:, slout] = utils.rebin(tmp.T, (len(krebin), len(xout)), statistic=np.sum)[maskin, :]  # matrix has dimensions (k, kout)
                line.append(block)
            self.value.append(line)
        self.value = np.bmat(self.value).A  # (in, out)

    propose_out = CorrelationFunctionSmoothWindowMatrix.propose_out

    def resum_input_odd_wide_angle(self, **kwargs):
        """
        Resum odd wide-angle orders. By default, line-of-sight is chosen as that save in :attr:`attrs` (``attrs['los_type']``).
        To override, use input ``kwargs`` which will be passed to :attr:`PowerSpectrumOddWideAngleMatrix`.
        """
        projsin = [proj for proj in self.projsin if proj.wa_order == 0]
        if projsin == self.projsin: return
        if 'los' not in kwargs and 'los_type' in self.attrs: kwargs['los'] = self.attrs['los_type']
        matrix = PowerSpectrumOddWideAngleMatrix(self.xin[0], projsin=projsin, projsout=self.projsin, **kwargs)
        self.__dict__.update(self.join(matrix, self).__dict__)


def wigner3j_square(ellout, ellin, prefactor=True):
    r"""
    Return the coefficients corresponding to the product of two Legendre polynomials, corresponding to :math:`C_{\ell \ell^{\prime} L}`
    of e.g. eq. 2.2 of https://arxiv.org/pdf/2106.06324.pdf, with :math:`\ell` corresponding to ``ellout`` and :math:`\ell^{\prime}` to ``ellin``.

    Parameters
    ----------
    ellout : int
        Output order.

    ellin : int
        Input order.

    prefactor : bool, default=True
        Whether to include prefactor :math:`(2 \ell + 1)/(2 L + 1)` for window convolution.

    Returns
    -------
    ells : list
        List of mulipole orders :math:`L`.

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
        for p in range(1, p + 1): toret *= (2 * p - 1)
        return toret, math.factorial(p)

    for p in range(min(ellin, ellout) + 1):

        numer, denom = [], []

        # numerator of product of G(x)
        for r in [G(ellout - p), G(p), G(ellin - p)]:
            numer.append(r[0])
            denom.append(r[1])

        # divide by this
        a, b = G(ellin + ellout - p)
        numer.append(b)
        denom.append(a)

        numer.append(2 * (ellin + ellout) - 4 * p + 1)
        denom.append(2 * (ellin + ellout) - 2 * p + 1)

        q = ellin + ellout - 2 * p
        if prefactor:
            numer.append(2 * ellout + 1)
            denom.append(2 * q + 1)

        coeffs.append(np.prod(numer, dtype='f8') * 1. / np.prod(denom, dtype='f8'))
        qvals.append(q)

    return qvals[::-1], coeffs[::-1]
