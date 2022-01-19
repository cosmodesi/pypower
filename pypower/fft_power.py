r"""
Implementation of power spectrum estimator, following https://arxiv.org/abs/1704.02357.
Apart from interface choices, differences w.r.t. original nbodykit's implementation
https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py are:

- real space positions are taken at mesh nodes, instead of 0.5 cell shift (matters only for ell > 0 in global line-of-sight)
- normalization is computed with density obtained by paintaing data/randoms to mesh, instead of relying on :math:`\bar{n}_{i}` column in the catalogs
- FKP weights are treated as other weights
"""

import time

import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from pmesh.pm import RealField, ComplexField

from .utils import BaseClass
from . import mpi, utils
from .mesh import _get_real_dtype, _get_compensation_window, _get_box, CatalogMesh
from .direct_power import _make_array, _format_positions, _format_weights, get_default_nrealizations, get_inverse_probability_weight, get_direct_power_engine


def get_real_Ylm(ell, m):
    """
    Return a function that computes the real spherical harmonic of order (ell, m).
    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py.

    Note
    ----
    Faster evaluation will be achieved if sympy and numexpr are available.
    Else, fallback to numpy and scipy's functions.

    Parameters
    ----------
    ell : int
        The degree of the harmonic.

    m : int
        The order of the harmonic; abs(m) <= ell.

    Returns
    -------
    Ylm : callable
        A function that takes 3 arguments: (xhat, yhat, zhat)
        unit-normalized Cartesian coordinates and returns the
        specified Ylm.

    References
    ----------
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    """
    # Make sure ell, m are integers
    ell = int(ell)
    m = int(m)

    # Normalization of Ylms
    amp = np.sqrt((2*ell + 1) / (4*np.pi))
    if m != 0:
        fac = 1
        for n in range(ell - abs(m) + 1, ell + abs(m) + 1): fac *= n # (ell + |m|)!/(ell - |m|)!
        amp *= np.sqrt(2. / fac)

    try: import sympy as sp
    except ImportError: sp = None

    # sympy is not installed, fallback to scipy
    if sp is None:

        from scipy import special

        def Ylm(xhat, yhat, zhat):
            # The cos(theta) dependence encoded by the associated Legendre polynomial
            toret = amp * (-1)**m * special.lpmv(abs(m), ell, zhat)
            # The phi dependence
            phi = np.arctan2(yhat, xhat)
            if m < 0:
                sin_phi = yhat/np.sqrt(xhat**2 + yhat**2)
                toret *= np.sin(abs(m)*phi)
            else:
                cos_phi = xhat/np.sqrt(xhat**2 + yhat**2)
                toret *= np.cos(abs(m)*phi)
            return toret

        # Attach some meta-data
        Ylm.l = ell
        Ylm.m = m
        return Ylm

    # The relevant cartesian and spherical symbols
    # Using intermediate variable r helps sympy simplify expressions
    x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
    xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
    phi, theta = sp.symbols('phi theta')
    defs = [(sp.sin(phi), y/sp.sqrt(x**2 + y**2)),
            (sp.cos(phi), x/sp.sqrt(x**2 + y**2)),
            (sp.cos(theta), z/sp.sqrt(x**2 + y**2 + z**2))]

    # The cos(theta) dependence encoded by the associated Legendre polynomial
    expr = (-1)**m * sp.assoc_legendre(ell, abs(m), sp.cos(theta))

    # The phi dependence
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m)*phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m*phi))

    # Simplify
    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x/r, xhat), (y/r, yhat), (z/r, zhat)])

    try: import numexpr
    except ImportError: numexpr = None
    Ylm = sp.lambdify((xhat, yhat, zhat), expr, modules='numexpr' if numexpr is not None else ['scipy', 'numpy'])

    # Attach some meta-data
    Ylm.expr = expr
    Ylm.l = ell
    Ylm.m = m
    return Ylm


def project_to_basis(y3d, edges, los=(0, 0, 1), ells=None, antisymmetric=False):
    r"""
    Project a 3D statistic on to the specified basis. The basis will be one of:

        - 2D :math:`(x, \mu)` bins: :math:`\mu` is the cosine of the angle to the line-of-sight
        - 2D :math:`(x, \ell)` bins: :math:`\ell` is the multipole number, which specifies
          the Legendre polynomial when weighting different :math:`\mu` bins.

    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/fftpower.py.

    Notes
    -----
    In single precision (float32/complex64) nbodykit's implementation is fairly imprecise
    due to incorrect binning of :math:`x` and :math:`\mu` modes.
    Here we cast mesh coordinates to the maximum precision of input ``edges``,
    which makes computation much more accurate in single precision.

    Notes
    -----
    We deliberately set to 0 the number of modes beyond Nyquist, as it is unclear whether to count Nyquist as :math:`\mu` or :math:`-\mu`
    (it should probably be half weight for both).
    Our safe choice ensures consistent results between hermitian compressed and their associated uncompressed fields.

    Notes
    -----
    The 2D :math:`(x, \ell)` bins will be computed only if ``ells`` is specified.
    See return types for further details.
    For both :math:`x` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end,
    i.e. mode `mode` falls in bin `i` if ``bins[i] <= mode < bins[i+1]``.
    However, last :math:`\mu`-bin is inclusive on both ends: ``bins[-2] <= mu <= bins[-1]``.
    Therefore, with e.g. :math:`\mu`-edges ``[0.2, 0.4, 1.0]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 1.0`.
    Similarly, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 0.8`.

    Warning
    -------
    Integration over Legendre polynomials for multipoles is performed between the first and last :math:`\mu`-edges,
    e.g. with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, integration is performed between :math:`\mu = 0.2` and :math:`\mu = 0.8`.

    Parameters
    ----------
    y3d : RealField or ComplexField
        The 3D array holding the statistic to be projected to the specified basis.

    edges : list of arrays, (2,)
        List of arrays specifying the edges of the desired :math:`x` bins and :math:`\mu` bins; assumed sorted.

    los : array_like, default=(0, 0, 1)
        The line-of-sight direction to use, which :math:`\mu` is defined with respect to.

    ells : tuple of ints, default=None
        If provided, a list of integers specifying multipole numbers to project the 2D :math:`(x, \mu)` bins on to.

    Returns
    -------
    result : tuple
        The 2D binned results; a tuple of ``(xmean2d, mumean2d, y2d, n2d)``, where:

            - xmean2d : array_like, (nx, nmu)
                The mean :math:`x` value in each 2D bin
            - mumean2d : array_like, (nx, nmu)
                The mean :math:`\mu` value in each 2D bin
            - y2d : array_like, (nx, nmu)
                The mean ``y3d`` value in each 2D bin
            - n2d : array_like, (nx, nmu)
                The number of values averaged in each 2D bin

    result_poles : tuple or None
        The multipole results; if ``ells`` supplied it is a tuple of ``(xmean1d, poles, n1d)``,
        where:

            - xmean1d : array_like, (nx,)
                The mean :math:`x` value in each 1D multipole bin
            - poles : array_like, (nell, nx)
                The mean multipoles value in each 1D bin
            - n1d : array_like, (nx,)
                The number of values averaged in each 1D bin
    """
    comm = y3d.pm.comm
    x3d = y3d.x
    hermitian_symmetric = y3d.compressed
    if antisymmetric: hermitian_symmetric *= -1

    from scipy.special import legendre

    # Setup the bin edges and number of bins
    xedges, muedges = edges
    nx = len(xedges) - 1
    nmu = len(muedges) - 1
    xdtype = max(xedges.dtype, muedges.dtype)
    # Always make sure first ell value is monopole, which is just (x, mu) projection since legendre of ell = 0 is 1
    ells = ells or []
    do_poles = len(ells) > 0
    unique_ells = sorted(set([0]) | set(ells))
    legpoly = [legendre(ell) for ell in unique_ells]
    nell = len(unique_ells)

    # valid ell values
    if any(ell < 0 for ell in unique_ells):
        raise ValueError('Multipole numbers must be non-negative integers')

    # Initialize the binning arrays
    musum = np.zeros((nx+2, nmu+2))
    xsum = np.zeros((nx+2, nmu+2))
    ysum = np.zeros((nell, nx+2, nmu+2), dtype=y3d.dtype) # extra dimension for multipoles
    nsum = np.zeros((nx+2, nmu+2), dtype='i8')

    # If input array is Hermitian symmetric, only half of the last  axis is stored in `y3d`

    # Iterate over y-z planes of the coordinate mesh
    for islab in range(x3d[0].shape[0]):
        # The square of coordinate mesh norm
        # (either Fourier space k or configuraton space x)
        xvec = (x3d[0][islab].real.astype(xdtype),) + tuple(x3d[i].real.astype(xdtype) for i in range(1,3))
        xnorm = sum(xx**2 for xx in xvec)**0.5

        # If empty, do nothing
        if len(xnorm.flat) == 0: continue

        # Get the bin indices for x on the slab
        dig_x = np.digitize(xnorm.flat, xedges, right=False)

        # Get the bin indices for mu on the slab
        mu = sum(xx*ll for xx, ll in zip(xvec, los))
        nonzero = xnorm != 0.
        mu[nonzero] /= xnorm[nonzero]

        if hermitian_symmetric == 0:
            mus = [mu]
        else:
            nonsingular = np.ones(xnorm.shape, dtype='?')
            # Get the indices that have positive freq along symmetry axis = -1
            nonsingular[...] = x3d[-1][0] > 0.
            mus = [mu, -mu]

        # Accounting for negative frequencies
        for imu, mu in enumerate(mus):
            # Make the multi-index
            dig_mu = np.digitize(mu.flat, muedges, right=False) # this is bins[i-1] <= x < bins[i]
            dig_mu[mu.real.flat == muedges[-1]] = nmu # last mu inclusive

            multi_index = np.ravel_multi_index([dig_x, dig_mu], (nx+2, nmu+2))

            if hermitian_symmetric and imu:
                multi_index = multi_index[nonsingular.flat]
                xnorm = xnorm[nonsingular] # it will be recomputed
                mu = mu[nonsingular]

            # Count number of modes in each bin
            nsum.flat += np.bincount(multi_index, minlength=nsum.size)
            # Sum up x in each bin
            xsum.flat += np.bincount(multi_index, weights=xnorm.flat, minlength=nsum.size)
            # Sum up mu in each bin
            musum.flat += np.bincount(multi_index, weights=mu.flat, minlength=nsum.size)

            # Compute multipoles by weighting by Legendre(ell, mu)
            for ill, ell in enumerate(unique_ells):

                weightedy3d = (2.*ell + 1.) * legpoly[ill](mu)

                if hermitian_symmetric and imu:
                    # Weight the input 3D array by the appropriate Legendre polynomial
                    weightedy3d = hermitian_symmetric * weightedy3d * y3d[islab][nonsingular[0]].conj() # hermitian_symmetric is 1 or -1
                else:
                    weightedy3d = weightedy3d * y3d[islab, ...]

                # Sum up the weighted y in each bin
                ysum[ill,...].real.flat += np.bincount(multi_index, weights=weightedy3d.real.flat, minlength=nsum.size)
                if np.iscomplexobj(ysum):
                    ysum[ill,...].imag.flat += np.bincount(multi_index, weights=weightedy3d.imag.flat, minlength=nsum.size)

    # Sum binning arrays across all ranks
    xsum = comm.allreduce(xsum)
    musum = comm.allreduce(musum)
    ysum = comm.allreduce(ysum)
    nsum = comm.allreduce(nsum)

    # It is not clear how to proceed with beyond Nyquist frequencies
    # At Nyquist, kN = - pi * N / L (appears once in y3d.x) is the same as pi * N / L, so corresponds to mu and -mu
    # Our treatment of hermitian symmetric field would sum this frequency twice (mu and -mu)
    # But this would appear only once in uncompressed field
    # As a default, set frequencies beyond to NaN
    xmax = y3d.Nmesh * (y3d.BoxSize/2/y3d.Nmesh if isinstance(y3d, RealField) else np.pi/y3d.BoxSize)
    mask_beyond_nyq = np.flatnonzero(xedges >= np.min(xmax))
    xsum[mask_beyond_nyq] = np.nan
    musum[mask_beyond_nyq] = np.nan
    ysum[:,mask_beyond_nyq] = np.nan
    nsum[mask_beyond_nyq] = 0

    # Reshape and slice to remove out of bounds points
    sl = slice(1, -1)
    with np.errstate(invalid='ignore', divide='ignore'):

        # 2D binned results
        y2d = (ysum[0,...] / nsum)[sl, sl] # ell=0 is first index
        xmean2d  = (xsum / nsum)[sl, sl]
        mumean2d = (musum / nsum)[sl, sl]
        n2d = nsum[sl, sl]

        # 1D multipole results (summing over mu (last) axis)
        if do_poles:
            n1d = nsum[sl, sl].sum(axis=-1)
            xmean1d = xsum[sl, sl].sum(axis=-1) / n1d
            poles = ysum[:, sl, sl].sum(axis=-1) / n1d
            poles = poles[[unique_ells.index(ell) for ell in ells],...]

    # Return y(x,mu) + (possibly empty) multipoles
    result = (xmean2d, mumean2d, y2d, n2d)
    result_poles = (xmean1d, poles, n1d) if do_poles else None
    return result, result_poles


def find_unique_edges(x, x0, xmin=0., xmax=np.inf, mpicomm=mpi.COMM_WORLD):
    """
    Construct unique edges for distribution of Cartesian distances corresponding to coordinates ``x``.
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/fftpower.py.

    Parameters
    ----------
    x : list of ndim arrays
        List of ndim (broadcastable) coordinate arrays.

    x0 : array_like of shape (ndim, )
        3-vector of fundamental coordinate separation.

    xmin : float, default=0.
        Minimum separation.

    xmax : float, default=np.inf
        Maximum separation.

    mpicomm : MPI communicator, default=MPI.COMM_WORLD
        The current MPI communicator.

    Returns
    -------
    edges : array
        Edges, starting at 0, such that each bin contains a unique value of Cartesian distances.
    """
    def find_unique_local(x, x0):
        fx2 = sum(xi**2 for xi in x).ravel()
        ix2 = np.int64(fx2 / (x0.min() * 0.5) ** 2 + 0.5)
        ix2, ind = np.unique(ix2, return_index=True)
        fx = fx2[ind] ** 0.5
        return fx[(fx >= xmin) & (fx <= xmax)]

    x0 = _make_array(x0, len(x), dtype='f8')
    fx = find_unique_local(x, x0)
    if mpicomm is not None:
        fx = np.concatenate(mpicomm.allgather(fx), axis=0)
    # may have duplicates after allgather
    fx = np.unique(fx)
    fx.sort()

    # now make edges around unique coordinates
    width = np.diff(fx)
    edges = np.concatenate([[xmin], fx[1:] - width/2., [min(fx[-1] + width[-1] / 2., xmax)]], axis=0)
    return edges


def _transform_rslab(rslab, boxsize):
    # We do not use the same conventions as pmesh:
    # rslab < 0 is sent back to [boxsize/2, boxsize]
    toret = []
    for ii, rr in enumerate(rslab):
        mask = rr < 0.
        rr[mask] += boxsize[ii]
        toret.append(rr)
    return toret


class BasePowerSpectrumStatistic(BaseClass):
    """
    Base template power statistic class.
    Specific power statistic should extend this class.
    """
    name = 'base'
    _attrs = ['name', 'edges', 'modes', 'power_nonorm', 'power_direct_nonorm', 'nmodes', 'wnorm', 'shotnoise_nonorm', 'attrs']

    def __init__(self, edges, modes, power_nonorm, nmodes, wnorm=1., shotnoise_nonorm=0., power_direct_nonorm=None, attrs=None):
        r"""
        Initialize :class:`BasePowerSpectrumStatistic`.

        Parameters
        ----------
        edges : tuple of ndim arrays
            Edges used to bin power spectrum measurement.

        modes : array
            Mean "wavevector" (e.g. :math:`(k, \mu)`) in each bin.

        power_nonorm : array
            Power spectrum in each bin, *without* normalization.

        nmodes : array
            Number of modes in each bin.

        wnorm : float, default=1.
            Power spectrum normalization.

        shotnoise_nonorm : float, default=0.
            Shot noise, *without* normalization.

        attrs : dict, default=None
            Dictionary of other attributes.
        """
        if np.ndim(edges[0]) == 0: edges = (edges,)
        if np.ndim(modes[0]) == 0: modes = (modes,)
        self.edges = list(np.asarray(edge) for edge in edges)
        self.modes = list(np.asarray(mode) for mode in modes)
        self.power_nonorm = np.asarray(power_nonorm)
        self.power_direct_nonorm = power_direct_nonorm
        if power_direct_nonorm is None:
            self.power_direct_nonorm = np.zeros_like(self.power_nonorm)
        else:
            self.power_direct_nonorm = np.asarray(power_direct_nonorm)
        self.nmodes = np.asarray(nmodes)
        self.wnorm = wnorm
        self.shotnoise_nonorm = shotnoise_nonorm
        self.attrs = attrs or {}

    @property
    def power(self):
        """Power spectrum, normalized and with shot noise removed."""
        return (self.power_nonorm + self.power_direct_nonorm - self.shotnoise_nonorm) / self.wnorm

    @property
    def shotnoise(self):
        """Normalized shot noise."""
        return self.shotnoise_nonorm / self.wnorm

    @property
    def k(self):
        """Wavenumbers."""
        return self.modes[0]

    @property
    def kedges(self):
        """Wavenumber edges."""
        return self.edges[0]

    @property
    def shape(self):
        """Return shape of binned power spectrum :attr:`power`."""
        return tuple(len(edges) - 1 for edges in self.edges)

    @property
    def ndim(self):
        """Return binning dimensionality."""
        return len(self.edges)

    def __call__(self):
        """Return :attr:`power`."""
        return self.power

    def rebin(self, factor=1):
        """
        Rebin power spectrum estimation, by factor(s) ``factor``.
        A tuple must be provided in case :attr:`ndim` is greater than 1.
        Input factors must divide :attr:`shape`.
        """
        if np.ndim(factor) == 0:
            factor = (factor,)
        if len(factor) != self.ndim:
            raise ValueError('Provide a rebinning factor for each dimension')
        if any(s % f for s,f in zip(self.shape, factor)):
            raise ValueError('Rebinning factor must divide shape')
        new_shape = tuple(s//f for s,f in zip(self.shape, factor))
        nmodes = self.nmodes
        self.nmodes = utils.rebin(nmodes, new_shape, statistic=np.sum)
        self.modes = [utils.rebin(m*nmodes, new_shape, statistic=np.sum)/self.nmodes for m in self.modes]
        extradim = self.power_nonorm.ndim > len(self.shape) # e.g. multipoles
        self.power_nonorm.shape = (-1,) + self.shape
        self.power_nonorm = np.asarray([utils.rebin(power*nmodes, new_shape, statistic=np.sum)/self.nmodes for power in self.power_nonorm])
        self.power_direct_nonorm.shape = (-1,) + self.shape
        self.power_direct_nonorm = np.asarray([utils.rebin(power, new_shape, statistic=np.sum)/np.prod(factor) for power in self.power_direct_nonorm])
        self.edges = [edges[::f] for edges, f in zip(self.edges, factor)]
        self.power_nonorm.shape = (-1,)*extradim + self.shape
        self.power_direct_nonorm.shape = (-1,)*extradim + self.shape

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __copy__(self):
        new = super(BasePowerSpectrumStatistic, self).__copy__()
        for name in ['edges', 'modes', 'attrs']:
            setattr(new, name, getattr(new, name).copy())
        return new

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)


def get_power_statistic(statistic='wedge'):
    """Return :class:`BasePowerSpectrumStatistic` subclass corresponding to ``statistic`` (either 'wedge' or 'multipole')."""
    if statistic == 'wedge':
        return PowerSpectrumWedge
    if statistic == 'multipole':
        return PowerSpectrumMultipole
    return BasePowerSpectrumStatistic


class MetaPowerSpectrumStatistic(type(BaseClass)):

    """Metaclass to return correct power spectrum statistic."""

    def __call__(cls, *args, statistic='wedge', **kwargs):
        return get_power_statistic(statistic=statistic)(*args, **kwargs)


class PowerSpectrumStatistic(BaseClass, metaclass=MetaPowerSpectrumStatistic):

    """Entry point to power spectrum statistics."""

    @classmethod
    def from_state(cls, state):
        state = state.copy()
        name = state.pop('name')
        return get_power_statistic(statistic=name).from_state(state)


class PowerSpectrumWedge(BasePowerSpectrumStatistic):

    r"""Power spectrum binned in :math:`(k, \mu)`."""

    name = 'wedge'

    @property
    def kavg(self):
        """Mode-weighted average wavenumber."""
        return np.nansum(self.k*self.nmodes, axis=1)/np.sum(self.nmodes, axis=1)

    @property
    def mu(self):
        """Cosine angle to line-of-sight."""
        return self.modes[1]

    @property
    def muavg(self):
        r"""Mode-weighted average :math:`\mu`."""
        return np.nansum(self.mu*self.nmodes, axis=0)/np.sum(self.nmodes, axis=0)

    @property
    def muedges(self):
        r""":math:`\mu`-edges."""
        return self.edges[1]

    def __call__(self, k=None, mu=None, complex=True):
        r"""
        Return :attr:`power` (shot noise removed), optionally performing linear interpolation over :math:`k` and :math:`\mu`.

        Parameters
        ----------
        k : float, array, default=None
            :math:`k` where to interpolate the power spectrum.
            Values outside :attr:`kavg` are set to the first/last power value;
            outside :attr:`edges[0]` to nan.
            Defaults to :attr:`kavg`.

        mu : float, array, default=None
            :math:`\mu` where to interpolate the power spectrum.
            Values outside :attr:`muavg` are set to the first/last power value;
            outside :attr:`edges[1]` to nan.
            Defaults to :attr:`muavg`.

        complex : bool, default=True
            Whether (``True``) to return the complex power spectrum,
            or (``False``) return its real part only.

        Returns
        -------
        toret : array
            (Optionally interpolated) power spectrum.
        """
        tmp = self.power
        if not complex and np.iscomplexobj(tmp): tmp = tmp.real
        if k is None and mu is None:
            return tmp
        kavg, muavg = self.kavg, self.muavg
        if k is None: k = kavg
        if mu is None: mu = muavg
        mask_finite_k, mask_finite_mu = ~np.isnan(kavg), ~np.isnan(muavg)
        kavg, muavg, tmp = kavg[mask_finite_k], muavg[mask_finite_mu], tmp[np.ix_(mask_finite_k, mask_finite_mu)]
        k, mu = np.asarray(k), np.asarray(mu)
        isscalar = k.ndim == 0 or mu.ndim == 0
        k, mu = np.atleast_1d(k), np.atleast_1d(mu)
        toret = np.nan * np.zeros((k.size, mu.size), dtype=tmp.dtype)
        mask_k = (k >= self.edges[0][0]) & (k <= self.edges[0][-1])
        mask_mu = (mu >= self.edges[1][0]) & (mu <= self.edges[1][-1])
        if mask_k.any() and mask_mu.any():
            if muavg.size == 1:
                interp = lambda array: UnivariateSpline(kavg, array, k=1, ext=3)(k[mask_k])[:, None]
            else:
                interp = lambda array: RectBivariateSpline(kavg, muavg, array, kx=1, ky=1, s=0)(k[mask_k], mu[mask_mu], grid=True)
            toret[np.ix_(mask_k, mask_mu)] = interp(tmp.real)
            if complex and np.iscomplexobj(tmp):
                toret[np.ix_(mask_k, mask_mu)] += 1j * interp(tmp.imag)
        if isscalar:
            return toret.ravel()
        return toret


class PowerSpectrumMultipole(BasePowerSpectrumStatistic):

    """Power spectrum multipoles binned in :math:`k`."""

    name = 'multipole'
    _attrs = BasePowerSpectrumStatistic._attrs + ['ells']

    def __init__(self, edges, modes, power_nonorm, nmodes, ells, **kwargs):
        r"""
        Initialize :class:`PowerSpectrumMultipole`.

        Parameters
        ----------
        edges : tuple of ndim arrays
            Edges used to bin power spectrum measurement.

        modes : array
            Mean "wavevector" (e.g. :math:`(k, \mu)`) in each bin.

        power_nonorm : array
            Power spectrum in each bin, *without* normalization.

        nmodes : array
            Number of modes in each bin.

        ells : tuple, list.
            Multipole orders.

        kwargs : dict
            Other arguments for :attr:`BasePowerSpectrumStatistic`.
        """
        self.ells = tuple(ells)
        super(PowerSpectrumMultipole, self).__init__(edges, modes, power_nonorm, nmodes, **kwargs)

    @property
    def kavg(self):
        """Mode-weighted average wavenumber = :attr:`k`."""
        return self.k

    @property
    def power(self):
        """Power spectrum, normalized and with shot noise removed from monopole."""
        power = (self.power_nonorm + self.power_direct_nonorm) / self.wnorm
        if 0 in self.ells:
            power[self.ells.index(0)] -= self.shotnoise
        return power

    def __call__(self, ell=None,  k=None, complex=True):
        r"""
        Return :attr:`power` (shot noise removed), optionally performing linear interpolation over :math:`k`.

        Parameters
        ----------
        ell : int, default=None
            Multipole order. Defaults to all multipoles.

        k : float, array, default=None
            :math:`k` where to interpolate the power spectrum.
            Values outside :attr:`kavg` are set to the first/last power value;
            outside :attr:`edges[0]` to nan.
            Defaults to :attr:`kavg` (no interpolation performed).

        complex : bool, default=True
            Whether (``True``) to return the complex power spectrum,
            or (``False``) return its real part only if ``ell`` is even, imaginary part if ``ell`` is odd.

        Returns
        -------
        toret : array
            (Optionally interpolated) power spectrum.
        """
        if ell is None:
            return np.array([self(ell=ell, k=k, complex=complex) for ell in self.ells])
        tmp = self.power[self.ells.index(ell)]
        if not complex and np.iscomplexobj(tmp): tmp = tmp.real if ell % 2 == 0 else tmp.imag
        if k is None:
            return tmp
        kavg = self.k
        mask_finite_k = ~np.isnan(kavg)
        kavg, tmp = kavg[mask_finite_k], tmp[mask_finite_k]
        k = np.asarray(k)
        toret = np.nan * np.zeros(k.shape, dtype=tmp.dtype)
        mask_k = (k >= self.edges[0][0]) & (k <= self.edges[0][-1])
        if mask_k.any():
            interp = lambda array: UnivariateSpline(kavg, array, k=1, ext=3)(k[mask_k])
            toret[mask_k] = interp(tmp.real)
            if complex and np.iscomplexobj(tmp): toret[mask_k] = toret[mask_k] + 1j * interp(tmp.imag)
        return toret


def normalization_from_nbar(nbar, weights=None, data_weights=None, mpicomm=mpi.COMM_WORLD):
    r"""
    Return BOSS/eBOSS-like normalization, summing over :math:`\bar{n}_{i}` and weight columns, i.e.:

    .. math::
        \alpha \sum_{i=0}^{N} w_{i} \bar{n}_{i}

    Parameters
    ----------
    nbar : array of shape (N,)
        :math:`\bar{n}_{i}` (comoving density) column.

    weights : array of shape (N,), default=None
        Weights, if any.

    data_weights : array of shape (N,), default=None
        Data weights, to normalize randoms ``weights`` by ``alpha = sum(data_weights)/sum(weights)``.

    mpicomm : MPI communicator, default=MPI.COMM_WORLD
        The MPI communicator.

    Returns
    -------
    norm : float
        Normalization.
    """
    if weights is None:
        weights = np.ones_like(nbar)
    if data_weights is not None:
        sum_data_weights = mpicomm.allreduce(np.sum(data_weights))
        sum_weights = mpicomm.allreduce(np.sum(weights))
        alpha = sum_data_weights/sum_weights
    else:
        alpha = 1.
    toret = mpicomm.allreduce(alpha * np.sum(nbar * weights))
    return toret


def normalization(mesh1, mesh2=None, uniform=False, resampler='cic', cellsize=10.):
    r"""
    Return DESI-like normalization, summing over mesh cells:

    .. math::

        A = dV \frac{\alpha_{2} \sum_{i} n_{d,1}^{i} n_{r,2}^{i} + \alpha_{1} \sum_{i} n_{d,2}^{i} n_{r,1}^{i}}{2}

    :math:`n_{d,1}^{i}` and :math:`n_{r,1}^{i}` are the first data and randoms density meshes, as obtained by
    painting data :math:`w_{d}` and random weights :math:`w_{r}` on the same mesh (of cell volume :math:`dV`),
    using the cic assignment scheme. The sum then runs over the mesh cells.
    :math:`\alpha_{1} = \sum_{i} w_{d,1}^{i} / \sum_{i} w_{r,1}^{i}` and :math:`\alpha_{2} = \sum_{i} w_{d,2}^{i} / \sum_{i} w_{r,2}^{i}`
    where the sum of weights is performed over the catalogs.
    If no randoms are provided, density is supposed to be uniform and ``mesh1`` and ``mesh2`` are assumed to occupy the same physical volume.

    Parameters
    ----------
    mesh1 : CatalogMesh, RealField
        First mesh. If :class:`RealField`, density is assumed to be uniform, ``mesh1.csum()/np.prod(mesh1.pm.BoxSize)``.

    mesh2 : CatalogMesh, RealField, default=None
        Second mesh, for cross-correlations.

    uniform : bool, default=False
        Whether to assume uniform selection function (only revelant when both ``mesh1`` and ``mesh2`` are :class:`CatalogMesh`).

    resampler : string, ResampleWindow, default='cic'
        Particle-mesh assignment scheme. Choices are ['ngp', 'cic', 'tsc', 'pcs'].

    cellsize : array, float
        Physical size of mesh cells used to paint ``mesh1`` and ``mesh2`` (if instance of :class:`CatalogMesh`).

    Returns
    -------
    norm : float
        Normalization.
    """
    if mesh2 is None: mesh2 = mesh1
    autocorr = mesh2 is mesh1

    if (not uniform) and isinstance(mesh1, CatalogMesh) and isinstance(mesh2, CatalogMesh):

        # If one of input meshes do not have randoms, assume uniform density (and same volume)
        if (not mesh1.with_randoms) or (not mesh2.with_randoms):
            return (mesh1.sum_data_weights * mesh2.sum_data_weights) / np.prod(mesh1.boxsize)

        # Make sure to put mesh1 and mesh2 on the same mesh
        def get_positions(mesh):
            positions = [mesh.data_positions]
            if mesh.with_randoms: positions += [mesh.randoms_positions]
            return positions

        positions = get_positions(mesh1)
        if not autocorr: positions += get_positions(mesh2)
        # Determine bounding box
        nmesh, boxsize, boxcenter = _get_box(cellsize=cellsize, positions=positions, boxpad=1.1, mpicomm=mesh1.mpicomm)
        nmesh += 1
        boxsize = nmesh*cellsize # enforce exact cellsize
        cellsize = boxsize/nmesh # just to get correct shape

        # Assign positions/weights to mesh
        def get_mesh_nbar(mesh, field='data'):
            if field == 'data':
                mesh = mesh.clone(data_positions=mesh.data_positions, data_weights=mesh.data_weights,
                                  nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, resampler=resampler, interlacing=False, position_type='pos').to_mesh()
            else:
                mesh = mesh.clone(data_positions=mesh.randoms_positions, data_weights=mesh.sum_data_weights/mesh.sum_randoms_weights*mesh.randoms_weights,
                                  nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, resampler=resampler, interlacing=False, position_type='pos').to_mesh()
            return mesh

        # Sum over meshes
        toret = (get_mesh_nbar(mesh1, field='data') * get_mesh_nbar(mesh2, field='randoms')).csum()
        if not autocorr:
            toret = (toret + (get_mesh_nbar(mesh2, field='data') * get_mesh_nbar(mesh1, field='randoms')).csum())/2.
        # Meshes are in "weights units" (1/dV missing in each mesh), so multiply by dV * (1/dV)^2
        toret /= np.prod(cellsize)
        return toret

    # One of input meshes does not come from a catalog; assume uniform density (and same volume)
    if isinstance(mesh1, CatalogMesh):
        s1 = mesh1.sum_data_weights
        boxsize = mesh1.boxsize
    else:
        s1 = mesh1.csum()
        boxsize = mesh1.pm.BoxSize
    if autocorr:
        s2 = s1
    else:
        if isinstance(mesh2, CatalogMesh):
            s2 = mesh2.sum_data_weights
        else:
            s2 = mesh2.csum()

    return (s1 * s2) / np.prod(boxsize)



class MeshFFTPower(BaseClass):
    """
    Class that computes power spectrum from input mesh(es), using global or local line-of-sight, following https://arxiv.org/abs/1704.02357.
    In effect, this class merges nbodykit's implementation of the global line-of-sight (periodic) algorithm of:
    https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/fftpower.py
    with the local line-of-sight algorithm of:
    https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py

    Attributes
    ----------
    poles : PowerSpectrumMultipole
        Estimated power spectrum multipoles.

    wedges : PowerSpectrumWedge
        Estimated power spectrum wedges (if relevant).
    """

    def __init__(self, mesh1, mesh2=None, edges=None, ells=(0, 2, 4), los='firstpoint', boxcenter=None, compensations=None, wnorm=None, shotnoise=None):
        r"""
        Initialize :class:`MeshFFTPower`, i.e. estimate power spectrum.

        Warning
        -------
        In case line-of-sight is not local, one can provide :math:`\mu`-edges. In this case, integration over Legendre polynomials for multipoles
        is performed between the first and last :math:`\mu`-edges.
        For example, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, integration is performed between :math:`\mu = 0.2` and :math:`\mu = 0.8`.
        In all other cases, integration is performed between :math:`\mu = -1.0` and :math:`\mu = 1.0`.

        Parameters
        ----------
        mesh1 : CatalogMesh, RealField
            First mesh.

        mesh2 : CatalogMesh, RealField, default=None
            In case of cross-correlation, second mesh, with same size and physical extent (``boxsize`` and ``boxcenter``) that ``mesh1``.

        edges : tuple, array, default=None
            If ``los`` is local (``None``), :math:`k`-edges for :attr:`poles`.
            Else, one can also provide :math:`\mu`-edges (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            If ``kedges`` is ``None``, defaults to edges containing unique :math:`k` (norm) values, see :func:`find_unique_edges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'dk' (in which case :func:`find_unique_edges` is used to find unique :math:`k` (norm) values).
            For both :math:`k` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end, i.e. ``bins[i] <= x < bins[i+1]``.
            However, last :math:`\mu`-bin is inclusive on both ends: ``bins[-2] <= mu <= bins[-1]``.
            Therefore, with e.g. :math:`\mu`-edges ``[0.2, 0.4, 1.0]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 1.0`.
            Similarly, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 0.8`.

        ells : list, tuple, default=(0, 2, 4)
            Multipole orders.

        los : string, array, default='firstpoint'
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
            Power spectrum normalization, to use instead of internal estimate obtained with :func:`normalization`.

        shotnoise : float, default=None
            Power spectrum shot noise, to use instead of internal estimate, which is 0 in case of cross-correlation
            or both ``mesh1`` and ``mesh2`` are :class:`pmesh.pm.RealField`,
            and in case of auto-correlation is obtained by dividing :meth:`CatalogMesh.unnormalized_shotnoise`
            of ``mesh1`` by power spectrum normalization.
        """
        self._set_compensations(compensations)
        self._set_los(los)
        self._set_ells(ells)
        self._set_mesh(mesh1, mesh2=mesh2, boxcenter=boxcenter)
        self._set_edges(edges)
        self.wnorm = wnorm
        if wnorm is None:
            self.wnorm = np.real(normalization(mesh1, mesh2))
        self.shotnoise = shotnoise
        if shotnoise is None:
            self.shotnoise = 0.
            # Shot noise is non zero only if we can estimate it
            if self.autocorr and isinstance(mesh1, CatalogMesh):
                self.shotnoise = mesh1.unnormalized_shotnoise()/self.wnorm
        self.attrs.update(self._get_attrs())
        if self.mpicomm.rank == 0:
            self.log_info('Running power spectrum estimation.')
        self.run()

    def _set_compensations(self, compensations):
        # Set :attr:`compensations`
        if compensations is None: compensations = [None]*2
        if not isinstance(compensations, (tuple, list)):
            compensations = [compensations]*2
        compensations = compensations.copy()
        compensations += [None]*(2 - len(compensations))

        def _format_compensation(compensation):
            if compensation is None: return None
            if isinstance(compensation, dict):
                return compensation
            resampler = None
            for name in ['ngp', 'cic', 'tsc', 'pcs']:
                if name in compensation:
                    resampler = name
            if resampler is None:
                raise ValueError('Specify resampler in compensation')
            shotnoise = 'shotnoise' in compensation or 'sn' in compensation
            return {'resampler':resampler, 'shotnoise':shotnoise}

        self.compensations = [_format_compensation(compensation) for compensation in compensations]

    def _set_mesh(self, mesh1, mesh2=None, boxcenter=None):
        self.mesh1 = mesh1
        self.mesh2 = mesh2
        self.autocorr = mesh2 is None or mesh2 is mesh1
        self.attrs = {}

        for i in range(1 if self.autocorr else 2):
            name = 'mesh{:d}'.format(i+1)
            mesh = locals()[name]
            if isinstance(mesh, CatalogMesh):
                if mesh.mpicomm.rank == 0:
                    self.log_info('Painting catalog {:d} to mesh {}.'.format(i+1, str(mesh)))
                setattr(self, name, mesh.to_mesh())
                if mesh.mpicomm.rank == 0:
                    self.log_info('Done painting catalog {:d} to mesh.'.format(i+1))
                if boxcenter is not None:
                    if not np.allclose(boxcenter, mesh.boxcenter):
                        self.log_warning('Provided boxcenter is not the same as that of provided {} instance'.format(mesh.__class__.__name__))
                else:
                    boxcenter = mesh.boxcenter
                compensation = mesh.compensation
                if self.compensations[i] is not None:
                    if self.compensations[i] != compensation:
                        self.log_warning('Provided compensation is not the same as that of provided {} instance'.format(mesh.__class__.__name__))
                else:
                    self.compensations[i] = compensation
                self.attrs['sum_data_weights{:d}'.format(i+1)] = mesh.sum_data_weights
                self.attrs['sum_randoms_weights{:d}'.format(i+1)] = mesh.sum_randoms_weights
                self.attrs['resampler{:d}'.format(i+1)] = mesh.compensation['resampler']
                self.attrs['interlacing{:d}'.format(i+1)] = mesh.interlacing
            else:
                setattr(self, name, mesh)
                self.attrs['sum_data_weights{:d}'.format(i+1)] = self.attrs['sum_randoms_weights{:d}'.format(i+1)] = mesh.csum().real
                self.attrs['resampler{:d}'.format(i+1)] = not self.compensations[i]['resampler'] if self.compensations[i] is not None else None
                self.attrs['interlacing{:d}'.format(i+1)] = not self.compensations[i]['shotnoise'] if self.compensations[i] is not None else False

        if self.autocorr:
            for name in ['sum_data_weights', 'sum_randoms_weights', 'resampler', 'interlacing']:
                self.attrs['{}2'.format(name)] = self.attrs['{}1'.format(name)]

        if self.autocorr:
            self.mesh2 = self.mesh1
        self.boxcenter = _make_array(boxcenter if boxcenter is not None else 0., 3, dtype='f8')

        if isinstance(mesh1, CatalogMesh) and isinstance(mesh2, CatalogMesh):
            if not np.allclose(mesh1.boxcenter, mesh2.boxcenter):
                raise ValueError('Mismatch in input box centers')
        if not np.allclose(self.mesh1.pm.BoxSize, self.mesh2.pm.BoxSize):
            raise ValueError('Mismatch in input box sizes')
        if not np.allclose(self.mesh1.pm.Nmesh, self.mesh2.pm.Nmesh):
            raise ValueError('Mismatch in input mesh sizes')
        if self.mesh2.pm.comm is not self.mesh1.pm.comm:
            raise ValueError('Communicator mismatch between input meshes')
        self.pm = self.mesh1.pm
        if np.any(self.nmesh % 2):
            raise NotImplementedError('For odd sizes pmesh k-coordinates are not wrapped in [-knyq, knyq]; please use even sizes for now')

    def _set_edges(self, edges):
        # Set :attr:`edges`
        if edges is None or isinstance(edges, dict) or (not isinstance(edges[0], dict) and np.ndim(edges[0]) == 0):
            edges = (edges,)
        if len(edges) == 1:
            kedges, muedges = edges[0], None
        else:
            kedges, muedges = edges
        if kedges is None:
            kedges = {}
        if isinstance(kedges, dict):
            kmin = kedges.get('min', 0.)
            kmax = kedges.get('max', np.pi/(self.boxsize/self.nmesh).max())
            dk = kedges.get('step', None)
            if dk is None:
                # find unique edges
                k = [k.real for k in self.pm.create_coords('complex')]
                dk = 2 * np.pi / self.boxsize
                kedges = find_unique_edges(k, dk, xmin=kmin, xmax=kmax*(1+1e-6), mpicomm=self.mpicomm) # margin required for float32
            else:
                kedges = np.arange(kmin, kmax*(1+1e-6), dk)
        if self.mpicomm.rank == 0:
            self.log_info('Using {:d} k-bins between {:.3f} and {:.3f}.'.format(len(kedges) - 1, kedges[0], kedges[-1]))
        if muedges is None:
            muedges = np.linspace(-1, 1, 2, endpoint=True) # single :math:`\mu`-wedge
        self.edges = (np.asarray(kedges, dtype='f8'), np.asarray(muedges, dtype='f8'))

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
        self.los_type = 'global'
        if los is None:
            self.los_type = 'firstpoint'
            self.los = None
        elif isinstance(los, str):
            los = los.lower()
            allowed_los = ['firstpoint', 'endpoint', 'x', 'y', 'z']
            if los not in allowed_los:
                raise ValueError('los should be one of {}'.format(allowed_los))
            if los in ['firstpoint', 'endpoint']:
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

    @property
    def mpicomm(self):
        """Current MPI communicator."""
        return self.pm.comm

    @property
    def boxsize(self):
        """Physical box size."""
        return self.pm.BoxSize

    @property
    def nmesh(self):
        """Mesh size."""
        return self.pm.Nmesh

    def _compensate(self, cfield, *compensations):
        if self.mpicomm.rank == 0:
            self.log_debug('Applying compensations {}.'.format(compensations))
        # Apply compensation window for particle-assignment scheme
        windows = [_get_compensation_window(**compensation) for compensation in compensations if compensation is not None]
        if not windows: return
        #from nbodykit.source.mesh.catalog import CompensateCIC
        #cfield.apply(func=CompensateCIC, kind='circular', out=Ellipsis)
        cellsize = self.boxsize/self.nmesh
        for k, slab in zip(cfield.slabs.x, cfield.slabs):
            kc = tuple(ki * ci for ki, ci in zip(k, cellsize))
            for window in windows:
                slab[...] /= window(*kc)

    def run(self):
        if self.los_type == 'global': # global (fixed) line-of-sight
            self._run_global_los()
        else: # local (varying) line-of-sight
            self._run_local_los()
        del self.mesh2, self.mesh1

    def _get_attrs(self):
        # Return some attributes, to be saved in :attr:`poles` and :attr:`wedges`
        state = {}
        for name in ['autocorr', 'nmesh', 'boxsize', 'boxcenter', 'los', 'los_type', 'compensations']:
            state[name] = getattr(self, name)
        return state

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {'attrs':self.attrs, **self._get_attrs()}
        for name in ['wedges', 'poles']:
            if hasattr(self, name):
                state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state."""
        super(MeshFFTPower, self).__setstate__(state)
        for name in ['wedges', 'poles']:
            if name in state:
                setattr(self, name, get_power_statistic(statistic=state[name].pop('name')).from_state(state[name]))

    def save(self, filename):
        """Save power to ``filename``."""
        if self.mpicomm.rank == 0:
            super(MeshFFTPower, self).save(filename)
        self.mpicomm.Barrier()

    def _run_global_los(self):

        rank = self.mpicomm.rank
        start = time.time()
        # Calculate the 3d power spectrum, slab-by-slab to save memory
        # FFT 1st density field and apply the resampler transfer kernel
        cfield2 = cfield1 = self.mesh1.r2c() # pmesh r2c convention is 1/N^3 e^{-ikr}
        #print(cfield1.value.sum(), cfield1.value.dtype, cfield1.value.shape)
        self._compensate(cfield1, self.compensations[0])

        if not self.autocorr:
            cfield2 = self.mesh2.r2c()
            self._compensate(cfield2, self.compensations[1])

        # cfield1 * cfield2.conj()
        for i, c1, c2 in zip(cfield1.slabs.i, cfield1.slabs, cfield2.slabs):
            c1[...] = c1 * c2.conj()
            mask_zero = True
            for ii in i: mask_zero = mask_zero & (ii == 0)
            c1[mask_zero] = 0.

        #from nbodykit.algorithms.fftpower import project_to_basis
        #result, result_poles = project_to_basis(cfield1, self.edges, poles=self.ells or [], los=self.los)
        result, result_poles = project_to_basis(cfield1, self.edges, ells=self.ells, los=self.los)

        stop = time.time()
        if rank == 0:
            self.log_info('Power spectrum computed in elapsed time {:.2f} s.'.format(stop - start))

        # Format the power results into :class:`PowerSpectrumWedge` instance
        kwargs = {'wnorm':self.wnorm, 'shotnoise_nonorm':self.shotnoise*self.wnorm, 'attrs':self.attrs}
        k, mu, power, nmodes = result[:4]
        # Correct pmesh convention here: assuming F(r) is real, F*(k) = 1/N^3 \sum_{r} e^{ikr} F(r)
        power = self.nmesh.prod()**2 * power.conj()
        self.wedges = PowerSpectrumWedge(modes=(k,mu), edges=self.edges, power_nonorm=power, nmodes=nmodes, **kwargs)

        if result_poles:
            # Format the power results into :class:`PolePowerSpectrum` instance
            k, power, nmodes = (np.squeeze(result_poles[ii]) for ii in [0,1,2])
            # Correct pmesh convention here: assuming F(r) is real, F*(k) = 1/N^3 \sum_{r} e^{ikr} F(r)
            power = self.nmesh.prod()**2 * power.conj()
            self.poles = PowerSpectrumMultipole(modes=k, edges=self.edges[0], power_nonorm=power, nmodes=nmodes, ells=self.ells, **kwargs)

    def _run_local_los(self):

        swap = self.los_type == 'endpoint'
        if swap: self.mesh1, self.mesh2 = self.mesh2, self.mesh1 # swap meshes + complex conjugaison at the end of run()

        rank = self.mpicomm.rank

        nonzeroells = ells = sorted(set(self.ells))
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]

        # FFT 1st density field and apply the resampler transfer kernel
        A0 = self.mesh2.r2c() # pmesh r2c convention is 1/N^3 e^{-ikr}
        # Set mean value or real field to 0
        for i, c in zip(A0.slabs.i, A0.slabs):
            mask_zero = True
            for ii in i: mask_zero = mask_zero & (ii == 0)
            c[mask_zero] = 0.

        # We will apply all compensation transfer functions to A0_1 (faster than applying to each Aell)
        compensations = [self.compensations[0]] * 2 if self.autocorr else self.compensations

        result = []
        # Loop over the higher order multipoles (ell > 0)
        start = time.time()

        if self.autocorr:
            if nonzeroells:
                # Higher-order multipole requested
                # If monopole requested, copy A0 without window in Aell
                if 0 in self.ells: Aell = A0.copy()
                self._compensate(A0, *compensations)
            else:
                # In case of autocorrelation, and only monopole requested, no A0_1 copy need be made
                # Apply a single window, which will be squared by the autocorrelation
                if 0 in self.ells: Aell = A0
                self._compensate(A0, compensations[0])
        else:
            # Cross-correlation, all windows on A0
            if 0 in self.ells: Aell = self.mesh1.r2c()
            self._compensate(A0, *compensations)

        if 0 in self.ells:

            for islab in range(A0.shape[0]):
                Aell[islab,...] = Aell[islab] * A0[islab].conj()

            # the 1D monopole
            #from nbodykit.algorithms.fftpower import project_to_basis
            proj_result = project_to_basis(Aell, self.edges)[0]
            result.append(np.squeeze(proj_result[2]))
            k, nmodes = proj_result[0], proj_result[-1]

            if rank == 0:
                self.log_info('ell = {:d} done; {:d} r2c completed'.format(0, 1))

        if nonzeroells:
            # Initialize the memory holding the Aell terms for
            # higher multipoles (this holds sum of m for fixed ell)
            # NOTE: this will hold FFTs of density field #1
            rfield = RealField(self.pm)
            cfield = ComplexField(self.pm)
            Aell = ComplexField(self.pm)

            # Spherical harmonic kernels (for ell > 0)
            Ylms = [[get_real_Ylm(ell, m) for m in range(-ell, ell+1)] for ell in nonzeroells]

            # Offset the box coordinate mesh ([-BoxSize/2, BoxSize]) back to the original (x,y,z) coords
            offset = self.boxcenter - self.boxsize/2.
            # NOTE: we do not apply half cell shift as in nbodykit below
            #offset = self.boxcenter - self.boxsize/2. + 0.5*self.boxsize / self.nmesh # in nbodykit
            #offset = self.boxcenter + 0.5*self.boxsize / self.nmesh # in nbodykit

            def _save_divide(num, denom):
                with np.errstate(divide='ignore', invalid='ignore'):
                    toret = num/denom
                toret[denom == 0.] = 0.
                return toret

            # The real-space grid
            xhat = [xx.real.astype('f8') + offset[ii] for ii, xx in enumerate(_transform_rslab(self.mesh1.slabs.optx, self.boxsize))]
            #xhat = [xx.astype('f8') + offset[ii] for ii, xx in enumerate(self.mesh1.slabs.optx)]
            xnorm = np.sqrt(sum(xx**2 for xx in xhat))
            xhat = [_save_divide(xx, xnorm) for xx in xhat]
            del xnorm

            # The Fourier-space grid
            khat = [kk.real.astype('f8') for kk in A0.slabs.optx]
            knorm = np.sqrt(sum(kk**2 for kk in khat))
            khat = [_save_divide(kk, knorm) for kk in khat]
            del knorm

        for ill, ell in enumerate(nonzeroells):

            Aell[:] = 0.
            # Iterate from m=-ell to m=ell and apply Ylm
            substart = time.time()
            for Ylm in Ylms[ill]:
                # Reset the real-space mesh to the original density #1
                rfield[:] = self.mesh1[:]

                # Apply the config-space Ylm
                for islab, slab in enumerate(rfield.slabs):
                    slab[:] *= Ylm(xhat[0][islab], xhat[1][islab], xhat[2][islab])

                # Real to complex of field #2
                rfield.r2c(out=cfield)

                # Apply the Fourier-space Ylm
                for islab, slab in enumerate(cfield.slabs):
                    slab[:] *= Ylm(khat[0][islab], khat[1][islab], khat[2][islab])

                # Add to the total sum
                Aell[:] += cfield[:]

                # And this contribution to the total sum
                substop = time.time()
                if rank == 0:
                    self.log_debug('Done term for Y(l={:d}, m={:d}) in {:.2f} s.'.format(Ylm.l, Ylm.m, substop - substart))

            if rank == 0:
                self.log_info('ell = {:d} done; {:d} r2c completed'.format(ell, len(Ylms[ill])))

            # Calculate the power spectrum multipoles, slab-by-slab to save memory
            # This computes (Aell of field #1) * (A0 of field #2).conj()
            for islab in range(A0.shape[0]):
                Aell[islab,...] = Aell[islab] * A0[islab].conj()

            # Project on to 1d k-basis (averaging over mu=[-1,1])
            proj_result = project_to_basis(Aell, self.edges, antisymmetric=bool(ell % 2))[0]
            result.append(4 * np.pi * np.squeeze(proj_result[2]))
            k, nmodes = proj_result[0], proj_result[-1]

        stop = time.time()
        if rank == 0:
            self.log_info('Power spectrum computed in elapsed time {:.2f} s.'.format(stop - start))
        # pmesh convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r)
        # Correct pmesh convention here: assuming F(r) is real, F*(k) = 1/N^3 \sum_{r} e^{ikr} F(r)
        poles = self.nmesh.prod()**2 * np.array([result[ells.index(ell)] for ell in self.ells]).conj()
        if swap: poles = poles.conj()
        # Format the power results into :class:`PolePowerSpectrum` instance
        k, nmodes = np.squeeze(k), np.squeeze(nmodes)
        kwargs = {'wnorm':self.wnorm, 'shotnoise_nonorm':self.shotnoise*self.wnorm, 'attrs':self.attrs}
        self.poles = PowerSpectrumMultipole(modes=k, edges=self.edges[0], power_nonorm=poles, nmodes=nmodes, ells=self.ells, **kwargs)

        if swap: self.mesh1, self.mesh2 = self.mesh2, self.mesh1


class CatalogFFTPower(MeshFFTPower):

    """Wrapper on :class:`MeshFFTPower` to estimate power spectrum directly from positions and weights."""

    def __init__(self, data_positions1, data_positions2=None, randoms_positions1=None, randoms_positions2=None,
                shifted_positions1=None, shifted_positions2=None,
                data_weights1=None, data_weights2=None, randoms_weights1=None, randoms_weights2=None,
                shifted_weights1=None, shifted_weights2=None,
                D1D2_twopoint_weights=None, D1R2_twopoint_weights=None, D2R1_twopoint_weights=None, D1S2_twopoint_weights=None, D2S1_twopoint_weights=None,
                edges=None, ells=(0, 2, 4), los=None,
                nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=2., dtype='f8',
                resampler='cic', interlacing=2, position_type='xyz', weight_type='auto', weight_attrs=None,
                direct_engine='kdtree', direct_limits=(0., 2./60.), direct_limit_type='degree', periodic=False,
                wnorm=None, shotnoise=None, mpiroot=None, mpicomm=mpi.COMM_WORLD):
        r"""
        Initialize :class:`CatalogFFTPower`, i.e. estimate power spectrum.

        Warning
        -------
        In case line-of-sight is not local, one can provide :math:`\mu`-edges. In this case, integration over Legendre polynomials for multipoles
        is performed between the first and last :math:`\mu`-edges.
        For example, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, integration is performed between :math:`\mu = 0.2` and :math:`\mu = 0.8`.
        In all other cases, integration is performed between :math:`\mu = -1.0` and :math:`\mu = 1.0`.

        Note
        ----
        When running with MPI, input positions and weights are assumed to be scatted on all MPI ranks of ``mpicomm``.
        If this is not the case, use :func:`mpi.scatter_array`.

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
            If ``los`` is local (``None``), :math:`k`-edges for :attr:`poles`.
            Else, one can also provide :math:`\mu`-edges (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            If ``kedges`` is ``None``, defaults to edges containing unique :math:`k` (norm) values, see :func:`find_unique_edges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'dk' (in which case :func:`find_unique_edges` is used to find unique :math:`k` (norm) values).
            For both :math:`k` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end, i.e. ``bins[i] <= x < bins[i+1]``.
            However, last :math:`\mu`-bin is inclusive on both ends: ``bins[-2] <= mu <= bins[-1]``.
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

        boxsize : float, default=None
            Physical size of the box, defaults to maximum extent taken by all input positions, times ``boxpad``.

        boxcenter : array, float, default=None
            Box center, defaults to center of the Cartesian box enclosing all input positions.

        cellsize : array, float, default=None
            Physical size of mesh cells.
            If not ``None``, and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` as ``nmesh * cellsize``.
            If ``nmesh`` is ``None``, it is set as (the nearest integer(s) to) ``boxsize/cellsize``.

        boxpad : float, default=2.
            When ``boxsize`` is determined from input positions, take ``boxpad`` times the smallest box enclosing positions as ``boxsize``.

        dtype : string, dtype, default='f8'
            The data type to use for input positions and weights and the mesh.

        resampler : string, ResampleWindow, default='cic'
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

        weight_attrs : dict, default=None
            Dictionary of weighting scheme attributes. In case ``weight_type`` is "inverse_bitwise",
            one can provide "nrealizations", the total number of realizations (*including* current one;
            defaulting to the number of bits in input weights plus one);
            "noffset", the offset to be added to the bitwise counts in the denominator (defaulting to 1)
            and "default_value", the default value of pairwise weights if the denominator is zero (defaulting to 0).
            Inverse probability weight is then computed as: :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2}))`.
            For example, for the "zero-truncated" estimator (arXiv:1912.08803), one would use noffset = 0.

        D1D2_twopoint_weights : WeightTwoPointEstimator, default=None
            Weights to be applied to each pair of particles between first and second data catalogs.
            A :class:`WeightTwoPointEstimator` instance (from *pycorr*) or any object with arrays ``sep``
            (separations) and ``weight`` (weight at given separation) as attributes
            (i.e. to be accessed through ``twopoint_weights.sep``, ``twopoint_weights.weight``)
            or as keys (i.e. ``twopoint_weights['sep']``, ``twopoint_weights['weight']``)
            or as element (i.e. ``sep, weight = twopoint_weights``).

        D1R2_twopoint_weights : WeightTwoPointEstimator, default=None
            Weights to be applied to each pair of particles between first data catalog and second randoms catalog.
            See ``D1D2_twopoint_weights``.

        D2R1_twopoint_weights : WeightTwoPointEstimator, default=None
            Weights to be applied to each pair of particles between second data catalog and first randoms catalog.
            See ``D1D2_twopoint_weights``.

        D1S2_twopoint_weights : WeightTwoPointEstimator, default=None
            Weights to be applied to each pair of particles between first data catalog and second shifted catalog.
            See ``D1D2_twopoint_weights``.

        D2S1_twopoint_weights : WeightTwoPointEstimator, default=None
            Weights to be applied to each pair of particles between second data catalog and first shifted catalog.
            See ``D1D2_twopoint_weights``.

        direct_engine : string, default='kdtree'
            Engine for direct power spectrum computation (if input weights are bitwise weights), one of ["kdtree"].

        direct_limits : tuple, default=(0., 2./60.)
            Limits of particle pair separations used in the direct power spectrum computation.

        direct_limit_type : string, default='degree'
            Type of ``direct_limits``; i.e. are those angular limits ("degree", "radian"), or 3D limits ("s")?

        periodic : bool, default=False
            Whether to assume periodic boundary conditions in direct power spectrum computation.

        wnorm : float, default=None
            Power spectrum normalization, to use instead of internal estimate obtained with :func:`normalization`.

        shotnoise : float, default=None
            Power spectrum shot noise, to use instead of internal estimate, which is 0 in case of cross-correlation
            and in case of auto-correlation is obtained by dividing :meth:`CatalogMesh.unnormalized_shotnoise` by power spectrum normalization.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=MPI.COMM_WORLD
            The MPI communicator.
        """
        rdtype = _get_real_dtype(dtype)
        bpositions, positions = [], {}
        for name in ['data_positions1', 'data_positions2', 'randoms_positions1', 'randoms_positions2', 'shifted_positions1', 'shifted_positions2']:
            tmp = _format_positions(locals()[name], position_type=position_type, dtype=rdtype, mpicomm=mpicomm, mpiroot=mpiroot)
            if tmp is not None:
                bpositions.append(tmp)
            label = name.replace('data_positions','D').replace('randoms_positions','R').replace('shifted_positions','S')
            positions[label] = tmp

        weight_attrs = (weight_attrs or {}).copy()
        noffset = weight_attrs.get('noffset', 1)
        default_value = weight_attrs.get('default_value', 0)
        weight_attrs.update(noffset=noffset, default_value=default_value)

        def get_nrealizations(weights):
            nrealizations = weight_attrs.get('nrealizations', None)
            if nrealizations is None: nrealizations = get_default_nrealizations(weights)
            return nrealizations

        bweights, n_bitwise_weights, weights = {}, {}, {}
        for name in ['data_weights1', 'data_weights2', 'randoms_weights1', 'randoms_weights2', 'shifted_weights1', 'shifted_weights2']:
            label = name.replace('data_weights','D').replace('randoms_weights','R').replace('shifted_weights','S')
            bweights[label], n_bitwise_weights[label] = _format_weights(locals()[name], weight_type=weight_type, dtype=rdtype, mpicomm=mpicomm, mpiroot=mpiroot)
            if n_bitwise_weights[label]:
                bitwise_weight = bweights[label][:n_bitwise_weights[label]]
                nrealizations = get_nrealizations(bitwise_weight)
                weights[label] = get_inverse_probability_weight(bitwise_weight, noffset=noffset, nrealizations=nrealizations, default_value=default_value)
                if len(bweights[label]) > n_bitwise_weights[label]:
                    weights[label] *= bweights[label][n_bitwise_weights[label]] # individual weights
            elif len(bweights[label]):
                weights[label] = bweights[label][0] # individual weights
            else:
                weights[label] = None

        with_shifted = positions['S1'] is not None
        with_randoms = positions['R1'] is not None
        autocorr = positions['D2'] is None
        if autocorr and (positions['R2'] is not None or positions['S2'] is not None):
            raise ValueError('randoms_positions2 or shifted_positions2 are provided, but not data_positions2')

        # Get box encompassing all catalogs
        nmesh, boxsize, boxcenter = _get_box(boxsize=boxsize, cellsize=cellsize, nmesh=nmesh, boxcenter=boxcenter, positions=bpositions, boxpad=boxpad, mpicomm=mpicomm)
        if not isinstance(resampler, tuple):
            resampler = (resampler,)*2
        if not isinstance(interlacing, tuple):
            interlacing = (interlacing,)*2

        # Get catalog meshes
        def get_mesh(data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, shifted_positions=None, shifted_weights=None, **kwargs):
            return CatalogMesh(data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights,
                               shifted_positions=shifted_positions, shifted_weights=shifted_weights,
                               nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, position_type='pos', dtype=dtype, mpicomm=mpicomm, **kwargs)

        mesh1 = get_mesh(positions['D1'], data_weights=weights['D1'], randoms_positions=positions['R1'], randoms_weights=weights['R1'],
                         shifted_positions=positions['S1'], shifted_weights=weights['S1'], resampler=resampler[0], interlacing=interlacing[0])
        mesh2 = None
        if not autocorr:
            mesh2 = get_mesh(positions['D2'], data_weights=weights['D2'], randoms_positions=positions['R2'], randoms_weights=weights['R2'],
                             shifted_positions=positions['S2'], shifted_weights=weights['S2'], resampler=resampler[1], interlacing=interlacing[1])
        # Now, run power spectrum estimation
        super(CatalogFFTPower, self).__init__(mesh1=mesh1, mesh2=mesh2, edges=edges, ells=ells, los=los, wnorm=wnorm, shotnoise=shotnoise)


        if self.ells:

            twopoint_weights = {'D1D2':D1D2_twopoint_weights, 'D1R2':D1R2_twopoint_weights, 'D2R1':D2R1_twopoint_weights, 'D1S2':D1S2_twopoint_weights, 'D2S1':D2S1_twopoint_weights}

            pairs = [(1, 'D1', 'D2')]
            if with_shifted:
                S1, S2 = 'S1', 'S2'
            else:
                S1, S2 = 'R1', 'R2'
            if with_shifted or with_randoms:
                #pairs.append((1, S1, S2))
                if autocorr:
                    pairs.append((-2, 'D1', S2))
                else:
                    pairs.append((-1, 'D1', S2))
                    pairs.append((-1, 'D2', S1))

            DirectPowerEngine = get_direct_power_engine(direct_engine)
            for coeff, label1, label2 in pairs:
                label12 = label1+label2
                if autocorr:
                    if label12 == 'D1D2': n_bitwise_weights[label2] = n_bitwise_weights[label1]
                    if label12 == 'D1R2': label2 = 'R1'
                    if label12 == 'D1S2': label2 = 'S1'
                if (n_bitwise_weights[label1] and n_bitwise_weights[label2]) or twopoint_weights[label12]:
                    if self.los_type == 'global':
                        raise NotImplementedError('mu-wedge direct computation not handled yet')
                    power = DirectPowerEngine(self.poles.k, positions[label1], weights1=bweights[label1], positions2=positions[label2], weights2=bweights[label2], ells=ells,
                                              limits=direct_limits, limit_type=direct_limit_type, weight_type='inverse_bitwise_minus_individual', twopoint_weights=twopoint_weights[label12],
                                              weight_attrs=weight_attrs, los=los, boxsize=self.boxsize if periodic else None, position_type='pos', mpicomm=self.mpicomm).power_nonorm
                    self.poles.power_direct_nonorm += coeff * power
