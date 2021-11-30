r"""
Implementation of power spectrum estimator, following https://arxiv.org/abs/1704.02357.
Apart from interface choices, differences w.r.t. original nbodykit's implementation
https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py are:

- real space positions are taken at mesh nodes, instead of 0.5 cell shift (matters only for ell > 0 in global line-of-sight)
- normalization is computed with density obtained by paintaing data/randoms to mesh, instead of relying on :math:`\bar{n}_{i}` column in the catalogs
- FKP weights are treated as other weights
"""

import time
import warnings

import numpy as np
from mpi4py import MPI

from pmesh.pm import RealField, ComplexField, ParticleMesh
from pmesh.window import FindResampler, ResampleWindow
from .utils import BaseClass
from . import mpi, utils


class FFTPowerError(Exception):

    """Error raised when issue with power spectrum computation."""


def get_real_Ylm(ell, m):
    """
    Return a function that computes the real spherical harmonic of order (ell, m).
    Mostly taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py.

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
    Ylm = sp.lambdify((xhat, yhat, zhat), expr, modules='numexpr' if numexpr is not None else 'numpy')

    # Attach some meta-data
    Ylm.expr = expr
    Ylm.l = ell
    Ylm.m = m
    return Ylm


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


def project_to_basis(y3d, edges, los=(0, 0, 1), ells=None):
    """
    Project a 3D statistic on to the specified basis. The basis will be one of:

    - 2D (`x`, `mu`) bins: `mu` is the cosine of the angle to the line-of-sight
    - 2D (`x`, `ell`) bins: `ell` is the multipole number, which specifies
      the Legendre polynomial when weighting different `mu` bins.

    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/fftpower.py.

    Notes
    -----
    The 2D (`x`, `mu`) bins will be computed only if `poles` is specified.
    See return types for further details.
    The `mu` bins (between -1 and 1) are half-inclusive half-exclusive,
    except the last bin is inclusive on both ends (to include `mu = 1.0`).

    Parameters
    ----------
    y3d : RealField or ComplexField
        The 3D array holding the statistic to be projected to the specified basis.

    edges : list of arrays, (2,)
        List of arrays specifying the edges of the desired `x` bins and `mu` bins.

    los : array_like, default=(0, 0, 1)
        The line-of-sight direction to use, which `mu` is defined with respect to.

    ells : tuple of ints, default=None
        If provided, a list of integers specifying multipole numbers to project the 2d `(x, mu)` bins on to.

    Returns
    -------
    result : tuple
        the 2D binned results; a tuple of ``(xmean2d, mumean2d, y2d, n2d)``, where:

        - xmean2d : array_like, (nx, nmu)
            the mean `x` value in each 2D bin
        - mumean2d : array_like, (nx, nmu)
            the mean `mu` value in each 2D bin
        - y2d : array_like, (nx, nmu)
            the mean `y3d` value in each 2D bin
        - n2d : array_like, (nx, nmu)
            the number of values averaged in each 2D bin

    result_poles : tuple or `None`
        the multipole results; if `ells` supplied it is a tuple of ``(xmean1d, poles, n1d)``,
        where:

        - xmean1d : array_like, (nx,)
            the mean `x` value in each 1D multipole bin
        - poles : array_like, (nell, nx)
            the mean multipoles value in each 1D bin
        - n1d : array_like, (nx,)
            the number of values averaged in each 1D bin
    """
    comm = y3d.pm.comm
    x3d = y3d.x
    hermitian_symmetric = y3d.compressed

    from scipy.special import legendre

    # setup the bin edges and number of bins
    xedges, muedges = edges
    x2edges = xedges**2
    nx = len(xedges) - 1
    nmu = len(muedges) - 1

    # always make sure first ell value is monopole, which
    # is just (x, mu) projection since legendre of ell=0 is 1
    ells = ells or []
    do_poles = len(ells) > 0
    unique_ells = sorted(set([0]) | set(ells))
    legpoly = [legendre(ell) for ell in unique_ells]
    nell = len(unique_ells)

    # valid ell values
    if any(ell < 0 for ell in unique_ells):
        raise ValueError('Multipole numbers must be non-negative integers')

    # initialize the binning arrays
    musum = np.zeros((nx+2, nmu+2))
    xsum = np.zeros((nx+2, nmu+2))
    ysum = np.zeros((nell, nx+2, nmu+2), dtype=y3d.dtype) # extra dimension for multipoles
    nsum = np.zeros((nx+2, nmu+2), dtype='i8')

    # if input array is Hermitian symmetric, only half of the last  axis is stored in `y3d`

    # iterate over y-z planes of the coordinate mesh
    for islab in range(x3d[0].shape[0]):
        # the square of coordinate mesh norm
        # (either Fourier space k or configuraton space x)
        xslab = (x3d[0][islab],) + tuple(x3d[i] for i in range(1,3))
        x2slab = sum(xx**2 for xx in xslab)

        # if empty, do nothing
        if len(x2slab.flat) == 0: continue

        # get the bin indices for x on the slab
        dig_x = np.digitize(x2slab.real.flat, x2edges)

        # get the bin indices for mu on the slab
        mu = sum(xx*ll for xx,ll in zip(xslab, los))
        xslab = x2slab**0.5
        nonzero = xslab != 0.
        mu[nonzero] /= xslab[nonzero]
        dig_mu = np.digitize(mu.real.flat, muedges)

        if hermitian_symmetric:
            # get the indices that have positive freq along symmetry axis = -1
            tmp = x3d[-1][0] > 0.
            nonsingular = np.ones(xslab.shape, dtype='?')
            nonsingular[...] = tmp
            hermitian_weights = np.ones_like(xslab.real)
            hermitian_weights[nonsingular] = 2.
        else:
            hermitian_weights = 1.
        # make the multi-index
        multi_index = np.ravel_multi_index([dig_x, dig_mu], (nx+2,nmu+2))

        # sum up x in each bin (accounting for negative freqs)
        xslab[:] *= hermitian_weights
        xsum.flat += np.bincount(multi_index, weights=xslab.real.flat, minlength=xsum.size)

        # count number of modes in each bin (accounting for negative freqs)
        nslab = np.ones_like(xslab.real) * hermitian_weights
        nsum.flat += np.bincount(multi_index, weights=nslab.flat, minlength=nsum.size)

        # compute multipoles by weighting by Legendre(ell, mu)
        for ill, ell in enumerate(unique_ells):

            # weight the input 3D array by the appropriate Legendre polynomial
            weightedy3d = legpoly[ill](mu.real) * y3d[islab,...]

            # add conjugate for this kx, ky, kz, corresponding to
            # the (-kx, -ky, -kz) --> need to make mu negative for conjugate
            # Below is identical to the sum of
            # Leg(ell)(+mu) * y3d[:, nonsingular]    (kx, ky, kz)
            # Leg(ell)(-mu) * y3d[:, nonsingular].conj()  (-kx, -ky, -kz)
            # or
            # weighted_y3d[:, nonsingular] += (-1)**ell * weighted_y3d[:, nonsingular].conj()
            # but numerically more accurate.
            if hermitian_symmetric:

                if ell % 2: # odd, real part cancels
                    weightedy3d.real[nonsingular] = 0.
                    weightedy3d.imag[nonsingular] *= 2.
                else:  # even, imag part cancels
                    weightedy3d.real[nonsingular] *= 2.
                    weightedy3d.imag[nonsingular] = 0.

            # sum up the weighted y in each bin
            weightedy3d *= (2.*ell + 1.)
            ysum[ill,...].real.flat += np.bincount(multi_index, weights=weightedy3d.real.flat, minlength=nsum.size)
            if np.iscomplexobj(ysum):
                ysum[ill,...].imag.flat += np.bincount(multi_index, weights=weightedy3d.imag.flat, minlength=nsum.size)

        # sum up the absolute mag of mu in each bin (accounting for negative freqs)
        mu[:] *= hermitian_weights
        musum.flat += np.bincount(multi_index, weights=mu.real.flat, minlength=musum.size)

    # sum binning arrays across all ranks
    xsum = comm.allreduce(xsum)
    musum = comm.allreduce(musum)
    ysum = comm.allreduce(ysum)
    nsum = comm.allreduce(nsum)

    # add the last 'internal' mu bin (mu == 1) to the last visible mu bin
    # this makes the last visible mu bin inclusive on both ends.
    ysum[..., -2] += ysum[..., -1]
    musum[:, -2] += musum[:, -1]
    xsum[:, -2] += xsum[:, -1]
    nsum[:, -2] += nsum[:, -1]

    # reshape and slice to remove out of bounds points
    sl = slice(1, -1)
    with np.errstate(invalid='ignore', divide='ignore'):

        # 2D binned results
        y2d = (ysum[0,...] / nsum)[sl,sl] # ell=0 is first index
        xmean2d  = (xsum / nsum)[sl,sl]
        mumean2d = (musum / nsum)[sl, sl]
        n2d = nsum[sl,sl]

        # 1D multipole results (summing over mu (last) axis)
        if do_poles:
            n1d = nsum[sl,sl].sum(axis=-1)
            xmean1d = xsum[sl,sl].sum(axis=-1) / n1d
            poles = ysum[:, sl,sl].sum(axis=-1) / n1d
            poles = poles[[unique_ells.index(ell) for ell in ells],...]

    # return y(x,mu) + (possibly empty) multipoles
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

    xo = _make_array(x0, len(x), dtype='f8')
    fx = find_unique_local(x, x0)
    if mpicomm is not None:
        fx = np.concatenate(mpicomm.allgather(fx), axis=0)
    # may have duplicates after allgather
    fx = np.unique(fx)
    fx.sort()

    # now make edges around unique coordinates
    width = np.diff(fx)
    edges = np.append(fx - width/2., [fx[-1] + width[-1] / 2.])
    edges[0] = 0.

    return edges


def _make_array(value, shape, dtype='f8'):
    # Return numpy array filled with value
    toret = np.empty(shape, dtype=dtype)
    toret[...] = value
    return toret


def _transform_rslab(rslab, boxsize):
    # We do not use the same conventions as pmesh:
    # rslab < 0 is sent back to [boxsize/2, boxsize]
    toret = []
    for ii,r in enumerate(rslab):
        mask = r < 0.
        r[mask] += boxsize[ii]
        toret.append(r)
    return toret


class BasePowerStatistic(BaseClass):
    """
    Base template power statistic class.
    Specific power statistic should extend this class.
    """
    name = 'base'
    _attrs = ['name', 'edges', 'modes', 'power_nonorm', 'nmodes', 'wnorm', 'shotnoise_nonorm']

    def __init__(self, edges, modes, power_nonorm, nmodes, wnorm=1., shotnoise_nonorm=0., attrs=None):
        r"""
        Initialize :class:`BasePowerStatistic`.

        Parameters
        ----------
        edges : tuple of ndim arrays
            Edges used to bin power spectrum measurement.

        modes : array
            Mean "wavevector" (e.g. :math:`(k, \mu)`) in each bin.

        power_nonorm : array
            Power spectrum in each bin, *without* normalization.

        nmodes : array
            Number of modes in each bin?

        wnorm : float, default=1.
            Power spectrum normalization.

        shotnoise_nonorm : float, default=0.
            Shot noise, *without* normalization.

        attrs : dict, default=None.
            Dictionary of other attributes.
        """
        self.edges = tuple(np.asarray(edge, dtype='f8') for edge in edges)
        self.modes = np.asarray(modes)
        self.power_nonorm = np.asarray(power_nonorm)
        self.nmodes = np.asarray(nmodes)
        self.wnorm = wnorm
        self.shotnoise_nonorm = shotnoise_nonorm
        self.attrs = attrs or {}

    @property
    def power(self):
        """Power spectrum, normalized and with shot noise removed."""
        return (self.power_nonorm - self.shotnoise_nonorm) / self.wnorm

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
        new_shape = tuple(s//f for s,f in zip(self.shape, factor))
        nmodes = self.nmodes
        self.nmodes = utils.rebin(nmodes, new_shape, statistic=np.sum)
        self.modes = [utils.rebin(m*nmodes, new_shape, statistic=np.sum)/self.nmodes for m in self.modes]
        self.power_nonorm.shape = (-1,) + self.shape
        self.power_nonorm = np.asarray([utils.rebin(power*nmodes, new_shape, statistic=np.sum)/self.nmodes for power in self.power_nonorm])
        self.edges = [edges[::f] for edges,f in zip(self.edges, factor)]

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


def get_power_statistic(statistic='wedge'):
    """Return :class:`BasePowerStatistic` subclass corresponding to ``statistic`` (either 'wedge' or 'multipole')."""
    if statistic == 'wedge':
        return WedgePowerSpectrum
    if statistic == 'multipole':
        return MultipolePowerSpectrum
    return BasePowerStatistic


class MetaPowerStatistic(type(BaseClass)):

    """Metaclass to return correct power statistic."""

    def __call__(cls, *args, statistic='wedge', **kwargs):
        return get_power_statistic(statistic=statistic)(*args, **kwargs)


class PowerStatistic(metaclass=MetaPowerStatistic):

    """Entry point to power statistics."""

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        name = state.pop('name')
        return get_power_statistic(statistic=name).from_state(state)


class WedgePowerSpectrum(BasePowerStatistic):

    r"""Power spectrum binned in :math:`(k, \mu)`."""

    name = 'wedge'

    @property
    def mu(self):
        """Cosine angle to line-of-sight of shape :attr:`shape` = (nk, nmu)."""
        return self.modes[1]

    @property
    def muedges(self):
        """Mu edges."""
        return self.edges[1]

    def __call__(self, mu=None):
        r"""Return :attr:`power`, restricted to the bin(s) corresponding to input :math:`\mu` if not ``None``."""
        if mu is not None:
            mu_indices = np.digitize(mu, self.edges[1], right=False) - 1
        else:
            mu_indices = Ellipsis
        return self.power[:,mu_indices]


class MultipolePowerSpectrum(BasePowerStatistic):

    """Power spectrum multipoles binned in :math:`k`."""

    name = 'multipole'
    _attrs = BasePowerStatistic._attrs + ['ells']

    def __init__(self, edges, modes, power_nonorm, nmodes, ells, **kwargs):
        r"""
        Initialize :class:`BasePowerStatistic`.

        Parameters
        ----------
        edges : tuple of ndim arrays
            Edges used to bin power spectrum measurement.

        modes : array
            Mean "wavevector" (e.g. :math:`(k, \mu)`) in each bin.

        power_nonorm : array
            Power spectrum in each bin, *without* normalization.

        nmodes : array
            Number of modes in each bin?

        ells : tuple, list.
            Multipole orders.

        kwargs : dict
            Other arguments for :attr:`BasePowerStatistic`.
        """
        self.ells = tuple(ells)
        super(MultipolePowerSpectrum, self).__init__((edges,), (modes,), power_nonorm, nmodes, **kwargs)

    @property
    def power(self):
        """Power spectrum, normalized and with shot noise removed from monopole."""
        power = self.power_nonorm / self.wnorm
        if 0 in self.ells:
            power[self.ells.index(0)] -= self.shotnoise
        return power

    def __call__(self, ell=None):
        r"""Return :attr:`power`, restricted to the multipole(s) corresponding to input :math:`\ell` if not ``None``."""
        isscalar = False
        if ell is not None:
            isscalar = np.ndim(ell) == 0
            if isscalar: ell = [ell]
            ills = [self.ells.index(el) for el in ell]
        else:
            ills = Ellipsis
        power = self.power[ills,:]
        if isscalar:
            return power.flatten()
        return power


def _get_resampler(resampler):
    # Return :class:`ResampleWindow` from string or :class:`ResampleWindow` instance
    if isinstance(resampler, ResampleWindow):
        return resampler
    conversions = {'ngp':'nnb', 'cic':'cic', 'tsc':'tsc', 'pcs':'pcs'}
    if resampler not in conversions:
        raise FFTPowerError('Unknown resampler {}, choices are {}'.format(resampler, list(conversions.keys())))
    resampler = conversions[resampler]
    return FindResampler(resampler)


def _get_resampler_name(resampler):
    # Translate input :class:`ResampleWindow` instance to string
    conversions = {'nearest':'ngp', 'tunedcic':'cic', 'tunedtsc':'tsc', 'tunedpcs':'pcs'}
    return conversions[resampler.kind]


def _format_positions(positions, position_type='xyz'):
    # Format input array of positions
    if position_type == 'pos': # array of shape (N, 3)
        positions = np.asarray(positions)
        if positions.shape[-1] != 3:
            raise FFTPowerError('For position type = {}, please provide a (N, 3) array for positions'.format(position_type))
        return positions
    # Array of shape (3, N)
    positions = list(positions)
    for ip, p in enumerate(positions):
        # cast to the input dtype if exists (may be set by previous weights)
        positions[ip] = np.asarray(p)
    size = len(positions[0])
    dtype = positions[0].dtype
    if not np.issubdtype(dtype, np.floating):
        raise FFTPowerError('Input position arrays should be of floating type, not {}'.format(dtype))
    for p in positions[1:]:
        if len(p) != size:
            raise FFTPowerError('All position arrays should be of the same size')
        if p.dtype != dtype:
            raise FFTPowerError('All position arrays should be of the same type, you can e.g. provide dtype')
    if len(positions) != 3:
        raise FFTPowerError('For position type = {}, please provide a list of 3 arrays for positions'.format(position_type))
    if position_type == 'rdd': # RA, Dec, distance
        positions = utils.sky_to_cartesian(positions, degree=True)
    elif position_type != 'xyz':
        raise FFTPowerError('Position type should be one of ["xyz", "rdd"]')
    return np.asarray(positions).T


def _get_box(nmesh=None, boxsize=None, boxcenter=None, cellsize=None, positions=None, boxpad=1.5, check=True, mpicomm=mpi.COMM_WORLD):
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

    boxpad : float, default=1.5
        When ``boxsize`` is determined from ``positions``, take ``boxpad`` times the smallest box enclosing ``positions`` as ``boxsize``.

    check : bool, default=True
        Whether to check input ``positions`` (if provided) are in enclosing box.

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
    if boxsize is None or boxcenter is None or (check and positions is not None):
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
        if (boxsize < delta).any():
            raise FFTPowerError('boxsize {} too small to contain all data (max {})'.format(boxsize, delta))

    if nmesh is None:
        if cellsize is not None:
            nmesh = np.rint(boxsize/cellsize).astype(int)
        else:
            raise FFTPowerError('nmesh (or cellsize) must be specified')
    nmesh = _make_array(nmesh, 3, dtype='i4')
    boxsize = _make_array(boxsize, 3, dtype='f8')
    boxcenter = _make_array(boxcenter, 3, dtype='f8')
    return nmesh, boxsize, boxcenter


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
    pm = ParticleMesh(boxsize=boxsize, Nmesh=shape, dtype=dtype, comm=mpicomm)
    mesh = pm.create(type='real')
    if mpicomm.rank == mpiroot:
        array = array.ravel() # ignore data from other ranks
    else:
        array = empty
    mesh.unravel(array)
    return mesh


class CatalogMesh(BaseClass):

    """Class to paint catalog of positions and weights to mesh."""

    def __init__(self, data_positions, data_weights=None, randoms_positions=None, randoms_weights=None,
                 nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=1.5, dtype='f8',
                 resampler='cic', interlacing=2, position_type='xyz', mpiroot=None, mpicomm=MPI.COMM_WORLD):
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

        boxpad : float, default=1.5
            When ``boxsize`` is determined from ``positions``, take ``boxpad`` times the smallest box enclosing ``positions`` as ``boxsize``.

        dtype : string, dtype, default='f8'
            The data type of the mesh when painting.

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

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scatted across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=MPI.COMM_WORLD
            The MPI communicator.
        """
        self.mpicomm = mpicomm
        self.dtype = np.dtype(dtype)
        self._set_positions(data_positions=data_positions, randoms_positions=randoms_positions, position_type=position_type, mpiroot=mpiroot)
        self._set_weights(data_weights=data_weights, randoms_weights=randoms_weights, mpiroot=mpiroot)
        self._set_box(boxsize=boxsize, cellsize=cellsize, nmesh=nmesh, boxcenter=boxcenter, boxpad=boxpad)
        self._set_resampler(resampler)
        self._set_interlacing(interlacing)

    @property
    def compensation(self):
        """Return dictionary specifying compensation scheme for particle-mesh resampling."""
        return {'resampler':_get_resampler_name(self.resampler), 'shotnoise': not bool(self.interlacing)}

    def clone(self, data_positions=None, data_weights=None, randoms_positions=None, randoms_weights=None,
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
        new.__init__(data_positions=data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights, position_type=position_type, **kwargs)
        return new

    def _set_interlacing(self, interlacing):
        self.interlacing = int(interlacing)
        if self.interlacing != interlacing:
            raise FFTPowerError('Interlacing must be either bool (False, 0) or an integer >= 2')
        if self.interlacing == 1:
            if self.mpicomm.rank == 0:
                self.log_warning('Provided interlacing is {}; setting it to 2.'.format(interlacing))
            self.interlacing = 2

    def _set_box(self, nmesh=None, boxsize=None, cellsize=None, boxcenter=None, boxpad=1.5, check=True):
        # Set :attr:`nmesh`, :attr:`boxsize` and :attr:`boxcenter`
        positions = [self.data_positions]
        if self.with_randoms: positions += [self.randoms_positions]
        self.nmesh, self.boxsize, self.boxcenter = _get_box(nmesh=nmesh, boxsize=boxsize, cellsize=cellsize, boxcenter=boxcenter,
                                                            positions=positions, boxpad=boxpad, check=check, mpicomm=self.mpicomm)

    def _set_positions(self, data_positions, randoms_positions=None, position_type='xyz', mpiroot=None):
        # Set data and optionally randoms positions, scattering on all ranks if not already
        self.data_positions = data_positions
        self.randoms_positions = None
        if mpiroot is None or self.mpicomm.rank == mpiroot:
            self.data_positions = _format_positions(data_positions, position_type=position_type)
            if randoms_positions is not None:
                self.randoms_positions = _format_positions(randoms_positions, position_type=position_type)
        if mpiroot is not None: # Scatter position arrays on all ranks
            self.data_positions = mpi.scatter_array(self.data_positions, root=mpiroot, mpicomm=self.mpicomm)
            if randoms_positions is not None:
                self.randoms_positions = mpi.scatter_array(self.randoms_positions, root=mpiroot, mpicomm=self.mpicomm)

    def _set_weights(self, data_weights, randoms_weights=None, mpiroot=None):
        # Set data and optionally randoms weights and their sum, scattering on all ranks if not already
        self.data_weights = data_weights
        self.randoms_weights = None
        is_data_none = data_weights is None
        if mpiroot is not None: is_data_none = self.mpicomm.bcast(is_data_none, root=mpiroot)
        is_randoms_none = randoms_weights is None
        if mpiroot is not None: is_randoms_none = self.mpicomm.bcast(is_randoms_none, root=mpiroot)
        if is_data_none: self.data_weights = 1.
        if self.with_randoms and is_randoms_none: self.randoms_weights = 1.
        if mpiroot is None or self.mpicomm.rank == mpiroot:
            if not is_data_none:
                self.data_weights = np.asarray(data_weights)
            if self.with_randoms and not is_randoms_none:
                self.randoms_weights = np.asarray(randoms_weights)
        if mpiroot is not None: # Scatter weight arrays on all ranks
            if not is_data_none:
                self.data_weights = mpi.scatter_array(self.data_weights, root=mpiroot, mpicomm=self.mpicomm)
            if self.with_randoms and not is_randoms_none:
                self.randoms_weights = mpi.scatter_array(self.randoms_weights, root=mpiroot, mpicomm=self.mpicomm)

        def sum_weights(positions, weights):
            if np.ndim(weights) == 0:
                return self.mpicomm.allreduce(len(positions))*weights
            return self.mpicomm.allreduce(sum(weights))

        self.sum_randoms_weights = self.sum_data_weights = sum_weights(self.data_positions, self.data_weights)
        if self.with_randoms:
            self.sum_randoms_weights = sum_weights(self.randoms_positions, self.randoms_weights)

    @property
    def with_randoms(self):
        """Whether randoms positions have been provided."""
        return self.randoms_positions is not None

    def _set_resampler(self, resampler='cic'):
        # Set :attr:`resampler`
        self.resampler = _get_resampler(resampler=resampler)

    def to_mesh(self, field=None, dtype=None):
        """
        Paint positions/weights to mesh.

        Parameters
        ----------
        field : string, default=None
            Field to paint to mesh, one of:

                - "data": data positions and weights
                - "randoms": randoms positions and weights
                - "normalized_randoms": randoms positions and weights, renormalized (by alpha)
                   such that their sum is same as data weights
                - "fkp": FKP field, i.e. data - alpha * randoms
                - None: defaults to "data" if no randoms, else "fkp"

        dtype : string, dtype, default='f8'
            The data type of the mesh when painting, to override current :attr:`dtype`.

        Returns
        -------
        out : RealField
            Mesh, with values in "weights" units (not *normalized* as density).
        """
        if dtype is None: dtype = self.dtype

        if field is None:
            field = 'fkp' if self.with_randoms else 'data'
        field = field.lower()
        allowed_fields = ['data'] + (['randoms', 'normalized_randoms', 'fkp'] if self.with_randoms else [])
        if field not in allowed_fields:
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, [None] + allowed_fields))
        positions, weights = [], []
        if field in ['data', 'fkp']:
            positions += [self.data_positions]
            weights += [self.data_weights]
        if field in ['randoms', 'normalized_randoms', 'fkp']:
            positions += [self.randoms_positions]
            if field == 'randoms':
                randoms_weights = self.randoms_weights
            elif field == 'normalized_randoms':
                randoms_weights = self.sum_data_weights/self.sum_randoms_weights*self.randoms_weights
            elif field == 'fkp':
                randoms_weights = -self.sum_data_weights/self.sum_randoms_weights*self.randoms_weights
            weights += [randoms_weights]

        pm = ParticleMesh(BoxSize=self.boxsize, Nmesh=self.nmesh, dtype=dtype, comm=self.mpicomm)
        offset = self.boxcenter - self.boxsize/2.
        #offset = self.boxcenter
        #offset = 0.

        def paint(positions, weights, out, transform=None):
            positions = positions - offset
            factor = bool(self.interlacing) + 0.5
            # decompose positions such that they live in the same region as the mesh in the current process
            layout = pm.decompose(positions, smoothing=factor * self.resampler.support)
            positions = layout.exchange(positions)
            if np.ndim(weights) != 0:
                weights = layout.exchange(weights)
            # hold = True means no zeroing of out
            pm.paint(positions, mass=weights, resampler=self.resampler, transform=transform, hold=True, out=out)

        out = pm.create(type='real', value=0.)
        for p, w in zip(positions, weights): paint(p, w, out)

        if self.interlacing:
            if self.mpicomm.rank == 0:
                self.log_info('Running interlacing at order {:d}.'.format(self.interlacing))
            cellsize = self.boxsize/self.nmesh
            shifts = np.arange(self.interlacing)*1./self.interlacing
            # remove 0 shift, already computed
            shifts = shifts[1:]
            out = out.r2c()
            for shift in shifts:
                transform = pm.affine.shift(shift)
                # paint to two shifted meshes
                mesh_shifted = pm.create(type='real', value=0.)
                for p, w in zip(positions, weights): paint(p, w, mesh_shifted, transform=transform)
                mesh_shifted = mesh_shifted.r2c()
                for k, s1, s2 in zip(out.slabs.x, out.slabs, mesh_shifted.slabs):
                    kc = sum(k[i] * cellsize[i] for i in range(3))
                    s1[...] = s1[...] + s2[...] * np.exp(shift * 1j * kc)
            out = out.c2r()
            out[:] /= self.interlacing
        #out[:] /= (self.sum_data_weights/self.nmesh.prod())
        return out

    def unnormalized_shotnoise(self):
        """
        Return unnormalized shotnoise, as:

        .. math::

            \sum_{i=1}^{N_{g}} w_{i,g}^{2} + \alpha^{2} \sum_{i=1}^{N_{r}} w_{i,r}^{2}

        Where the sum runs over data (and optionally) randoms weights.
        """
        def sum_weights2(positions, weights=None):
            if np.ndim(weights) == 0:
                return self.mpicomm.allreduce(len(positions))*weights**2
            return self.mpicomm.allreduce(sum(weights**2))

        shotnoise = sum_weights2(self.data_positions, self.data_weights)
        if self.with_randoms:
            alpha = self.sum_data_weights/self.sum_randoms_weights
            shotnoise += alpha**2 * sum_weights2(self.randoms_positions, self.randoms_weights)
        return shotnoise


class MeshFFTPower(BaseClass):
    """
    Class that computes power spectrum from input mesh(es), using global or local line-of-sight, following https://arxiv.org/abs/1704.02357.
    In effect, this merge nbodykit's implementation of the global line-of-sight (periodic) algorithm of:
    https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/fftpower.py
    with the local line-of-sight algorithm of:
    https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py

    Attributes
    ----------
    poles : MultipolePowerSpectrum
        Estimated power spectrum multipoles.

    wedges : WedgePowerSpectrum
        Estimated power spectrum wedges.
    """

    def __init__(self, mesh1, mesh2=None, edges=None, ells=(0, 2, 4), los=None, boxcenter=None, compensations=None, wnorm=None, shotnoise=None):
        r"""
        Initialize :class:`MeshFFTPower`.

        Parameters
        ----------
        mesh1 : CatalogMesh, RealField
            First mesh.

        mesh2 : CatalogMesh, RealField, default=None
            In case of cross-correlation, second mesh, with same size and physical extent (``boxsize`` and ``boxcenter``) that ``mesh1``.

        edges : tuple, array, default=None
            If ``los`` is local (``None``), :math:`k`-edges for :attr:`poles`.
            Else, one can also provide :math:`\mu-edges` (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            If ``kedges`` is ``None``, defaults to edges containing unique :math:`k` (norm) values, see :func:`find_unique_edges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :amth:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'dk' (in which case :func:`find_unique_edges` is used to find unique :math:`k` (norm) values).

        ells : list, tuple, default=(0, 2, 4)
            Multipole orders.

        los : array, default=None
            If ``los`` is ``None``, use local (varying) line-of-sight.
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
            Power spectrum shot noise, to used instead of internal estimate, which is 0 in case of cross-correlation
            or both ``mesh1`` and ``mesh2`` are :class:`pmesh.pm.RealField`,
            and in case of auto-correlation is obtained by dividing :meth:`CatalogMesh.unnormalized_shotnoise`
            result on ``mesh1`` by power spectrum normalization.
        """
        self._set_compensations(compensations)

        self.mesh1 = mesh1
        self.mesh2 = mesh2
        self.autocorr = mesh2 is None or mesh2 is mesh1

        # Complex type is required for odd mutipoles
        requires_complex = los is None and any(ell % 2 == 1 for ell in ells)

        def get_dtype(mesh):
            if requires_complex and not mesh.dtype.name.startswith('complex'):
                dtype = np.dtype('c{:d}'.format(mesh.dtype.itemsize*2))
                if mesh.mpicomm.rank == 0:
                    self.log_warning('Odd multipoles are requested but input {} '\
                                     'has floating type; switching to {} as mandatory for odd multipoles.'.format(mesh.__class__.__name__, dtype))
            else:
                dtype = mesh.dtype
            return dtype

        for i in range(1 if self.autocorr else 2):
            name = 'mesh{:d}'.format(i+1)
            mesh = locals()[name]
            if isinstance(mesh, CatalogMesh):
                if mesh.mpicomm.rank == 0:
                    self.log_info('Painting catalog {:d} to mesh.'.format(i+1))
                setattr(self, name, mesh.to_mesh(dtype=get_dtype(mesh)))
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
            else:
                setattr(self, name, mesh)

        def enforce_dype(mesh):
            # Cast mesh to correct dtype
            dtype = get_dtype(mesh)
            toret = mesh
            if dtype != mesh.dtype:
                pm = ParticleMesh(BoxSize=mesh.pm.BoxSize, Nmesh=mesh.pm.Nmesh, dtype=dtype, comm=mesh.pm.comm)
                toret = pm.create(type='real')
                toret[...] = mesh[...]
            return toret

        self.mesh1 = enforce_dype(self.mesh1)
        if self.autocorr:
            self.mesh2 = self.mesh1
        else:
            self.mesh2 = enforce_dype(self.mesh2)
        self.boxcenter = _make_array(boxcenter if boxcenter is not None else 0., 3, dtype='f8')

        if isinstance(mesh1, CatalogMesh) and isinstance(mesh2, CatalogMesh):
            if not np.allclose(mesh1.boxcenter, mesh2.boxcenter):
                raise FFTPowerError('Mismatch in input box centers')
        if not np.allclose(self.mesh1.pm.BoxSize, self.mesh2.pm.BoxSize):
            raise FFTPowerError('Mismatch in input box sizes')
        if not np.allclose(self.mesh1.pm.Nmesh, self.mesh2.pm.Nmesh):
            raise FFTPowerError('Mismatch in input mesh sizes')
        if self.mesh2.pm.comm is not self.mesh1.pm.comm:
            raise FFTPowerError('Communicator mismatch between input meshes')

        self._set_edges(edges)
        self._set_los(los)
        self._set_ells(ells)
        self.wnorm = wnorm
        if wnorm is None:
            self.wnorm = normalization(mesh1, mesh2)
        self.shotnoise = shotnoise
        if shotnoise is None:
            self.shotnoise = 0.
            # Shot noise is non zero only if we can estimate it
            if self.autocorr and isinstance(mesh1, CatalogMesh):
                self.shotnoise = mesh1.unnormalized_shotnoise()/self.wnorm
        if self.mpicomm.rank == 0:
            self.log_info('Running power spectrum estimation.')
        self.run()

    def _set_compensations(self, compensations):
        # Set :attr:`compensations`
        if compensations is None: compensations = [None]*2
        if not isinstance(compensations, (tuple, list)):
            compensations = [compensations]*2

        def _format_compensation(compensation):
            if isinstance(compensation, dict):
                return compensation
            resampler = None
            for name in ['ngp', 'cic', 'tsc', 'pcs']:
                if name in compensation:
                    resampler = name
            if resampler is None:
                raise FFTPowerError('Specify resampler in compensation')
            shotnoise = 'shotnoise' in compensation or 'sn' in compensation
            return {'resampler':resampler, 'shotnoise':shotnoise}

        self.compensations = [_format_compensation(compensation) if compensation is not None else None for compensation in compensations]

    def _set_edges(self, edges):
        # Set :attr:`edges`
        if edges is None or (not isinstance(edges[0], dict) and np.ndim(edges[0]) == 0):
            edges = (edges,)
        if len(edges) == 1:
            kedges, muedges = edges[0], None
        else:
            kedges, muedges = edges
        if kedges is None:
            kedges = {}
        if isinstance(kedges, dict):
            kmin = kedges.get('min', 0.)
            kmax = kedges.get('max', np.pi/(self.nmesh/self.boxsize).max())
            dk = kedges.get('step', None)
            if dk is None:
                # find unique edges
                k = self.mesh1.pm.create_coords(type='complex')
                kedges = find_unique_edges(k, dk, xmin=kmin, xmax=kmax, mpicomm=self.mpicomm)
            else:
                kedges = np.arange(kmin, kmax*(1+1e-9), dk)
        if muedges is None:
            muedges = np.linspace(-1, 1, 2, endpoint=True) # single :math:`\mu`-wedge
        self.edges = (np.asarray(kedges, dtype='f8'), np.asarray(muedges, dtype='f8'))

    def _set_ells(self, ells):
        # Set :attr:`ells`
        if ells is None:
            if self.los is None:
                raise FFTPowerError('Specify non-empty list of ells')
            self.ells = None
        else:
            if np.ndim(ells) == 0:
                ells = (ells,)
            self.ells = tuple(ells)
            if self.los is None and not self.ells:
                raise FFTPowerError('Specify non-empty list of ells')
            if any(ell < 0 for ell in self.ells):
                raise FFTPowerError('Multipole numbers must be non-negative integers')

    def _set_los(self, los):
        # Set :attr:`los`
        if los is None:
            self.los = None
        else:
            if isinstance(los, str):
                los = 'xyz'.index(los)
            if np.ndim(los) == 0:
                ilos = los
                los = np.zeros(3, dtype='f8')
                los[ilos] = 1.
            los = np.array(los, dtype='f8')
            self.los = los/utils.distance(los)

    @property
    def mpicomm(self):
        """Current MPI communicator."""
        return self.mesh1.pm.comm

    @property
    def boxsize(self):
        """Physical box size."""
        return self.mesh1.pm.BoxSize

    @property
    def nmesh(self):
        """Mesh size."""
        return self.mesh1.pm.Nmesh

    def _compensate(self, cfield, *compensations):
        if self.mpicomm.rank == 0:
            self.log_info('Applying compensations {}.'.format(compensations))
        # Apply compensation window for particle-assignment scheme
        windows = [_get_compensation_window(**compensation) for compensation in compensations]
        #from nbodykit.source.mesh.catalog import CompensateCIC
        #cfield.apply(func=CompensateCIC, kind='circular', out=Ellipsis)
        cellsize = self.boxsize/self.nmesh
        for k, slab in zip(cfield.slabs.x, cfield.slabs):
            kc = tuple(ki * ci for ki, ci in zip(k, cellsize))
            for window in windows:
                slab[...] /= window(*kc)

    def run(self):
        if self.los is None: # local (varying) line-of-sight
            self._run_local_los()
        else: # global (fixed) line-of-sight
            self._run_global_los()

    def _get_attrs(self):
        # Return some attributes, to be saved in :attr:`poles` and :attr:`wedges`
        state = {}
        for name in ['nmesh', 'boxsize', 'boxcenter', 'los', 'compensations']:
            state[name] = getattr(self, name)
        return state

    def __getstate__(self):
        """Return this class state dictionary."""
        state = self._get_attrs()
        for name in ['wedges', 'poles']:
            if hasattr(self, name):
                state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state."""
        self.__dict__.update(state)
        for name in ['wedges', 'poles']:
            if name in state:
                setattr(self, name, get_power_statistic(statistic=state[name].pop('name')).from_state(state[name]))

    def _run_global_los(self):

        rank = self.mpicomm.rank
        start = time.time()
        # Calculate the 3d power spectrum, slab-by-slab to save memory
        # FFT 1st density field and apply the resampler transfer kernel
        cfield1 = self.mesh1.r2c()
        #print(cfield1.value.sum(), cfield1.value.dtype, cfield1.value.shape)
        if self.compensations[0] is not None:
            self._compensate(cfield1, self.compensations[0])

        if self.autocorr:
            cfield2 = cfield1
        else:
            cfield2 = self.mesh2.r2c()
            if self.compensations[1] is not None:
                self._compensate(cfield2, self.compensations[1])

        for i, c1, c2 in zip(cfield1.slabs.i, cfield1.slabs, cfield2.slabs):
            c1[...] = c1 * c2.conj()
            mask_zero = True
            for ii in i: mask_zero = mask_zero & (ii == 0)
            c1[mask_zero] = 0.

        #cfield1[:] *= self.boxsize.prod()

        #from nbodykit.algorithms.fftpower import project_to_basis
        #result, result_poles = project_to_basis(cfield1, self.edges, poles=[], los=self.los)
        result, result_poles = project_to_basis(cfield1, self.edges, ells=self.ells, los=self.los)

        stop = time.time()
        if rank == 0:
            self.log_info('Power spectrum computed in elapsed time {:.2f} s.'.format(stop - start))

        # Format the power results into :class:`WedgePowerSpectrum` instance
        attrs = self._get_attrs()
        kwargs = {'wnorm':self.wnorm, 'shotnoise_nonorm':self.shotnoise*self.wnorm, 'attrs':attrs}
        k, mu, power, nmodes = (np.squeeze(result[ii]) for ii in [0,1,2,3])
        norm = self.nmesh.prod()**2
        power *= norm
        self.wedges = WedgePowerSpectrum(modes=(k,mu), edges=self.edges, power_nonorm=power, nmodes=nmodes, **kwargs)

        if result_poles:
            # Format the power results into :class:`PolePowerSpectrum` instance
            k, power, nmodes = (np.squeeze(result_poles[ii]) for ii in [0,1,2])
            power *= norm
            self.poles = MultipolePowerSpectrum(modes=k, edges=self.edges[0], power_nonorm=power, nmodes=nmodes, ells=self.ells, **kwargs)

    def _run_local_los(self):

        rank = self.mpicomm.rank

        nonzeroells = ells = sorted(set(self.ells))
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]

        # FFT 1st density field and apply the resampler transfer kernel
        A0_1 = self.mesh1.r2c()
        # Set mean value or real field to 0
        for i, c1 in zip(A0_1.slabs.i, A0_1.slabs):
            mask_zero = True
            for ii in i: mask_zero = mask_zero & (ii == 0)
            c1[mask_zero] = 0.

        # We will apply all compensation transfer functions to A0_1 (faster than applying to each Aell)
        compensations = [self.compensations[0]] * 2 if self.autocorr else self.compensations
        compensations = [compensation for compensation in compensations if compensation is not None]

        result = []
        # Loop over the higher order multipoles (ell > 0)
        start = time.time()

        if self.autocorr:
            if nonzeroells:
                # Higher-order multipole requested
                # If monopole requested, copy A0_1 without window in Aell
                if 0 in self.ells: Aell = A0_1.copy()
                self._compensate(A0_1, *compensations)
            else:
                # In case of autocorrelation, and only monopole requested, no A0_1 copy need be made
                # Apply a single window, which will be squared by the autocorrelation
                if 0 in self.ells: Aell = A0_1
                self._compensate(A0_1, compensations[0])
        else:
            # Cross-correlation, all windows on A0_1
            if 0 in self.ells: Aell = self.mesh2.r2c()
            self._compensate(A0_1, *compensations)

        if 0 in self.ells:

            for islab in range(A0_1.shape[0]):
                Aell[islab,...] = A0_1[islab] * Aell[islab].conj()

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
            # NOTE: this will hold FFTs of density field #2
            rfield = RealField(self.mesh2.pm)
            cfield = ComplexField(self.mesh2.pm)
            Aell = ComplexField(self.mesh2.pm)

            # Spherical harmonic kernels (for ell > 0)
            Ylms = [[get_real_Ylm(ell,m) for m in range(-ell, ell+1)] for ell in nonzeroells]

            # Offset the box coordinate mesh ([-BoxSize/2, BoxSize]) back to the original (x,y,z) coords
            offset = self.boxcenter - self.boxsize/2.
            # NOTE: we do not apply half cell shift as in nbodykit below
            #offset = self.boxcenter - self.boxsize/2. + 0.5*self.boxsize / self.nmesh # in nbodykit
            #offset = self.boxcenter + 0.5*self.boxsize / self.nmesh # in nbodykit

            # The real-space grid
            xgrid = [xx.real.astype('f8') + offset[ii] for ii, xx in enumerate(_transform_rslab(self.mesh1.slabs.optx, self.boxsize))]
            #xgrid = [xx.astype('f8') + offset[ii] for ii, xx in enumerate(self.mesh1.slabs.optx)]
            xnorm = np.sqrt(sum(xx**2 for xx in xgrid))
            xgrid = [x/xnorm for x in xgrid]

            # The Fourier-space grid
            kgrid = [kk.real.astype('f8') for kk in A0_1.slabs.optx]
            knorm = np.sqrt(sum(kk**2 for kk in kgrid)); knorm[knorm==0.] = np.inf
            kgrid = [k/knorm for k in kgrid]

        for ill, ell in enumerate(nonzeroells):

            Aell[:] = 0.
            # Iterate from m=-ell to m=ell and apply Ylm
            substart = time.time()
            for Ylm in Ylms[ill]:
                # Reset the real-space mesh to the original density #2
                rfield[:] = self.mesh2[:]

                # Apply the config-space Ylm
                for islab, slab in enumerate(rfield.slabs):
                    slab[:] *= Ylm(xgrid[0][islab], xgrid[1][islab], xgrid[2][islab])

                # Real to complex of field #2
                rfield.r2c(out=cfield)

                # Apply the Fourier-space Ylm
                for islab, slab in enumerate(cfield.slabs):
                    slab[:] *= Ylm(kgrid[0][islab], kgrid[1][islab], kgrid[2][islab])

                # Add to the total sum
                Aell[:] += cfield[:]

                # And this contribution to the total sum
                substop = time.time()
                if rank == 0:
                    self.log_debug('Done term for Y(l={:d}, m={:d}) in {:.2f} s.'.format(Ylm.l, Ylm.m, substop - substart))

            if rank == 0:
                self.log_info('ell = {:d} done; {:d} r2c completed'.format(ell, len(Ylms[ill])))

            # Calculate the power spectrum multipoles, slab-by-slab to save memory
            # This computes (A0 of field #1) * (Aell of field #2).conj()
            for islab in range(A0_1.shape[0]):
                Aell[islab,...] = A0_1[islab] * Aell[islab].conj()

            # Project on to 1d k-basis (averaging over mu=[-1,1])
            proj_result = project_to_basis(Aell, self.edges)[0]
            result.append(4 * np.pi * np.squeeze(proj_result[2]))
            k, nmodes = proj_result[0], proj_result[-1]

        stop = time.time()
        if rank == 0:
            self.log_info('Power spectrum computed in elapsed time {:.2f} s.'.format(stop - start))
        # Factor of 4*pi from spherical harmonic addition theorem + volume factor
        norm = self.nmesh.prod()**2
        poles = np.array([result[ells.index(ell)] for ell in self.ells]) * norm
        # Format the power results into :class:`PolePowerSpectrum` instance
        k, nmodes = np.squeeze(k), np.squeeze(nmodes)
        attrs = self._get_attrs()
        kwargs = {'wnorm':self.wnorm, 'shotnoise_nonorm':self.shotnoise*self.wnorm, 'attrs':attrs}
        self.poles = MultipolePowerSpectrum(modes=k, edges=self.edges[0], power_nonorm=poles, nmodes=nmodes, ells=self.ells, **kwargs)


class CatalogFFTPower(MeshFFTPower):

    """Wrapper on :class:`MeshFFTPower` to start directly from positions and weights."""

    def __init__(self, data_positions1, data_positions2=None, randoms_positions1=None, randoms_positions2=None,
                data_weights1=None, data_weights2=None, randoms_weights1=None, randoms_weights2=None,
                edges=None, ells=(0, 2, 4), los=None,
                nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=1.5, dtype='f8',
                resampler='cic', interlacing=2, position_type='xyz', wnorm=None, shotnoise=None, mpiroot=None, mpicomm=mpi.COMM_WORLD):
        r"""
        Initialize :class:`CatalogFFTPower`.

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
            Positions in the first randoms catalog. Typically of shape (3, N) or (N, 3).

        randoms_positions2 : list, array, default=None
            Optionally (for cross-correlation), positions in the second randoms catalog. See ``randoms_positions1``.

        data_weights1 : array of shape (N,), default=None
            Optionally, weights in the first data catalog.

        data_weights2 : array of shape (N,), default=None
            Optionally (for cross-correlation), weights in the second data catalog.

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

        ells : list, tuple, default=(0, 2, 4)
            Multipole orders.

        los : array, default=None
            If ``los`` is ``None``, use local (varying) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

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

        boxpad : float, default=1.5
            When ``boxsize`` is determined from ``positions``, take ``boxpad`` times the smallest box enclosing ``positions`` as ``boxsize``.

        dtype : string, dtype, default='f8'
            The data type of the mesh when painting.

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

        wnorm : float, default=None
            Power spectrum normalization, to use instead of internal estimate obtained with :func:`normalization`.

        shotnoise : float, default=None
            Power spectrum shot noise, to used instead of internal estimate, which is 0 in case of cross-correlation
            or both ``mesh1`` and ``mesh2`` are :class:`pmesh.pm.RealField`,
            and in case of auto-correlation is obtained by dividing :meth:`CatalogMesh.unnormalized_shotnoise`
            result on ``mesh1`` by power spectrum normalization.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scatted across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=MPI.COMM_WORLD
            The MPI communicator.
        """
        positions = []
        d = {}
        for name in ['data_positions1', 'data_positions2', 'randoms_positions1', 'randoms_positions2']:
            tmp = locals()[name]
            if tmp is not None:
                tmp = _format_positions(tmp, position_type=position_type)
            if mpiroot is not None and mpicomm.bcast(tmp is not None, root=mpiroot):
                tmp = mpi.scatter_array(tmp, root=mpiroot, mpicomm=mpicomm)
            if tmp is not None:
                positions.append(tmp)
            d[name] = tmp

        for name in ['data_weights1', 'data_weights2', 'randoms_weights1', 'randoms_weights2']:
            tmp = locals()[name]
            if tmp is not None:
                tmp = np.asarray(tmp)
            if mpiroot is not None and mpicomm.bcast(tmp is not None, root=mpiroot):
                tmp = mpi.scatter_array(tmp, root=mpiroot, mpicomm=mpicomm)
            d[name] = tmp

        # Get box encompassing all catalogs
        nmesh, boxsize, boxcenter = _get_box(boxsize=boxsize, cellsize=cellsize, nmesh=nmesh, boxcenter=boxcenter, positions=positions, boxpad=boxpad, mpicomm=mpicomm)

        # Get catalog meshes
        def get_mesh(data_positions, data_weights=None, randoms_positions=None, randoms_weights=None):
            return CatalogMesh(data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights,
                               nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, resampler=resampler, interlacing=interlacing,
                               position_type='pos', dtype=dtype, mpicomm=mpicomm)

        mesh1 = get_mesh(d['data_positions1'], data_weights=d['data_weights1'], randoms_positions=d['randoms_positions1'], randoms_weights=d['randoms_weights1'])
        mesh2 = None
        if d['data_positions2'] is not None:
            mesh2 = get_mesh(d['data_positions2'], data_weights=d['data_weights2'], randoms_positions=d['randoms_positions2'], randoms_weights=d['randoms_weights2'])
        # Now, run power spectrum estimation
        super(CatalogFFTPower,self).__init__(mesh1=mesh1, mesh2=mesh2, edges=edges, ells=ells, los=los, wnorm=wnorm, shotnoise=shotnoise)
