"""
Implementation of the FFTlog algorithm, very much inspired by mcfit (https://github.com/eelregit/mcfit) and implementation in
https://github.com/sfschen/velocileptors/blob/master/velocileptors/Utils/spherical_bessel_transform_fftw.py
"""

import os
import warnings

import numpy as np


class FFTlog(object):
    r"""
    Implementation of the FFTlog algorithm presented in https://jila.colorado.edu/~ajsh/FFTLog/, which computes the generic integral:

    .. math::

        G(y) = \int_{0}^{\infty} x dx F(x) K(xy)

    with :math:`F(x)` input function, :math:`K(xy)` a kernel.

    This transform is (mathematically) invariant under a power law transformation:

    .. math::

        G_{q}(y) = \int_{0}^{\infty} x dx F_{q}(x) K_{q}(xy)

    where :math:`F_{q}(x) = G(x)x^{-q}`, :math:`K_{q}(t) = K(t)t^{q}` and :math:`G_{q}(y) = G(y)y^{q}`.
    """
    def __init__(self, x, kernel, q=0, minfolds=2, lowring=True, xy=1, check_level=0, engine='numpy', **engine_kwargs):
        """
        Initialize :class:`FFTlog`, which can perform several transforms at once.

        Parameters
        ----------
        x : array_like
            Input log-spaced coordinates. Must be strictly increasing.
            If 1D, is broadcast to the number of provided kernels.

        kernel : callable, list of callables
            Mellin transform of the kernel:
            .. math:: U_{K}(z) = \int_{0}^{\infty} t^{z-1} K(t) dt
            If a list of kernels is provided, will perform all transforms at once.

        q : float, list of floats
            Power-law tilt(s) to regularise integration.

        minfolds : int
            The c is chosen with minimum :math:`n` chosen such that ``2**n > minfolds * x.size``.

        lowring : bool
            If ``True`` set output coordinates according to the low-ringing condition, otherwise set it with ``xy``.

        xy : float, list of floats
            Enforce the reciprocal product (i.e. ``x[0] * y[-1]``) of the input ``x`` and output ``y`` coordinates.

        check_level : int
            If non-zero run sanity checks on input.

        engine : string, default='numpy'
            FFT engine. See :meth:`set_engine`.

        engine_kwargs : dict
            Arguments for FFT engine.

        Note
        ----
        Kernel definition is different from that of https://jila.colorado.edu/~ajsh/FFTLog/, which uses (eq. 10):

        .. math:: U_{K}(z) = \int_{0}^{\infty} t^{z} K(t) dt

        Therefore, one should use :math:`q = 1` for Bessel functions to match :math:`q = 0` in  https://jila.colorado.edu/~ajsh/FFTLog/.
        """
        self.kernel = kernel
        self.inparallel = isinstance(kernel,list)
        if not self.inparallel:
            self.kernel = [kernel]
        self.q = q
        if not isinstance(q,list):
            self.q = [q]*self.nparallel
        self.x = np.asarray(x)
        if not self.inparallel:
            self.x = self.x[None,:]
        elif self.x.ndim == 1:
            self.x = np.tile(self.x[None,:],(self.nparallel,1))
        self.xy = xy
        if not isinstance(xy,list):
            self.xy = [xy]*self.nparallel
        self.check_level = check_level
        if self.check_level:
            if len(self.x) != self.nparallel:
                raise ValueError('x and kernel must of same length')
            if len(self.q) != self.nparallel:
                raise ValueError('q and kernel must be lists of same length')
            if len(self.xy) != self.nparallel:
                raise ValueError('xy and kernel must be lists of same length')
        self.minfolds = minfolds
        self.lowring = lowring
        self.setup()
        self.set_engine(engine,**engine_kwargs)

    def set_engine(self, engine='numpy', **engine_kwargs):
        """
        Set up FFT engine.
        See :func:`get_engine`

        Parameters
        ----------
        engine : BaseEngine, string, default='numpy'
            FFT engine, or one of ['numpy', 'fftw'].

        engine_kwargs : dict
            Arguments for FFT engine.
        """
        self._engine = get_engine(engine,size=self.padded_size,nparallel=self.nparallel,**engine_kwargs)

    @property
    def nparallel(self):
        """Number of transforms performed in parallel."""
        return len(self.kernel)

    def setup(self):
        """Set up u funtions."""
        self.size = self.x.shape[-1]
        self.delta = np.log(self.x[:,-1]/self.x[:,0])/(self.size-1)

        nfolds = (self.size * self.minfolds - 1).bit_length()
        self.padded_size = 2**nfolds
        npad = self.padded_size - self.size
        self.padded_size_in_left,self.padded_size_in_right = npad//2, npad - npad//2
        self.padded_size_out_left,self.padded_size_out_right = npad - npad//2, npad//2

        if self.check_level:
            if not np.allclose(np.log(self.x[:,1:]/self.x[:,:-1]),self.delta,rtol=1e-3):
                raise ValueError('Input x must be log-spaced')
            if self.padded_size < self.size:
                raise ValueError('Convolution size must be larger than input x size')

        if self.lowring:
            self.lnxy = np.array([delta / np.pi * np.angle(kernel(q + 1j * np.pi / delta)) for kernel,delta,q in zip(self.kernel,self.delta,self.q)])
        else:
            self.lnxy = np.log(self.xy) + self.delta

        self.y = np.exp(self.lnxy - self.delta)[:,None] / self.x[:,::-1]

        m = np.arange(0, self.padded_size//2 + 1)
        self.padded_u,self.padded_prefactor,self.padded_postfactor = [],[],[]
        self.padded_x = pad(self.x,(self.padded_size_in_left,self.padded_size_in_right),axis=-1,extrap='log')
        self.padded_y = pad(self.y,(self.padded_size_out_left,self.padded_size_out_right),axis=-1,extrap='log')
        prev_kernel,prev_q,prev_lnxy,prev_u = None,None,None,None
        for kernel,padded_x,padded_y,lnxy,delta,q in zip(self.kernel,self.padded_x,self.padded_y,self.lnxy,self.delta,self.q):
            if q == prev_q and xy == prev_xy:
                self.padded_prefactor.append(self.padded_prefactor[-1])
                self.padded_postfactor.append(self.padded_postfactor[-1])
            else:
                self.padded_prefactor.append(padded_x**(-q))
                self.padded_postfactor.append(padded_y**(-q))
            if kernel == prev_kernel and q == prev_q:
                u = prev_u
            else:
                u = prev_u = kernel(q + 2j * np.pi / self.padded_size / delta * m)
            if lnxy == prev_lnxy:
                self.padded_u.append(self.padded_u[-1])
            else:
                self.padded_u.append(u*np.exp(-2j * np.pi * lnxy / self.padded_size / delta * m))
        self.padded_u = np.array(self.padded_u)
        self.padded_prefactor = np.array(self.padded_prefactor)
        self.padded_postfactor = np.array(self.padded_postfactor)

    def __call__(self, fun, extrap=0, keep_padding=False):
        """
        Perform the transforms.

        Parameters
        ----------
        fun : array_like
            Function to be transformed.
            Last dimensions should match (:attr:`nparallel`,len(x)) where ``len(x)`` is the size of the input x-coordinates.
            (if :attr:`nparallel` is 1, the only requirement is the last dimension to be (len(x))).

        extrap : float, string
            How to extrapolate function outside of  ``x`` range to fit the integration range.
            If 'log', performs a log-log extrapolation.
            If 'edge', pad ``fun`` with its edge values.
            Else, pad ``fun`` with the provided value.
            Pass a tuple to differentiate between left and right sides.

        keep_padding : bool
            Whether to return function padded to the number of points in the integral.
            By default, crop it to its original size.

        Returns
        -------
        y : numpy.ndarray
            Array of new coordinates.

        fftloged : numpy.ndarray
            Transformed function.
        """
        scales = np.linspace(1.,3.,3)
        padded_fun = pad(fun,(self.padded_size_in_left,self.padded_size_in_right),axis=-1,extrap=extrap)
        fftloged = self._engine.backward(self._engine.forward(padded_fun*self.padded_prefactor) * self.padded_u) * self.padded_postfactor

        if not keep_padding:
            y,fftloged = self.y,fftloged[...,self.padded_size_out_left:self.padded_size_out_left+self.size]
        else:
            y,fftloged = self.padded_y,fftloged
        if not self.inparallel:
            y = y[0]
            fftloged.shape = fun.shape if not keep_padding else fun.shape[:-1] + (self.padded_size,)
        return y,fftloged

    def inv(self):
        """Inverse the transform."""
        self.x, self.y = self.y, self.x
        self.padded_x, self.padded_y = self.y, self.x
        self.padded_prefactor, self.padded_postfactor = 1 / self.padded_postfactor, 1 / self.padded_prefactor
        self.padded_u = 1 / self.padded_u.conj()


class HankelTransform(FFTlog):
    """
    Hankel transform implementation using :class:`FFTlog`.

    It relies on Bessel function kernels.
    """
    def __init__(self, x, nu=0, **kwargs):
        """
        Initialize Hankel transform.

        Parameters
        ----------
        x : array_like
            Input log-spaced coordinates.
            If 1D, is broadcast to the number of provided ``nu``.

        nu : int, list of int
            Order of Bessel functions.
            If a list is provided, will perform all transforms at once.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        if np.ndim(nu) == 0:
            kernel = BesselJKernel(nu)
        else:
            kernel = [BesselJKernel(nu_) for nu_ in nu]
        FFTlog.__init__(self, x, kernel, **kwargs)
        self.padded_prefactor *= self.padded_x**2


class PowerToCorrelation(FFTlog):
    r"""
    Power spectrum to correlation function transform, defined as:

    .. math::
        \xi_{\ell}(s) = \frac{(-i)^{\ell}}{2 \pi^{2}} \int dk k^{2} P_{\ell}(k) j_{\ell}(ks)

    """
    def __init__(self, k, ell=0, q=0, complex=True, **kwargs):
        """
        Initialize power to correlation transform.

        Parameters
        ----------
        k : array_like
            Input log-spaced wavenumbers.
            If 1D, is broadcast to the number of provided ``ell``.

        ell : int, list of int
            Poles. If a list is provided, will perform all transforms at once.

        q : float, list of floats
            Power-law tilt(s) to regularise integration.

        complex : bool, default=True
            ``False`` returns the real part of even poles, and the imaginary part of odd poles.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        if np.ndim(ell) == 0:
            kernel = SphericalBesselJKernel(ell)
        else:
            kernel = [SphericalBesselJKernel(ell_) for ell_ in ell]
        FFTlog.__init__(self, k, kernel, q=1.5+q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 / (2*np.pi)**1.5
        # Convention is (-i)^ell/(2 pi^2)
        ell = np.atleast_1d(ell)
        if complex:
            phase = (-1j) ** ell
        else:
            # We return imaginary part of odd poles
            phase = (-1)**((ell + 1)//2)
        # Not in-place as phase (and hence padded_postfactor) may be complex instead of float
        self.padded_postfactor = self.padded_postfactor * phase[:,None]


class CorrelationToPower(FFTlog):
    r"""
    Correlation function to power spectrum transform, defined as:

    .. math::
        P_{\ell}(k) = 4 \pi i^{\ell} \int ds s^{2} \xi_{\ell}(s) j_{\ell}(ks)

    """
    def __init__(self, s, ell=0, q=0, complex=True, **kwargs):
        """
        Initialize power to correlation transform.

        Parameters
        ----------
        s : array_like
            Input log-spaced separations.
            If 1D, is broadcast to the number of provided ``ell``.

        ell : int, list of int
            Poles. If a list is provided, will perform all transforms at once.

        q : float, list of floats
            Power-law tilt(s) to regularise integration.

        complex : bool, default=True
            ``False`` returns the real part of even poles, and the imaginary part of odd poles.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        if np.ndim(ell) == 0:
            kernel = SphericalBesselJKernel(ell)
        else:
            kernel = [SphericalBesselJKernel(ell_) for ell_ in ell]
        FFTlog.__init__(self, s, kernel, q=1.5+q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 * (2*np.pi)**1.5
        # Convention is 4 \pi i^ell, and we return imaginary part of odd poles
        ell = np.atleast_1d(ell)
        if complex:
            phase = (-1j) ** ell
        else:
            # We return imaginary part of odd poles
            phase = (-1)**(ell//2)
        self.padded_postfactor = self.padded_postfactor * phase[:,None]


class TophatVariance(FFTlog):
    """
    Variance in tophat window.

    It relies on tophat kernel.
    """
    def __init__(self, k, q=0, **kwargs):
        """
        Initialize tophat variance transform.

        Parameters
        ----------
        k : array_like
            Input log-spaced wavenumbers.
            If 1D, is broadcast to the number of provided ``ell``.

        q : float, list of floats
            Power-law tilt(s) to regularise integration.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        kernel = TophatSqKernel(ndim=3)
        FFTlog.__init__(self, k, kernel, q=1.5+q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 / (2 * np.pi**2)


class GaussianVariance(FFTlog):
    """
    Variance in Gaussian window.

    It relies on Gaussian kernel.
    """
    def __init__(self, k, q=0, **kwargs):
        """
        Initialize Gaussian variance transform.

        Parameters
        ----------
        k : array_like
            Input log-spaced wavenumbers.
            If 1D, is broadcast to the number of provided ``ell``.

        q : float, list of floats
            Power-law tilt(s) to regularise integration.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        kernel = GaussianSqKernel()
        FFTlog.__init__(self, k, kernel, q=1.5+q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 / (2 * np.pi**2)


def pad(array, pad_width, axis=-1, extrap=0):
    """
    Pad array along ``axis``.

    Parameters
    ----------
    array : array_like
        Input array to be padded.

    pad_width : int, tuple of ints
        Number of points to be added on both sides of the array.
        Pass a tuple to differentiate between left and right sides.

    axis : int
        Axis along which padding is to be applied.

    extrap : string, float
        If 'log', performs a log-log extrapolation.
        If 'edge', pad ``array`` with its edge values.
        Else, pad ``array`` with the provided value.
        Pass a tuple to differentiate between left and right sides.

    Returns
    -------
    array : numpy.ndarray
        Padded array.
    """
    array = np.asarray(array)

    try:
        pad_width_left, pad_width_right = pad_width
    except (TypeError, ValueError):
        pad_width_left = pad_width_right = pad_width

    try:
        extrap_left, extrap_right = extrap
    except (TypeError, ValueError):
        extrap_left = extrap_right = extrap

    axis = axis % array.ndim
    to_axis = [1] * array.ndim
    to_axis[axis] = -1

    if extrap_left == 'edge':
        end = np.take(array, [0], axis=axis)
        pad_left = np.repeat(end, pad_width_left, axis=axis)
    elif extrap_left == 'log':
        end = np.take(array, [0], axis=axis)
        ratio = np.take(array, [1], axis=axis) / end
        exp = np.arange(-pad_width_left, 0).reshape(to_axis)
        pad_left = end * ratio ** exp
    else:
        pad_left = np.full(array.shape[:axis]+(pad_width_left,)+array.shape[axis+1:],extrap_left)

    if extrap_right == 'edge':
        end = np.take(array, [-1], axis=axis)
        pad_right = np.repeat(end, pad_width_right, axis=axis)
    elif extrap_right == 'log':
        end = np.take(array, [-1], axis=axis)
        ratio = np.take(array, [-2], axis=axis) / end
        exp = np.arange(1, pad_width_right + 1).reshape(to_axis)
        pad_right = end / ratio ** exp
    else:
        pad_right = np.full(array.shape[:axis]+(pad_width_right,)+array.shape[axis+1:],extrap_right)

    return np.concatenate([pad_left, array, pad_right], axis=axis)


class BaseFFTEngine(object):

    """Base FFT engine."""

    def __init__(self, size, nparallel=1, nthreads=None):
        """
        Initialize FFT engine.

        Parameters
        ----------
        size : int
            Array size.

        nparallel : int
            Number of FFTs to be performed in parallel.

        nthreads : int, default=None
            Number of threads.
        """
        self.size = size
        self.nparallel = nparallel
        if nthreads is not None:
            os.environ['OMP_NUM_THREADS'] = str(nthreads)
        self.nthreads = int(os.environ.get('OMP_NUM_THREADS',1))


class NumpyFFTEngine(BaseFFTEngine):

    """FFT engine based on :mod:`numpy.fft`."""

    def forward(self, fun):
        """Forward transform of ``fun``."""
        return np.fft.rfft(fun,axis=-1)

    def backward(self, fun):
        """Backward transform of ``fun``."""
        return np.fft.hfft(fun,n=self.size,axis=-1) / self.size


def apply_along_last_axes(func, array, naxes=1, toret=None):
    """Apply callable ``func`` over the last ``naxes`` of ``array``."""
    if toret is None:
        toret = np.empty_like(array)
    shape_bak = array.shape
    array.shape = (-1,) + shape_bak[-naxes:]
    newshape_bak = toret.shape
    toret.shape = (-1,) + newshape_bak[-naxes:]
    for iarr,arr in enumerate(array):
        toret[iarr] = func(arr)
    array.shape = shape_bak
    toret.shape = newshape_bak
    return toret


try: import pyfftw
except ImportError: pyfftw = None


class FFTWEngine(BaseFFTEngine):

    """FFT engine based on :mod:`pyfftw`."""

    def __init__(self, size, nparallel=1, nthreads=None, wisdom=None, plan='measure'):
        """
        Initialize :mod:`pyfftw` engine.

        Parameters
        ----------
        size : int
            Array size.

        nparallel : int
            Number of FFTs to be performed in parallel.

        nthreads : int, default=None
            Number of threads.

        wisdom : string, tuple, default=None
            :mod:`pyfftw` wisdom, to speed up initialization of FFTs.
            If a string, should be a path to the save FFT wisdom (with :func:`numpy.save`).
            If a tuple, directly corresponds to the wisdom.

        plan : string, default='measure'
            Choices are ['estimate', 'measure', 'patient', 'exhaustive'].
            The increasing amount of effort spent during the planning stage to create the fastest possible transform.
            Usually 'measure' is a good compromise.
        """
        if pyfftw is None:
            raise NotImplementedError('Install pyfftw to use {}'.format(self.__class__.__name__))
        super(FFTWEngine, self).__init__(size,nparallel=nparallel,nthreads=nthreads)
        plan = plan.lower()
        allowed_plans = ['estimate', 'measure', 'patient', 'exhaustive']
        if plan not in allowed_plans:
            raise MeshError('Plan {} unknown'.format(plan))
        plan = 'FFTW_{}'.format(plan.upper())

        if isinstance(wisdom, str):
            wisdom = tuple(np.load(wisdom))
        if wisdom is not None:
            pyfftw.import_wisdom(wisdom)
        else:
            pyfftw.forget_wisdom()
        #flags = ('FFTW_DESTROY_INPUT','FFTW_MEASURE')
        self.fftw_f = pyfftw.empty_aligned((self.nparallel,self.size),dtype='float64')
        self.fftw_fk = pyfftw.empty_aligned((self.nparallel,self.size//2+1),dtype='complex128')
        self.fftw_gk = pyfftw.empty_aligned((self.nparallel,self.size//2+1),dtype='complex128')
        self.fftw_g = pyfftw.empty_aligned((self.nparallel,self.size),dtype='float64')

        #pyfftw.config.NUM_THREADS = threads
        self.fftw_forward_object = pyfftw.FFTW(self.fftw_f,self.fftw_fk,direction='FFTW_FORWARD',flags=(plan,),threads=self.nthreads)
        self.fftw_backward_object = pyfftw.FFTW(self.fftw_gk,self.fftw_g,direction='FFTW_BACKWARD',flags=(plan,),threads=self.nthreads)

    def forward(self, fun):
        """Forward transform of ``fun``."""
        if fun.ndim > 1 and fun.shape[:-1] != (self.nparallel,):
            # if nparallel match, apply along two last axes, else only last axis (nparallel should be 1)
            toret = np.empty_like(self.fftw_fk,shape=fun.shape[:-1] + self.fftw_fk.shape[-1:])
            return apply_along_last_axes(self.forward,fun,naxes=1+(fun.shape[-2] == self.nparallel),toret=toret)
        #fun.shape = self.fftw_f.shape
        self.fftw_f[...] = fun
        return self.fftw_forward_object(normalise_idft=True)

    def backward(self, fun):
        """Backward transform of ``fun``."""
        if fun.ndim > 1 and fun.shape[:-1] != (self.nparallel,):
            toret = np.empty_like(self.fftw_g,shape=fun.shape[:-1] + self.fftw_g.shape[-1:])
            return apply_along_last_axes(self.backward,fun,naxes=1+(fun.shape[-2] == self.nparallel),toret=toret)
        #fun.shape = self.fftw_gk.shape
        self.fftw_gk[...] = np.conj(fun)
        return self.fftw_backward_object(normalise_idft=True)


def get_engine(engine, *args, **kwargs):
    """
    Return FFT engine.

    Parameters
    ----------
    engine : BaseFFTEngine, string
        FFT engine, or one of ['numpy', 'fftw'].

    args, kwargs : tuple, dict
        Arguments for FFT engine.

    Returns
    -------
    engine : BaseFFTEngine
    """
    if isinstance(engine, str):
        if engine.lower() == 'numpy':
            return NumpyFFTEngine(*args, **kwargs)
        if engine.lower() == 'fftw':
            return FFTWEngine(*args, **kwargs)
        raise ValueError('FFT engine {} is unknown'.format(engine))
    return engine

from scipy.special import gamma, loggamma


class BaseKernel(object):

    """Base kernel."""

    def __call__(self, z):
        return self.eval(z)

    def __eq__(self, other):
        return other.__class__ == self.__class__


class BaseBesselKernel(BaseKernel):

    """Base Bessel kernel."""

    def __init__(self, nu):
        self.nu = nu

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.nu == self.nu


class BesselJKernel(BaseBesselKernel):

    """(Mellin transform of) Bessel kernel."""

    def eval(self, z):
        return np.exp(np.log(2)*(z-1) + loggamma(0.5*(self.nu+z)) - loggamma(0.5*(2+self.nu-z)))


class SphericalBesselJKernel(BaseBesselKernel):

    """(Mellin transform of) spherical Bessel kernel."""

    def eval(self, z):
        return np.exp(np.log(2)*(z-1.5) + loggamma(0.5*(self.nu+z)) - loggamma(0.5*(3+self.nu-z)))


class BaseTophatKernel(BaseKernel):

    """Base tophat kernel."""

    def __init__(self, ndim=1):
        self.ndim = ndim

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.ndim == self.ndim


class TophatKernel(BaseTophatKernel):

    """(Mellin transform of) tophat kernel."""

    def eval(self, z):
        return np.exp(np.log(2)*(z-1) + loggamma(1+0.5*self.ndim) + loggamma(0.5*z) - loggamma(0.5*(2+self.ndim-z)))


class TophatSqKernel(BaseTophatKernel):

    """(Mellin transform of) square of tophat kernel."""

    def __init__(self, ndim=1):
        self.ndim = ndim
        if self.ndim == 1:

            def eval(z):
                return -0.25*np.sqrt(np.pi) * np.exp(loggamma(0.5*(z-2)) - loggamma(0.5*(3-z)))

        elif self.ndim == 3:

            def eval(z):
                return 2.25*np.sqrt(np.pi)*(z-2)/(z-6) * np.exp(loggamma(0.5*(z-4)) - loggamma(0.5*(5-z)))

        else:

            def eval(z):
                return np.exp(np.log(2)*(self.ndim-1) + 2*loggamma(1+0.5*self.ndim) \
                        + loggamma(0.5*(1+self.ndim-z)) + loggamma(0.5*z) \
                        - loggamma(1+self.ndim-0.5*z) - loggamma(0.5*(2+self.ndim-z))) / np.sqrt(np.pi)

        self.eval = eval


class GaussianKernel(BaseKernel):

    """(Mellin transform of) Gaussian kernel."""

    def eval(self, z):
        return 2**(0.5*z-1) * gamma(0.5*z)


class GaussianSqKernel(BaseKernel):

    """(Mellin transform of) square of Gaussian kernel."""

    def eval(self, z):
        return 0.5 * gamma(0.5*z)
