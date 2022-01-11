"""
Implementation of window function estimation, following https://github.com/cosmodesi/GC_derivations,
and https://fr.overleaf.com/project/60e99d5d5a0f5a3a220de2cc.
"""

import numpy as np
from scipy import special

from .utils import BaseClass
from .fftlog import PowerToCorrelation
from .fft_power import MeshFFTPower, get_real_Ylm
from .wide_angle import BaseMatrix


Si = lambda x: special.sici(x)[0]
# derivative of correlation function w.r.t. k-bins, precomputed with sympy
_correlation_function_tophat_derivatives = {}
_correlation_function_tophat_derivatives[0] = lambda s, a, b: (-(-a*np.cos(a*s)/s + np.sin(a*s)/s**2)/s + (-b*np.cos(b*s)/s + np.sin(b*s)/s**2)/s)/(2*np.pi**2)
_correlation_function_tophat_derivatives[2] = lambda s, a, b: -(-(a*s*np.cos(a*s) - 4*np.sin(a*s) + 3*Si(a*s))/s**3 + (b*s*np.cos(b*s) - 4*np.sin(b*s) + 3*Si(b*s))/s**3)/(2*np.pi**2)
_correlation_function_tophat_derivatives[4] = lambda s, a, b: (-(-a*s**3*np.cos(a*s) + 11*s**2*np.sin(a*s) + 15*s**2*Si(a*s)/2 + 105*s*np.cos(a*s)/(2*a) - 105*np.sin(a*s)/(2*a**2))/s**5 +\
                                                              (-b*s**3*np.cos(b*s) + 11*s**2*np.sin(b*s) + 15*s**2*Si(b*s)/2 + 105*s*np.cos(b*s)/(2*b) - 105*np.sin(b*s)/(2*b**2))/s**5)/(2*np.pi**2)


def get_correlation_function_tophat_derivative(kedges, ell=0, k=None, **kwargs):
    r"""
    Return a list of callable corresponding to the derivative of the correlation function
    w.r.t. :math:`k`-bins.

    Parameters
    ----------
    kedges : array
        :math:`k`-edges of the :math:`k`-bins.

    ell : int, default=0
        Multipole order.

    k : array, default=None
        If ``None``, calculation will be analytic, which will work if ``ell`` in [0, 2, 4], or sympy package is installed
        (such analytic integration with sympy may take several seconds).
        If not ``None``, this is the :math:`k` log-spaced array for numerical FFTlog integration.

    kwargs : dict
        If ``k`` is not ``None``, other arguments for :class:`fftlog.PowerToCorrelation`.

    Returns
    -------
    toret : list
        List of callables, taking configuration-space separation ``s`` as input.
    """
    try: import numexpr
    except ImportError: numexpr = None

    if k is None:
        if ell in _correlation_function_tophat_derivatives:
            fun = _correlation_function_tophat_derivatives[ell]
        else:
            try:
                import sympy as sp
            except ImportError as exc:
                raise ImportError('Install sympy to for analytic computation') from exc
            k, s, a, b = sp.symbols('k s a b', real=True, positive=True)
            integrand = sp.simplify(k**2 * sp.expand_func(sp.jn(ell, k*s)))
            # i^ell; we provide the imaginary part of the odd correlation function multipoles
            expr = (-1)**(ell//2)/(2*sp.pi**2) * sp.integrate(integrand, (k, a, b))
            fun = sp.lambdify((s, a, b), expr, modules=['numpy', {'Si': Si}])
        toret = []
        for kmin, kmax in zip(kedges[:-1], kedges[1:]):
            toret.append(lambda s: fun(s, kmin, kmax))

        return toret

    fftlog = PowerToCorrelation(k, ell=ell, **kwargs)
    toret = []
    for kmin, kmax in zip(kedges[:-1], kedges[1:]):
        tophat = np.zeros_like(k)
        tophat[(k >= kmin) & (k <= kmax)] = 1.
        sep, fun = fftlog(tophat)
        # current prefactor is i^ell
        fun = fun * (-1j)**ell * (-1)**(ell//2) # we provide the imaginary part of the odd correlation function multipoles
        toret.append(lambda s: np.interp(s, sep, fun.real))
    return toret


class MeshFFTWindowMatrix(BaseMatrix):

    """Window matrix, relating "theory" input to "observed" output."""

    def __init__(self, matrix, xin, xout, projsin, projsout, nmodes, wnorm=1., attrs=None):
        """
        Initialize :class:`MeshFFTWindowMatrix`.

        Parameters
        ----------
        matrix : array
            2D array representing window matrix.

        xin : array, list
            List of input "theory" coordinates.
            If single array, assumed to be the same for all input projections ``projsin``.

        xout : list
            List of output "theory" coordinates.
            If single array, assumed to be the same for all output projections ``projsout``.

        projsin : list
            List of input "theory" projections.

        projsout : list
            List of output "observed" projections.

        nmodes : array
            Number of modes in each bin.

        wnorm : float, default=1.
            Window function normalization.

        attrs : dict, default=None
            Dictionary of other attributes.
        """
        super(MeshFFTWindowMatrix, self).__init__(matrix, xin, xout, projsin, projsout, weightsout=nmodes, attrs=attrs)
        self.wnorm = wnorm

    @property
    def nmodes(self):
        return self.weightsout

    @nmodes.setter
    def nmodes(self, nmodes):
        self.weightsout = nmodes

    @classmethod
    def from_power(cls, power, xin, projin=(0, 0)):
        """
        Create window function from input :class:`MultipolePowerSpectrum`.

        Parameters
        ----------
        power : MultipolePowerSpectrum
            Power spectrum measurement to convert into :class:`PowerSpectrumWindowMatrix`.

        xin : float
            Input "theory" bin.

        projin : tuple, Projection, default=(0, 0)
            Input "theory" projection, i.e. (multipole, wide-angle order) tuple.

        Returns
        -------
        matrix : PowerSpectrumWindowMatrix
        """
        xin = [np.asarray([xin])]
        projsin = [projin]
        xout = np.squeeze(np.array([modes.ravel() for modes in power.modes]).T)
        projsout = [Projection(ell=ell, wa_order=None) for ell in power.ells]
        matrix = np.atleast_2d(power.power.ravel()).T
        return cls(matrix, xin, xout, projsin, projsout, power.nmodes, wnorm=power.wnorm, edges=power.edges)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = super(MeshFFTWindowMatrix, self).__getstate__()
        for name in ['wnorm']:
            state[name] = getattr(self, name)
        return state


class MeshFFTWindow(MeshFFTPower):
    """
    Class that computes window function from input mesh(es), using global or local line-of-sight, https://github.com/cosmodesi/GC_derivations.

    Attributes
    ----------
    poles : MeshFFTWindowMatrix
        Window matrix.
    """
    def __init__(self, mesh1, mesh2=None, edgesin=None, projsin=None, power_ref=None, edges=None, ells=None, los=None, boxcenter=None, compensations=None, wnorm=None, shotnoise=None):
        """
        Initialize :class:`MeshFFTWindow`.

        Parameters
        ----------
        mesh1 : CatalogMesh, RealField
            First mesh.

        mesh2 : CatalogMesh, RealField, default=None
            In case of cross-correlation, second mesh, with same size and physical extent (``boxsize`` and ``boxcenter``) that ``mesh1``.

        edgesin : dict, array, list
            Dictionary of projection: list of derivative (callable) of theory correlation function w.r.t. each theory basis vector,
            e.g. each in :math:`k` bin.
            If not callable, an array of :math:`k`-edges which defines the theory binning; corresponding derivatives will be computed
            using :func:`get_correlation_function_tophat_derivative`.
            If not a dictionary, input is assumed the same for all projections.

        projsin : list, default=None
            List of :class:`Projection` instances or (multipole, wide-angle order) tuples.
            If ``None``, and ``power_ref`` is provided, the list of projections is set
            to be able to compute window convolution of theory power spectrum multipoles of orders ``power_ref.ells``.

        power_ref : MultipolePowerSpectrum, default=None
            "Reference" power spectrum estimation, e.g. of the actual data.
            It is used to set default values for ``edges``, ``ells``, ``los``, ``boxcenter``, ``compensations`` and ``wnorm`` if those are ``None``.

        edges : tuple, array, default=None
            If ``los`` is local (``None``), :math:`k`-edges for :attr:`poles`.
            Else, one can also provide :math:`\mu`-edges (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            If ``kedges`` is ``None``, defaults to edges containing unique :math:`k` (norm) values, see :func:`find_unique_edges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :amth:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'dk' (in which case :func:`find_unique_edges` is used to find unique :math:`k` (norm) values).
            For both :math:`k` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end, i.e. ``bins[i] <= x < bins[i+1]``.
            However, last :math:`\mu`-bin is inclusive on both ends: ``bins[-2] <= mu <= bins[-1]``.
            Therefore, with e.g. :math:`\mu`-edges ``[0.2, 0.4, 1.0]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 1.0`.
            Similarly, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 0.8`.
            If ``None``, defaults to the edges used in estimation of ``power_ref``.

        ells : list, tuple, default=(0, 2, 4)
            Output multipole orders.
            If ``None``, defaults to the multipoles used in estimation of ``power_ref``.

        los : string, array, default='firstpoint'
            If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.
            If ``None``, defaults to the line-of-sight used in estimation of ``power_ref``.

        boxcenter : float, array, default=None
            Box center; defaults to 0.
            Used only if provided ``mesh1`` and ``mesh2`` are not ``CatalogMesh``.
            If ``None``, defaults to the value used in estimation of ``power_ref``.

        compensations : list, tuple, string, default=None
            Compensations to apply to mesh to (optionally) correct for particle-mesh assignment scheme;
            e.g. 'cic' (resp. 'cic-sn') for cic assignment scheme, with (resp. without) interlacing.
            In case ``mesh2`` is not ``None`` (cross-correlation), provide a list (or tuple) of two such strings
            (for ``mesh1`` and ``mesh2``, respectively).
            Used only if provided ``mesh1`` or ``mesh2`` are not ``CatalogMesh``.

        wnorm : float, default=None
            Window function normalization.
            If ``None``, defaults to the value used in estimation of ``power_ref``,
            rescaled to the input random weights --- which yields a correct normalization of the window function
            for the power spectrum estimation ``power_ref``.
            If ``power_ref`` provided, use internal estimate obtained with :func:`normalization` --- which is wrong
            (the normalieation :attr:`poles.wnorm` can be reset a posteriori using the above recipe).

        shotnoise : float, default=None
            Power spectrum shot noise, to use instead of internal estimate, which is 0 in case of cross-correlation
            or both ``mesh1`` and ``mesh2`` are :class:`pmesh.pm.RealField`,
            and in case of auto-correlation is obtained by dividing :meth:`CatalogMesh.unnormalized_shotnoise`
            of ``mesh1`` by power spectrum normalization.
        """
        if power_ref is not None:
            if edges is None: edges = power_ref.edges
            if ells is None: ells = power_ref.ells
            if los is None: los = power_ref.attrs['los']
            if boxcenter is None: boxcenter = power_ref.attrs['boxcenter']
            if compensations is None: compensations = power_ref.attrs['compensations']
            if projs is None:
                ellmax = max(power_ref.ells)
                with_odd = int(any(ell % 2 for ell in power_ref.ells))
                projs = [(ell, 0) for ell in range(0, ellmax + 1, 2 - with_odd)]
                if los is None or isinstance(los, str) and los in ['firstpoint', 'endpoint']:
                    projs += [(ell, 1) for ell in range(1 - with_odd, ellmax + 1, 2 - with_odd)]

        self.projsin = [Projection(proj) for proj in projsin]

        self._set_compensations(compensations)
        self._set_mesh(mesh1, mesh2=mesh2)
        self._set_xin(edgesin)
        self._set_edges(edges)
        self._set_los(los)
        if self.los_type == 'global' and any(proj.wa_order != 0 for proj in self.projsin):
            raise ValueError('With global line-of-sight, input wide_angle order = 0 only is suppored')
        self._set_ells(ells)
        self.wnorm = wnorm
        if wnorm is None:
            if power_ref is not None:
                ialpha2 = np.prod([self.attrs[name]/power_ref.attrs[name] for name in ['sum_data_weights1', 'sum_data_weights2']])
                self.wnorm = ialpha2 * power_ref.wnorm
            else:
                self.wnorm = normalization(mesh1, mesh2)
        self.shotnoise = shotnoise
        if shotnoise is None:
            self.shotnoise = 0.
            # Shot noise is non zero only if we can estimate it
            if self.autocorr and isinstance(mesh1, CatalogMesh):
                self.shotnoise = mesh1.unnormalized_shotnoise()/self.wnorm
        self.attrs.update(self._get_attrs())
        if self.mpicomm.rank == 0:
            self.log_info('Running window function estimation.')
        self.run()

    def _set_xin(self, edgesin):
        if not isinstance(edgesin, dict):
            edgesin = {proj: edgesin for proj in self.projsin}
        else:
            edgesin = {Projection(proj): edge for proj, edge in projsin.items()}
        self.xin, self.deriv = {}, {}
        for proj in self.projsin:
            if proj not in edgesin:
                raise ValueError('Projection {} not in edgesin'.format(proj))
            iscallable = [callable(f) for f in edgesin[proj]]
            if any(iscallable):
                if not all(iscallable): raise ValueError('Provide callables or floats only for edgesin')
                self.deriv[proj] = edgesin
                self.xin[proj] = np.arange(len(self.deriv[proj]))
            else:
                edges = np.asarray(edgesin[projin])
                self.xin[proj] = 3./4. * (edges[1:]**4 - edges[:-1]**4) / (edges[1:]**3 - edges[:-1]**3)
                self.deriv[proj] = get_correlation_function_tophat_derivative(edges, ell=proj.ell)

    def _get_q(self, ellout, mout, projin):

        Ylmout = get_real_Ylm(ellout, mout)
        Ylmins = [get_real_Ylm(ell, m) for m in range(-projin.ell, projin.ell+1)]

        rfield = RealField(self.rfield2.pm)
        rfield[:] = self.rfield2[:]
        for islab, slab in enumerate(rfield.slabs):
            tmp = Ylmout(self.xgrid[0][islab], self.xgrid[1][islab], self.xgrid[2][islab])
            if projin.wa_order != 0: tmp /= self.rgrid[islab]**projin.wa_order
            slab[:] *= tmp
        cfield2 = rfield.r2c(out=cfield).conj()

        cfield = ComplexField(self.rfield1.pm)
        toret = RealField(self.rfield1.pm)
        toret[:] = 0.

        for Ylm in Ylmins:

            rfield[:] = self.rfield1[:]
            for islab, slab in enumerate(rfield.slabs):
                slab[:] *= Ylm(self.xgrid[0][islab], self.xgrid[1][islab], self.xgrid[2][islab])
            rfield.r2c(out=cfield)
            cfield[:] *= cfield2[:]

            cfield.r2c(out=rfield)
            for islab, slab in enumerate(rfield.slabs):
                slab[:] *= 4.*np.pi/(2*projin.ell + 1) * Ylm(self.xgridw[0][islab], self.xgridw[1][islab], self.xgridw[2][islab])
            toret[:] += rfield[:]

        return toret

    def _run_local_los(self, projin, deriv):

        result = []

        for ellout in self.ellsout:
            dfield = RealField(self.rfield1.pm)
            for islab, slab in enumerate(dfield.slabs):
                tmp = deriv(self.rgridw[islab])
                if projin.wa_order != 0: tmp *= self.rgridw[islab]**projin.wa_order
                slab[:] = tmp

            wfield = ComplexField(self.rfiel1.pm)
            wfield[:] = 0.
            for mout in range(-ellout, ellout+1):
                qfield = self._get_q(ellout=ellout, mout=mout, projin=projin)
                qfield[:] *= dfield[:]
                cfield = qfield.r2c()
                for islab, slab in enumerate(cfield.slabs):
                    slab[:] *= 4.*np.pi/(2*ellout + 1) * Ylm(kgrid[0][islab], kgrid[1][islab], kgrid[2][islab])
                wfield[:] += cfield[:]

            del dfield
            proj_result = project_to_basis(wfield, self.edges)[0]
            result.append(np.squeeze(proj_result[2]))
            k, nmodes,r = proj_result[0], proj_result[-1]

        k, nmodes = np.squeeze(k), np.squeeze(nmodes)
        kwargs = {'wnorm':self.wnorm, 'shotnoise_nonorm':self.shotnoise*self.wnorm, 'attrs':self.attrs}
        self.poles = MultipolePowerSpectrum(modes=k, edges=self.edges[0], power_nonorm=poles, nmodes=nmodes, ells=self.ells, **kwargs)

    def _run_global_los(self, projin, deriv):

        qfield = self.qfield.copy()

        legendre = special.legendre(projin.ell)
        for islab, slab in enumerate(qfield.slabs):
            tmp = deriv(self.rgridw[islab])
            if projin.ell:
                mu = sum(xx*ll for xx, ll in zip(self.xgridw[islab], self.los))
                tmp *= legendre(mu)
            slab[:] *= tmp

        wfield = qfield.r2c()

        result, result_poles = project_to_basis(wfield, self.edges, ells=self.ells, los=self.los)
        # Format the power results into :class:`WedgePowerSpectrum` instance
        kwargs = {'wnorm':self.wnorm, 'shotnoise_nonorm':self.shotnoise*self.wnorm, 'attrs':self.attrs}
        k, mu, power, nmodes = (np.squeeze(result[ii]) for ii in [0,1,2,3])
        norm = self.nmesh.prod()**2
        power *= norm
        self.wedges = WedgePowerSpectrum(modes=(k, mu), edges=self.edges, power_nonorm=power, nmodes=nmodes, **kwargs)

        if result_poles:
            # Format the power results into :class:`PolePowerSpectrum` instance
            k, power, nmodes = (np.squeeze(result_poles[ii]) for ii in [0,1,2])
            power *= norm
            self.poles = MultipolePowerSpectrum(modes=k, edges=self.edges[0], power_nonorm=power, nmodes=nmodes, ells=self.ells, **kwargs)

    def run(self):

        swap = self.los_type == 'firstpoint'
        if swap: self.mesh1, self.mesh2 = self.mesh2, self.mesh1 # swap meshes + complex conjugaison at the end of run()

        def _wrap_rslab(rslab):
            # We do not use the same conventions as pmesh:
            # rslab < 0 is sent back to [boxsize/2, boxsize]
            toret = []
            for ii, rr in enumerate(rslab):
                mask = rr > self.boxsize[ii]/2.
                rr[mask] -= self.boxsize[ii]
                toret.append(rr)
            return toret

        xgridw = _wrap_rslab(_transform_rslab(self.mesh1.slabs.optx, self.boxsize)) # this should just give self.mesh1.slabs.optx
        self.rgridw = np.sqrt(sum(xx**2 for xx in xgridw))

        cfield2 = cfield1 = self.mesh1.r2c()
        # Set mean value or real field to 0
        for i, c1 in zip(cfield1.slabs.i, cfield1.slabs):
            mask_zero = True
            for ii in i: mask_zero = mask_zero & (ii == 0)
            c1[mask_zero] = 0.

        if self.autocorr:
            self._compensate(cfield1, compensations[0])
        else:
            # We will apply all compensation transfer functions to cfield1
            compensations = [self.compensations[0]] * 2 if self.autocorr else self.compensations
            compensations = [compensation for compensation in compensations if compensation is not None]
            self._compensate(cfield1, *compensations)
            cfield2 = self.mesh2.r2c()

        if self.los_type == 'global': # global (fixed) line-of-sight

            cfield1[:] *= cfield2[:]
            self.qfield = cfield2.c2r()
            del self.mesh2, self.mesh1, cfield2, cfield1
            run_projin = self._run_global_los

        else: # local (varying) line-of-sight

            self.xgridw = [xx/self.rgridw for xx in xgridw]

            # Offset the box coordinate mesh ([-BoxSize/2, BoxSize]) back to the original (x,y,z) coords
            offset = self.boxcenter - self.boxsize/2.
            rgrid = [xx.real.astype('f8') + offset[ii] for ii, xx in enumerate(_transform_rslab(self.mesh1.slabs.optx, self.boxsize))]
            self.rgrid = np.sqrt(sum(xx**2 for xx in xgrid))
            self.xgrid = [xx/self.rgrid for xx in xgrid]

            # The Fourier-space grid
            kgrid = [kk.real.astype('f8') for kk in cfield1.slabs.optx]
            knorm = np.sqrt(sum(kk**2 for kk in kgrid)); knorm[knorm==0.] = np.inf
            self.kgrid = [kk/knorm for kk in kgrid]

            self.rfield2 = self.rfield1 = cfield1.c2r()
            if not self.autocorr:
                self.rfield2 = cfield2.r2c()
                del self.mesh2, self.mesh1, cfield2, cfield1
            run_projin = self._run_local_los

        poles, wedges = [], []
        for proj in self.projsin:
            poles_x, wedges_x = [], []
            for iin, xin in enumerate(self.xin[proj]):
                run_projin(projin, self.deriv[projin][iin])
                poles_x.append(MeshFFTWindowMatrix.from_power(self.poles, xin, projin))
                if self.los_type == 'global':
                    wedges_x.append(MeshFFTWindowMatrix.from_power(self.wedges, xin, projin))
            poles.append(MeshFFTWindowMatrix.concatenate_x(*poles_x, axis='in'))
            if wedges_x:
                wedges.append(MeshFFTWindowMatrix.concatenate_x(*wedges_x, axis='in'))
        self.poles = MeshFFTWindowMatrix.concatenate_proj(poles, axis='in')
        if wedges:
            self.wedges = MeshFFTWindowMatrix.concatenate_proj(wedges, axis='in')

    @classmethod
    def concatenate_proj(cls, *others):
        new = others[0].copy()
        for name in ['poles', 'wedges']:
            if hasattr(others[0], name):
                setattr(new, name, others[0].concatenate_proj(*[getattr(other, name) for other in others], axis='in'))
        return new

    @classmethod
    def concatenate_x(cls, *others):
        new = others[0].copy()
        for name in ['poles', 'wedges']:
            if hasattr(others[0], name):
                setattr(new, name, others[0].concatenate_x(*[getattr(other, name) for other in others], axis='in'))
        return new


class CatalogFFTWindow(MeshFFTPower):

    """Wrapper on :class:`MeshFFTPower` to estimate window function from input random positions and weigths."""

    def __init__(self, randoms_positions1=None, randoms_positions2=None,
                randoms_weights1=None, randoms_weights2=None,
                edgesin=None, projsin=None, edges=None, ells=None, power_ref=None,
                los=None, nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=2., dtype=None,
                resampler=None, interlacing=None, position_type='xyz', weight_type='auto', weight_attrs=None,
                wnorm=None, shotnoise=None, mpiroot=None, mpicomm=mpi.COMM_WORLD):
        r"""
        Initialize :class:`CatalogFFTWindow`, i.e. estimate power spectrum window matrix.

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

        edgesin : dict, array, list
            Dictionary of projection: list of derivative (callable) of theory correlation function w.r.t. each theory basis vector,
            e.g. each in :math:`k` bin.
            If not callable, an array of :math:`k`-edges which defines the theory binning; corresponding derivatives will be computed
            using :func:`get_correlation_function_tophat_derivative`.
            If not a dictionary, input is assumed the same for all projections.

        projsin : list, default=None
            List of :class:`Projection` instances or (multipole, wide-angle order) tuples.
            If ``None``, and ``power_ref`` is provided, the list of projections is set
            to be able to compute window convolution of theory power spectrum multipoles of orders ``power_ref.ells``.

        power_ref : MultipolePowerSpectrum, default=None
            "Reference" power spectrum estimation, e.g. of the actual data.
            It is used to set default values for ``edges``, ``ells``, ``los``, ``boxsize``, ``boxcenter``, ``nmesh``,
            ``interlacing``, ``resampler`` and ``wnorm`` if those are ``None``.

        edges : tuple, array, default=None
            If ``los`` is local (``None``), :math:`k`-edges for :attr:`poles`.
            Else, one can also provide :math:`\mu`-edges (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            If ``kedges`` is ``None``, defaults to edges containing unique :math:`k` (norm) values, see :func:`find_unique_edges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :amth:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'dk' (in which case :func:`find_unique_edges` is used to find unique :math:`k` (norm) values).
            For both :math:`k` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end, i.e. ``bins[i] <= x < bins[i+1]``.
            However, last :math:`\mu`-bin is inclusive on both ends: ``bins[-2] <= mu <= bins[-1]``.
            Therefore, with e.g. :math:`\mu`-edges ``[0.2, 0.4, 1.0]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 1.0`.
            Similarly, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 0.8`.
            If ``None``, defaults to the edges used in estimation of ``power_ref``.

        ells : list, tuple, default=(0, 2, 4)
            Output multipole orders.

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
        mesh_names = ['nmesh', 'boxsize', 'boxcenter']
        loc = locals()
        mesh_attrs = {name: loc[name] for name in mesh_names if loc[name] is not None}
        if power_ref is not None:
            for name in mesh_names:
                mesh_attrs.setdefault(name, power_ref.attrs[name])
            if interlacing is None:
                interlacing = tuple(power_ref.attrs['interlacing{:d}'.format(i+1)] for i in range(2))
            if resampler is None:
                resampler = tuple(power_ref.attrs['resampler{:d}'.format(i+1)] for i in range(2))

        if cellsize is not None: # if cellsize is provided, remove default nmesh or boxsize value from old_matrix instance.
            mesh_attrs['cellsize'] = cellsize
            if nmesh is None: mesh_attrs.pop('nmesh')
            elif boxsize is None: mesh_attrs.pop('boxsize')

        bpositions, positions = [], {}
        for name in ['randoms_positions1', 'randoms_positions2']:
            tmp = _format_positions(locals()[name], position_type=position_type, dtype=dtype, mpicomm=mpicomm, mpiroot=mpiroot)
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
            weight, n_bitwise_weights = _format_weights(locals()[name], weight_type=weight_type, dtype=dtype, mpicomm=mpicomm, mpiroot=mpiroot)

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

        # Get catalog meshes
        def get_mesh(data_positions, data_weights=None, **kwargs):
            return CatalogMesh(data_positions, data_weights=data_weights,
                               nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter,
                               position_type='pos', dtype=dtype, mpicomm=mpicomm, **kwargs)

        mesh1 = get_mesh(positions['R1'], data_weights=weights['R1'], resampler=resampler[0], interlacing=interlacing[0])
        mesh2 = None
        if not autocorr:
            mesh2 = get_mesh(positions['R2'], data_weights=weights['R2'], resampler=resampler[1], interlacing=interlacing[1])

        # Now, run power spectrum estimation
        super(CatalogFFTWindow, self).__init__(mesh1=mesh1, mesh2=mesh2, edgesin=edgesin, projsin=projsin, power_ref=power_ref, edges=edges, ells=ells, los=los, wnorm=wnorm, shotnoise=shotnoise)
