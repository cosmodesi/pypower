"""
Implementation of window function estimation, following https://github.com/cosmodesi/GC_derivations,
and https://fr.overleaf.com/project/60e99d5d5a0f5a3a220de2cc.
"""

import numpy as np
from scipy import special
from pmesh.pm import ParticleMesh, RealField, ComplexField

from . import mpi
from .utils import BaseClass
from .fftlog import PowerToCorrelation
from .fft_power import MeshFFTPower, get_real_Ylm, _transform_rslab, _get_real_dtype, _make_array, project_to_basis, PowerSpectrumMultipoles, PowerSpectrumWedges, normalization
from .wide_angle import BaseMatrix, Projection
from .mesh import CatalogMesh, _get_box, _wrap_in_place
from .direct_power import _format_positions, _format_weights


Si = lambda x: special.sici(x)[0]
# derivative of correlation function w.r.t. k-bins, precomputed with sympy
_correlation_function_tophat_derivatives = {}
_correlation_function_tophat_derivatives[0] = lambda s, a, b: (-(-a*np.cos(a*s)/s + np.sin(a*s)/s**2)/s + (-b*np.cos(b*s)/s + np.sin(b*s)/s**2)/s)/(2*np.pi**2)
_correlation_function_tophat_derivatives[1] = lambda s, a, b: (-(-a*np.sin(a*s) - 2*np.cos(a*s)/s)/s**2 + (-b*np.sin(b*s) - 2*np.cos(b*s)/s)/s**2)/(2*np.pi**2)
_correlation_function_tophat_derivatives[2] = lambda s, a, b: -(-(a*s*np.cos(a*s) - 4*np.sin(a*s) + 3*Si(a*s))/s**3 + (b*s*np.cos(b*s) - 4*np.sin(b*s) + 3*Si(b*s))/s**3)/(2*np.pi**2)
_correlation_function_tophat_derivatives[3] = lambda s, a, b: -(-(a*s**2*np.sin(a*s) + 7*s*np.cos(a*s) - 15*np.sin(a*s)/a)/s**4 + (b*s**2*np.sin(b*s) + 7*s*np.cos(b*s) - 15*np.sin(b*s)/b)/s**4)/(2*np.pi**2)
_correlation_function_tophat_derivatives[4] = lambda s, a, b: (-(-a*s**3*np.cos(a*s) + 11*s**2*np.sin(a*s) + 15*s**2*Si(a*s)/2 + 105*s*np.cos(a*s)/(2*a) - 105*np.sin(a*s)/(2*a**2))/s**5 +\
                                                              (-b*s**3*np.cos(b*s) + 11*s**2*np.sin(b*s) + 15*s**2*Si(b*s)/2 + 105*s*np.cos(b*s)/(2*b) - 105*np.sin(b*s)/(2*b**2))/s**5)/(2*np.pi**2)

def _get_attr_in_inst(obj, name, insts=(None,)):
    # Search for ``name`` in instances of name ``insts`` of obj
    for inst in insts:
        if inst is None:
            if hasattr(obj, name):
                return getattr(obj, name)
        else:
            if hasattr(obj, inst) and hasattr(getattr(obj, inst), name):
                return getattr(getattr(obj, inst), name)


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
            # i^ell; we take in the imaginary part of the odd power spectrum multipoles
            expr = (-1)**(ell//2) / (2*sp.pi**2) * sp.integrate(integrand, (k, a, b))
            fun = sp.lambdify((s, a, b), expr, modules=['numpy', {'Si': Si}])

        def _make_fun(kmin, kmax):
            return lambda s: fun(s, kmin, kmax)

        toret = []
        for kmin, kmax in zip(kedges[:-1], kedges[1:]):
            toret.append(_make_fun(kmin, kmax))

        return toret

    fftlog = PowerToCorrelation(k, ell=ell, complex=False, **kwargs)

    def _make_fun(sep, fun):
        return lambda s: np.interp(s, sep, fun)

    toret = []
    for kmin, kmax in zip(kedges[:-1], kedges[1:]):
        tophat = np.zeros_like(k)
        tophat[(k >= kmin) & (k <= kmax)] = 1.
        sep, fun = fftlog(tophat)
        toret.append(_make_fun(sep, fun))
    return toret


class PowerSpectrumFFTWindowMatrix(BaseMatrix):

    """Window matrix, relating "theory" input to "observed" output."""

    def __init__(self, matrix, xin, xout, projsin, projsout, nmodes, wnorm=1., attrs=None):
        """
        Initialize :class:`PowerSpectrumFFTWindowMatrix`.

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
        super(PowerSpectrumFFTWindowMatrix, self).__init__(matrix, xin, xout, projsin, projsout, weightsout=nmodes, attrs=attrs)
        self.cvalue = self.value # let us just keep the original value somewhere
        value = []
        nout = 0
        for iout, xout in enumerate(self.xout):
            slout = slice(nout, nout+len(xout))
            tmp = self.cvalue[:,slout]
            tmp = tmp.real if self.projsout[iout].ell % 2 == 0 else tmp.imag
            value.append(tmp)
            nout = slout.stop
        self.value = np.concatenate(value, axis=-1)
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
        Create window function from input :class:`PowerSpectrumMultipoles`.

        Parameters
        ----------
        power : PowerSpectrumMultipoles
            Power spectrum measurement to convert into :class:`PowerSpectrumFFTWindowMatrix`.

        xin : float
            Input "theory" bin.

        projin : tuple, Projection, default=(0, 0)
            Input "theory" projection, i.e. (multipole, wide-angle order) tuple.

        Returns
        -------
        matrix : PowerSpectrumFFTWindowMatrix
        """
        xin = [np.asarray([xin])]
        projsin = [projin]
        ells = getattr(power, 'ells', [0]) # in case of PowerSpectrumWedges, only 0
        projsout = [Projection(ell=ell, wa_order=None) for ell in ells]
        xout = [np.squeeze(np.array([modes.ravel() for modes in power.modes]).T)]*len(projsout) # modes are k for PowerSpectrumMultipoles, (k, mu) for PowerSpectrumWedges
        weights = [power.nmodes.ravel()]*len(projsout)
        matrix = np.atleast_2d(power.power.ravel())
        attrs = power.attrs.copy()
        attrs['edges'] = power.edges
        return cls(matrix, xin, xout, projsin, projsout, weights, wnorm=power.wnorm, attrs=attrs)

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

    def __getstate__(self):
        """Return this class state dictionary."""
        state = super(PowerSpectrumFFTWindowMatrix, self).__getstate__()
        for name in ['cvalue', 'wnorm']:
            state[name] = getattr(self, name)
        return state


class MeshFFTWindow(MeshFFTPower):
    """
    Class that computes window function from input mesh(es), using global or local line-of-sight, see:

        - https://github.com/cosmodesi/GC_derivations
        - https://fr.overleaf.com/read/hpgbwqzmtcxn

    Attributes
    ----------
    poles : PowerSpectrumFFTWindowMatrix
        Window matrix.
    """
    def __init__(self, mesh1=None, mesh2=None, edgesin=None, projsin=None, power_ref=None, edges=None, ells=None, los=None, periodic=False, boxcenter=None, compensations=None, wnorm=None, **kwargs):
        """
        Initialize :class:`MeshFFTWindow`.

        Parameters
        ----------
        mesh1 : CatalogMesh, RealField
            First mesh.

        mesh2 : CatalogMesh, RealField, default=None
            In case of cross-correlation, second mesh, with same size and physical extent (``boxsize`` and ``boxcenter``) that ``mesh1``.

        edgesin : dict, array, list
            An array of :math:`k`-edges which defines the theory :math:`k`-binning; corresponding derivatives will be computed
            using :func:`get_correlation_function_tophat_derivative`; or a dictionary of such array for each theory projection.
            Else a list of derivatives (callable) of theory correlation function w.r.t. each theory basis vector, e.g. each in :math:`k`-bin;
            or a dictionary of such list for each theory projection.
            If ``periodic`` is ``True``, this should correspond to the derivatives of theory *power spectrum* (instead of correlation function)
            w.r.t. each theory basis vector, e.g. each in :math:`k` bin.

        projsin : list, default=None
            List of :class:`Projection` instances or (multipole, wide-angle order) tuples.
            If ``None``, and ``power_ref`` is provided, the list of projections is set
            to be able to compute window convolution of theory power spectrum multipoles of orders ``power_ref.ells``.

        power_ref : CatalogFFTPower, MeshFFTPower, PowerSpectrumWedges, PowerSpectrumMultipoles, default=None
            "Reference" power spectrum estimation, e.g. of the actual data.
            It is used to set default values for ``edges``, ``ells``, ``los``, ``boxcenter``, ``compensations`` and ``wnorm`` if those are ``None``.

        edges : tuple, array, default=None
            If ``los`` is local (``None``), :math:`k`-edges for :attr:`poles`.
            Else, one can also provide :math:`\mu`-edges (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            If ``kedges`` is ``None``, defaults to edges containing unique :math:`k` (norm) values, see :func:`find_unique_edges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'dk' (in which case :func:`find_unique_edges` is used to find unique :math:`k` (norm) values).
            For both :math:`k` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end, i.e. ``edges[i] <= x < edges[i+1]``.
            However, last :math:`\mu`-bin is inclusive on both ends: ``edges[-2] <= mu <= edges[-1]``.
            Therefore, with e.g. :math:`\mu`-edges ``[0.2, 0.4, 1.0]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 1.0`.
            Similarly, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 0.8`.
            If ``None``, defaults to the edges used in estimation of ``power_ref``.

        ells : list, tuple, default=(0, 2, 4)
            Output multipole orders.
            If ``None``, defaults to the multipoles used in estimation of ``power_ref``.

        los : string, array, default=None
            If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.
            If ``None``, defaults to the line-of-sight used in estimation of ``power_ref``.

        periodic : bool, default=False
            If ``True``, selection function is assumed uniform, periodic.

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
            (the normalization :attr:`poles.wnorm` can be reset a posteriori using the above recipe).
        """
        if power_ref is not None:

            if edges is None: edges = _get_attr_in_inst(power_ref, 'edges', insts=(None, 'wedges', 'poles'))
            attrs_ref = _get_attr_in_inst(power_ref, 'attrs', insts=(None, 'wedges', 'poles'))
            if los is None: los = attrs_ref['los']
            if boxcenter is None: boxcenter = attrs_ref['boxcenter']
            if compensations is None: compensations = attrs_ref['compensations']
            if ells is None: ells = _get_attr_in_inst(power_ref, 'ells', insts=(None, 'poles'))

        self._set_los(los)
        self._set_ells(ells)
        self._set_periodic(periodic)
        if mesh1 is None:
            if not self.periodic:
                raise ValueError('mesh1 can be "None" only if periodic = True')
            attrs_pm = {'dtype':'f8', 'mpicomm':mpi.COMM_WORLD}
            if  power_ref is not None:
                attrs_pm.update(boxsize=attrs_ref['boxsize'], nmesh=attrs_ref['nmesh'], dtype=attrs_ref.get('dtype', attrs_pm['dtype']))
            attrs_pm.update(kwargs)
            translate = {'boxsize': 'BoxSize', 'nmesh': 'Nmesh', 'mpicomm': 'comm'}
            attrs_pm = {translate.get(key, key): value for key, value in attrs_pm.items()}
            mesh1 = ParticleMesh(**attrs_pm)
        self._set_compensations(compensations)
        self._set_mesh(mesh1, mesh2=mesh2, boxcenter=boxcenter)
        self._set_projsin(projsin)
        self._set_edges(edges)
        self._set_xin(edgesin)

        self.wnorm = wnorm
        if wnorm is None:
            if self.periodic:
                self.wnorm = 1.
            else:
                if power_ref is not None:
                    ialpha2 = np.prod([self.attrs[name]/power_ref.attrs[name] for name in ['sum_data_weights1', 'sum_data_weights2']])
                    self.wnorm = ialpha2 * power_ref.wnorm
                else:
                    self.wnorm = normalization(mesh1, mesh2)
        self.attrs.update(self._get_attrs())
        if self.mpicomm.rank == 0:
            self.log_info('Running window function estimation.')
        self.run()

    def _set_periodic(self, periodic=False):
        self.periodic = periodic
        if self.periodic and self.los_type != 'global':
            raise ValueError('Cannot set "periodic" if line-of-sight is local.')

    def _set_mesh(self, mesh1, mesh2=None, boxcenter=None):
        if self.periodic:
            self.attrs = {}
            self.autocorr = True
            if isinstance(mesh1, ParticleMesh):
                self.pm = mesh1
            else:
                self.pm = mesh1.pm
            self.boxcenter = _make_array(boxcenter if boxcenter is not None else 0., 3, dtype='f8')
        else:
            super(MeshFFTWindow, self)._set_mesh(mesh1, mesh2=mesh2, boxcenter=boxcenter)

    def _set_projsin(self, projsin):
        if projsin is None:
            if self.ells is None:
                raise ValueError('If no output multipoles requested, provide "projsin"')
            ellmax = max(self.ells)
            with_odd = int(any(ell % 2 for ell in self.ells))
            projsin = [(ell, 0) for ell in range(0, ellmax + 1, 2 - with_odd)]
            if self.los_type in ['firstpoint', 'endpoint']:
                projsin += [(ell, 1) for ell in range(1 - with_odd, ellmax + 1, 2 - with_odd)]
        self.projsin = [Projection(proj) for proj in projsin]
        if self.los_type == 'global' and any(proj.wa_order != 0 for proj in self.projsin):
            raise ValueError('With global line-of-sight, input wide_angle order = 0 only is supported')

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
                edges = np.asarray(edgesin[proj])
                self.xin[proj] = 3./4. * (edges[1:]**4 - edges[:-1]**4) / (edges[1:]**3 - edges[:-1]**3)
                if self.periodic:

                    def _make_fun(low, high):
                        return lambda k: 1. * ((k >= low) & (k < high))

                    self.deriv[proj] = [_make_fun(*lh) for lh in zip(edges[:-1], edges[1:])]
                else:
                    self.deriv[proj] = get_correlation_function_tophat_derivative(edges, ell=proj.ell)

    def _get_q(self, ellout, mout, projin):
        # Called for local (varying) line-of-sight only
        # This corresponds to Q defined in https://fr.overleaf.com/read/hpgbwqzmtcxn
        # ellout is \ell, mout is m, projin = (\ell^\prime, m^\prime)
        Ylmout = get_real_Ylm(ellout, mout)
        Ylmins = [get_real_Ylm(projin.ell, m) for m in range(-projin.ell, projin.ell+1)]

        rfield = RealField(self.pm)
        cfield = ComplexField(self.pm)
        toret = RealField(self.pm)
        toret[:] = 0.

        for Ylmin in Ylmins:

            for islab, slab in enumerate(rfield.slabs):
                slab[:] = self.mesh1[islab] * Ylmin(self.xhat[0][islab], self.xhat[1][islab], self.xhat[2][islab]) * Ylmout(self.xhat[0][islab], self.xhat[1][islab], self.xhat[2][islab])
                if projin.wa_order != 0: slab[:] /= self.xnorm[islab]**projin.wa_order
            rfield.r2c(out=cfield)

            for islab in range(cfield.shape[0]):
                cfield[islab,...] = cfield[islab].conj() * self.cfield2[islab]

            cfield.c2r(out=rfield)
            for islab, slab in enumerate(rfield.slabs):
                # No 1/N^6 factor due to pmesh convention
                slab[:] = slab[:] * 4.*np.pi/(2*projin.ell + 1) * Ylmin(self.xwhat[0][islab], self.xwhat[1][islab], self.xwhat[2][islab])
                mask_zero = True
                for ii in slab.i: mask_zero = mask_zero & (ii == 0)
                slab[mask_zero] = 0.
            toret[:] += rfield[:]

        return toret

    def _run_local_los(self, projin, deriv):
        # We we perform the sum of Q defined in https://fr.overleaf.com/read/hpgbwqzmtcxn
        # projin is \ell^\prime, n
        # deriv is \xi^{(n)}_{\ell^{\prime},\beta \ell^\prime}(s^w)
        swap = self.los_type == 'endpoint'
        if swap: self.mesh1, self.mesh2 = self.mesh2, self.mesh1 # swap meshes + complex conjugaison at the end of run()

        result = []
        ells = sorted(set(self.ells))

        for ellout in ells:
            dfield = RealField(self.pm)
            for islab, slab in enumerate(dfield.slabs):
                tmp = np.zeros_like(self.xwnorm[islab])
                mask_nonzero = self.xwnorm[islab] != 0.
                tmp[mask_nonzero] = deriv(self.xwnorm[islab][mask_nonzero])
                if projin.wa_order != 0: tmp *= self.xwnorm[islab]**projin.wa_order # s_w^n
                slab[:] = tmp

            wfield = ComplexField(self.pm)
            wfield[:] = 0.
            for mout in range(-ellout, ellout+1):
                Ylm = get_real_Ylm(ellout, mout)
                qfield = self._get_q(ellout=ellout, mout=mout, projin=projin)
                qfield[:] *= dfield[:]
                cfield = qfield.r2c()
                for islab, slab in enumerate(cfield.slabs):
                    slab[:] *= 4.*np.pi * Ylm(self.khat[0][islab], self.khat[1][islab], self.khat[2][islab])
                wfield[:] += cfield[:]

            del dfield
            proj_result = project_to_basis(wfield, self.edges, antisymmetric=bool(ellout % 2))[0]
            result.append(np.squeeze(proj_result[2]))
            k, nmodes = proj_result[0], proj_result[-1]

        poles = self.nmesh.prod()**2 * np.array([result[ells.index(ell)] for ell in self.ells]).conj()
        if swap: poles = poles.conj()
        k, nmodes = np.squeeze(k), np.squeeze(nmodes)
        kwargs = {'wnorm':self.wnorm, 'shotnoise_nonorm':0., 'attrs':self.attrs}
        self.poles = PowerSpectrumMultipoles(modes=k, edges=self.edges[0], power_nonorm=poles, nmodes=nmodes, ells=self.ells, **kwargs)

        if swap: self.mesh1, self.mesh2 = self.mesh2, self.mesh1

    def _run_global_los(self, projin, deriv):
        # projin is \ell^\prime
        # deriv is \xi^{\ell^{\prime},\beta \ell^\prime}(s^w)
        qfield = self.qfield.copy()

        legendre = special.legendre(projin.ell)
        for islab, slab in enumerate(qfield.slabs):
            tmp = np.zeros_like(self.xwnorm[islab])
            mask_nonzero = self.xwnorm[islab] != 0.
            tmp[mask_nonzero] = deriv(self.xwnorm[islab][mask_nonzero])
            if projin.ell:
                mu = sum(xx[islab]*ll for xx, ll in zip(self.xwhat, self.los))
                tmp *= legendre(mu)
            slab[:] *= tmp

        wfield = qfield.r2c()

        result, result_poles = project_to_basis(wfield, self.edges, ells=self.ells, los=self.los)
        # Format the power results into :class:`PowerSpectrumWedges` instance
        kwargs = {'wnorm':self.wnorm, 'shotnoise_nonorm':0., 'attrs':self.attrs}
        k, mu, power, nmodes = (np.squeeze(result[ii]) for ii in [0,1,2,3])
        power = self.nmesh.prod()**2 * power.conj()
        self.wedges = PowerSpectrumWedges(modes=(k, mu), edges=self.edges, power_nonorm=power, nmodes=nmodes, **kwargs)

        if result_poles:
            # Format the power results into :class:`PolePowerSpectrum` instance
            k, power, nmodes = (np.squeeze(result_poles[ii]) for ii in [0,1,2])
            power = self.nmesh.prod()**2 * power.conj()
            self.poles = PowerSpectrumMultipoles(modes=k, edges=self.edges[0], power_nonorm=power, nmodes=nmodes, ells=self.ells, **kwargs)

    def _run_periodic(self, projin, deriv):
        legendre = special.legendre(projin.ell)
        for islab, slab in enumerate(self.qfield.slabs):
            tmp = deriv(self.knorm[islab])
            if projin.ell:
                mu = sum(xx[islab]*ll for xx, ll in zip(self.khat, self.los))
                tmp *= legendre(mu)
            slab[:] = tmp

        result, result_poles = project_to_basis(self.qfield, self.edges, ells=self.ells, los=self.los)
        # Format the power results into :class:`PowerSpectrumWedges` instance
        kwargs = {'wnorm':self.wnorm, 'shotnoise_nonorm':0., 'attrs':self.attrs}
        k, mu, power, nmodes = (np.squeeze(result[ii]) for ii in [0,1,2,3])
        self.wedges = PowerSpectrumWedges(modes=(k, mu), edges=self.edges, power_nonorm=power, nmodes=nmodes, **kwargs)

        if result_poles:
            # Format the power results into :class:`PolePowerSpectrum` instance
            k, power, nmodes = (np.squeeze(result_poles[ii]) for ii in [0,1,2])
            self.poles = PowerSpectrumMultipoles(modes=k, edges=self.edges[0], power_nonorm=power, nmodes=nmodes, ells=self.ells, **kwargs)

    def run(self):

        def _wrap_rslab(rslab):
            # We do not use the same conventions as pmesh:
            # rslab < 0 is sent back to [boxsize/2, boxsize]
            toret = []
            for ii, rr in enumerate(rslab):
                mask = rr > self.boxsize[ii]/2.
                rr[mask] -= self.boxsize[ii]
                toret.append(rr)
            return toret

        def _safe_divide(num, denom):
            with np.errstate(divide='ignore', invalid='ignore'):
                toret = num/denom
            toret[denom == 0.] = 0.
            return toret

        if self.periodic:
            self.qfield = ComplexField(self.pm)
            # The Fourier-space grid
            self.khat = [kk.real.astype('f8') for kk in ComplexField(self.pm).slabs.optx]
            self.knorm = np.sqrt(sum(kk**2 for kk in self.khat))
            self.khat = [_safe_divide(kk, self.knorm) for kk in self.khat]
        else:
            self.xwhat = [xx.real.astype('f8') for xx in _wrap_rslab(_transform_rslab(self.mesh1.slabs.optx, self.boxsize))] # this should just give self.mesh1.slabs.optx
            self.xwnorm = np.sqrt(sum(xx**2 for xx in self.xwhat))
            self.xwhat = [_safe_divide(xx, self.xwnorm) for xx in self.xwhat]

        if self.los_type == 'global': # global (fixed) line-of-sight

            if self.periodic:
                run_projin = self._run_periodic

            else:
                cfield2 = cfield1 = self.mesh1.r2c()
                # Set mean value or real field to 0
                #for i, c1 in zip(cfield1.slabs.i, cfield1.slabs):
                #    mask_zero = True
                #    for ii in i: mask_zero = mask_zero & (ii == 0)
                #    c1[mask_zero] = 0.

                if self.autocorr:
                    self._compensate(cfield1, self.compensations[0])
                else:
                    # We apply all compensation transfer functions to cfield1
                    self._compensate(cfield1, *self.compensations)
                    cfield2 = self.mesh2.r2c()

                for islab in range(cfield1.shape[0]):
                    cfield1[islab,...] = cfield1[islab].conj() * cfield2[islab]
                # No 1/N^6 factor due to pmesh convention
                self.qfield = cfield1.c2r()
                del self.mesh2, self.mesh1, cfield2, cfield1
                run_projin = self._run_global_los

        else: # local (varying) line-of-sight

            if self.autocorr:
                self.cfield2 = self.mesh1.r2c()
                compensations = [self.compensations[0]] * 2
            else:
                self.cfield2 = self.mesh2.r2c()
                compensations = self.compensations
            # We apply all compensation transfer functions to cfield2
            self._compensate(self.cfield2, *compensations)
            for i, c in zip(self.cfield2.slabs.i, self.cfield2.slabs):
                mask_zero = True
                for ii in i: mask_zero = mask_zero & (ii == 0)
                c[mask_zero] = 0.

            # Offset the box coordinate mesh ([-BoxSize/2, BoxSize]) back to the original (x,y,z) coords
            offset = self.boxcenter - self.boxsize/2.
            self.xhat = [xx.real.astype('f8') + offset[ii] for ii, xx in enumerate(_transform_rslab(self.mesh1.slabs.optx, self.boxsize))]
            self.xnorm = np.sqrt(sum(xx**2 for xx in self.xhat))
            self.xhat = [_safe_divide(xx, self.xnorm) for xx in self.xhat]

            # The Fourier-space grid
            self.khat = [kk.real.astype('f8') for kk in ComplexField(self.pm).slabs.optx]
            knorm = np.sqrt(sum(kk**2 for kk in self.khat))
            self.khat = [_safe_divide(kk, knorm) for kk in self.khat]
            del knorm

            run_projin = self._run_local_los

        poles, wedges = [], []
        for projin in self.projsin:
            poles_x, wedges_x = [], []
            for iin, xin in enumerate(self.xin[projin]):
                run_projin(projin, self.deriv[projin][iin])
                if self.ells:
                    poles_x.append(PowerSpectrumFFTWindowMatrix.from_power(self.poles, xin, projin))
                if self.los_type == 'global':
                    wedges_x.append(PowerSpectrumFFTWindowMatrix.from_power(self.wedges, xin, projin))
            if poles_x:
                poles.append(PowerSpectrumFFTWindowMatrix.concatenate_x(*poles_x, axis='in'))
            if wedges_x:
                wedges.append(PowerSpectrumFFTWindowMatrix.concatenate_x(*wedges_x, axis='in'))
        if poles:
            self.poles = PowerSpectrumFFTWindowMatrix.concatenate_proj(*poles, axis='in')
        if wedges:
            self.wedges = PowerSpectrumFFTWindowMatrix.concatenate_proj(*wedges, axis='in')

        for name in ['mesh1', 'mesh2', 'cfield2', 'qfield']:
            if hasattr(self, name): delattr(self, name)

    @classmethod
    def concatenate_proj(cls, *others):
        new = others[0].copy()
        for name in ['poles', 'wedges']:
            if hasattr(others[0], name):
                setattr(new, name, PowerSpectrumFFTWindowMatrix.concatenate_proj(*[getattr(other, name) for other in others], axis='in'))
        return new

    @classmethod
    def concatenate_x(cls, *others):
        new = others[0].copy()
        for name in ['poles', 'wedges']:
            if hasattr(others[0], name):
                setattr(new, name, PowerSpectrumFFTWindowMatrix.concatenate_x(*[getattr(other, name) for other in others], axis='in'))
        return new

    def __setstate__(self, state):
        """Set this class state."""
        super(MeshFFTPower, self).__setstate__(state) # MeshFFTPower to get BaseClass.__setstate__(state)
        for name in ['wedges', 'poles']:
            if name in state:
                setattr(self, name, PowerSpectrumFFTWindowMatrix.from_state(state[name]))


class CatalogFFTWindow(MeshFFTWindow):

    """Wrapper on :class:`MeshFFTWindow` to estimate window function from input random positions and weigths."""

    def __init__(self, randoms_positions1=None, randoms_positions2=None,
                randoms_weights1=None, randoms_weights2=None,
                edgesin=None, projsin=None, edges=None, ells=None, power_ref=None,
                los=None, nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=2., wrap=False, dtype=None,
                resampler=None, interlacing=None, position_type='xyz', weight_type='auto', weight_attrs=None,
                wnorm=None, mpiroot=None, mpicomm=mpi.COMM_WORLD):
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
            An array of :math:`k`-edges which defines the theory :math:`k`-binning; corresponding derivatives will be computed
            using :func:`get_correlation_function_tophat_derivative`; or a dictionary of such array for each theory projection.
            Else a list of derivatives (callable) of theory correlation function w.r.t. each theory basis vector, e.g. each in :math:`k`-bin;
            or a dictionary of such list for each theory projection.

        projsin : list, default=None
            List of :class:`Projection` instances or (multipole, wide-angle order) tuples.
            If ``None``, and ``power_ref`` is provided, the list of projections is set
            to be able to compute window convolution of theory power spectrum multipoles of orders ``power_ref.ells``.

        power_ref : PowerSpectrumMultipoles, default=None
            "Reference" power spectrum estimation, e.g. of the actual data.
            It is used to set default values for ``edges``, ``ells``, ``los``, ``boxsize``, ``boxcenter``, ``nmesh``,
            ``interlacing``, ``resampler`` and ``wnorm`` if those are ``None``.

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

        wrap : bool, default=False
            Whether to wrap input positions?
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
            attrs_ref = _get_attr_in_inst(power_ref, 'attrs', insts=(None, 'poles', 'wedges'))
            for name in mesh_names:
                mesh_attrs.setdefault(name, attrs_ref[name])
            if interlacing is None:
                interlacing = tuple(attrs_ref['interlacing{:d}'.format(i+1)] for i in range(2))
            if resampler is None:
                resampler = tuple(attrs_ref['resampler{:d}'.format(i+1)] for i in range(2))
            if dtype is None: dtype = attrs_ref.get('dtype', 'f8')

        if dtype is None: dtype = 'f8'
        rdtype = _get_real_dtype(dtype)

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
        nmesh, boxsize, boxcenter = _get_box(**mesh_attrs, positions=bpositions, boxpad=boxpad, check=not wrap, mpicomm=mpicomm)
        if resampler is None: resampler = 'cic'
        if interlacing is None: interlacing = 2
        if not isinstance(resampler, tuple):
            resampler = (resampler,)*2
        if not isinstance(interlacing, tuple):
            interlacing = (interlacing,)*2

        if wrap:
            for position in positions.values():
                if position is not None:
                    _wrap_in_place(position, boxsize, boxcenter - boxsize/2.)

        if wnorm is None and power_ref is not None:
            wsum = [mpicomm.allreduce(sum(weights['R1']) if weights['R1'] is not None else len(positions['R1']))]*2
            if not autocorr: wsum[1] = mpicomm.allreduce(sum(weights['R2']) if weights['R2'] is not None else len(positions['R2']))
            ialpha2 = np.prod([wsum[ii]/attrs_ref[name] for ii, name in enumerate(['sum_data_weights1', 'sum_data_weights2'])])
            wnorm = ialpha2 * _get_attr_in_inst(power_ref, 'wnorm', insts=(None, 'poles', 'wedges'))

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
        super(CatalogFFTWindow, self).__init__(mesh1=mesh1, mesh2=mesh2, edgesin=edgesin, projsin=projsin, power_ref=power_ref, edges=edges, ells=ells, los=los, wnorm=wnorm)
