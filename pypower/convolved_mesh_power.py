"""
This algorithm directly takes mesh as input.
No normalization nor shotnoise is computed.
Changes w.r.t. original algorithm are highlighted with # CHANGE.
"""
import logging
import time
import warnings

import numpy as np

from pmesh.pm import ComplexField
from nbodykit import CurrentMPIComm
from nbodykit.utils import timer
from nbodykit.binned_statistic import BinnedStatistic
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.source.mesh.catalog import get_compensation
from nbodykit.algorithms.fftpower import project_to_basis, _find_unique_edges, _cast_source
from nbodykit.algorithms.convpower.fkp import get_real_Ylm, copy_meta, get_compensation
from .convolved_fkp_power import ConvolvedFKPFFTPower


class ConvolvedMeshFFTPower(ConvolvedFKPFFTPower):
    """
    Algorithm to compute power spectrum multipoles using FFTs
    for a data survey with non-trivial geometry.

    Contrary to :class:`ConvolvedFKPFFTPower`, input are meshes.
    No normalization nor shot noise is computed.

    Due to the geometry, the estimator computes the true power spectrum
    convolved with the window function (FFT of the geometry).

    This estimator implemented in this class is described in detail in
    Hand et al. 2017 (arxiv:1704.02357). It uses the spherical harmonic
    addition theorem such that only :math:`2\ell+1` FFTs are required to
    compute each multipole. This differs from the implementation in
    Bianchi et al. and Scoccimarro et al., which requires
    :math:`(\ell+1)(\ell+2)/2` FFTs.

    Results are computed when the object is inititalized, and the result is
    stored in the :attr:`poles` attribute. Important meta-data computed
    during algorithm execution is stored in the :attr:`attrs` dict. See the
    documentation of :func:`~ConvolvedMeshFFTPower.run`.

    Parameters
    ----------
    first : MeshSource
        The first source.
    poles : list of int
        A list of integer multipole numbers ``ell`` to compute.
    second : FKPCatalog, FKPCatalogMesh, default=None
        The second source.
    edges : list, array, dict, default=None
        Array of wavenumber edges, or a dictionary with (optional) keys:
            - min: the edge of the first wavenumber bin; default is 0
            - max: the limit of the last wavenumber bin; default is None, no limit.
            - step: the spacing in wavenumber to use; if not provided; the fundamental mode of the box is used
    Nmesh : int, optional
        the number of cells per side in the particle mesh used to paint the source
    BoxSize : int, 3-vector, optional
        the size of the box
    BoxCenter : float, 3-vector, default=None
        The box center.
        If not provided, use ``first.attrs['BoxCenter']``.
    norm : float, default=None
        Normalization to use, in unit of 1/volume.
        If not provided, use ``first.attrs['norm']`` if exists, else 1.
    shotnoise : float, default=None
        Shot noise to use, in unit of 1/volume.
        If not provided, use ``first.attrs['shotnoise']`` if exists, else 0.
    """
    logger = logging.getLogger('ConvolvedMeshFFTPower')

    def __init__(self, first, poles,
                 second=None,
                 edges=None,
                 Nmesh=None,
                 BoxSize=None,
                 BoxCenter=None,
                 norm=None,
                 shotnoise=None):

        first = _cast_source(first, Nmesh=Nmesh, BoxSize=BoxSize)
        if second is not None:
            second = _cast_source(second, Nmesh=Nmesh, BoxSize=BoxSize)
        else:
            second = first

        self.first = first
        self.second = second

        # grab comm from first source
        self.comm = first.comm

        # check for comm mismatch
        assert second.comm is first.comm, "communicator mismatch between input sources"

        # make a list of multipole numbers
        if np.ndim(poles) == 0:
            poles = [poles]

        # store meta-data
        self.attrs = {}
        self.attrs['poles'] = poles
        self.attrs['edges'] = edges
        if norm is not None:
            self.attrs['norm'] = norm
        else:
            self.attrs['norm'] = self.first.attrs.get('norm',1)
        if shotnoise is not None:
            self.attrs['shotnoise'] = shotnoise
        else:
            self.attrs['shotnoise'] = self.first.attrs.get('shotnoise',0.)

        self.attrs['BoxCenter'] = np.empty(3,dtype='f8')
        if BoxCenter is not None:
            self.attrs['BoxCenter'][:] = BoxCenter
        else:
            self.attrs['BoxCenter'][:] = self.first.attrs['BoxCenter']

        # store BoxSize and BoxCenter from source
        self.attrs['Nmesh'] = self.first.attrs['Nmesh'].copy()
        self.attrs['BoxSize'] = self.first.attrs['BoxSize'].copy()

        # and run
        self.run()


    def _compute_multipoles(self, kedges):
        """
        Compute the window-convoled power spectrum multipoles, for a data set
        with non-trivial survey geometry.

        This estimator builds upon the work presented in Bianchi et al. 2015
        and Scoccimarro et al. 2015, but differs in the implementation. This
        class uses the spherical harmonic addition theorem such that
        only :math:`2\ell+1` FFTs are required per multipole, rather than the
        :math:`(\ell+1)(\ell+2)/2` FFTs in the implementation presented by
        Bianchi et al. and Scoccimarro et al.

        References
        ----------
        * Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
          MNRAS, 2015
        * Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
        """
        for mesh in [self.first, self.second]:
            # clear compensation from the actions
            # this will clear all actions as well...
            mesh.actions[:] = []; mesh.compensated = False
            assert len(mesh.actions) == 0

        compensation = {}
        for name, mesh in zip(['first', 'second'], [self.first, self.second]):
            compensation[name] = get_compensation(mesh)
            if self.comm.rank == 0:
                if compensation[name] is not None:
                    args = (compensation[name]['func'].__name__, name)
                    self.logger.info("using compensation function %s for source '%s'" % args)
                else:
                    self.logger.warning("no compensation applied for source '%s'" % name)

        rank = self.comm.rank
        pm   = self.first.pm

        # setup the 1D-binning
        muedges = np.linspace(-1, 1, 2, endpoint=True)
        edges = [kedges, muedges]

        # make a structured array to hold the results
        cols   = ['k'] + ['power_%d' %l for l in sorted(self.attrs['poles'])] + ['modes']
        dtype  = ['f8'] + ['c8']*len(self.attrs['poles']) + ['i8']
        dtype  = np.dtype(list(zip(cols, dtype)))
        result = np.empty(len(kedges)-1, dtype=dtype)

        # offset the box coordinate mesh ([-BoxSize/2, BoxSize]) back to
        # the original (x,y,z) coords
        offset = self.attrs['BoxCenter'] + 0.5*pm.BoxSize / pm.Nmesh

        # always need to compute ell=0
        poles = sorted(self.attrs['poles'])
        if 0 not in poles:
            poles = [0] + poles
        assert poles[0] == 0

        # spherical harmonic kernels (for ell > 0)
        Ylms = [[get_real_Ylm(l,m) for m in range(-l, l+1)] for l in poles[1:]]

        # paint the 1st FKP density field to the mesh (paints: data - alpha*randoms, essentially)
        rfield1 = self.first.compute(Nmesh=self.attrs['Nmesh'])
        meta1 = rfield1.attrs.copy()
        if rank == 0:
            self.logger.info("%s painting of 'first' done" %self.first.resampler)

        # FFT 1st density field and apply the resampler transfer kernel
        cfield = rfield1.r2c()
        if compensation['first'] is not None:
            cfield.apply(out=Ellipsis, **compensation['first'])
        if rank == 0: self.logger.info('ell = 0 done; 1 r2c completed')

        # monopole A0 is just the FFT of the FKP density field
        # NOTE: this holds FFT of density field #1
        volume = pm.BoxSize.prod()
        A0_1 = ComplexField(pm)
        A0_1[:] = cfield[:] * volume # normalize with a factor of volume

        # paint second mesh too?
        if self.first is not self.second:

            # paint the second field
            rfield2 = self.second.compute(Nmesh=self.attrs['Nmesh'])
            meta2 = rfield2.attrs.copy()
            if rank == 0: self.logger.info("%s painting of 'second' done" %self.second.resampler)

            # need monopole of second field
            if 0 in self.attrs['poles']:

                # FFT density field and apply the resampler transfer kernel
                A0_2 = rfield2.r2c()
                A0_2[:] *= volume
                if compensation['second'] is not None:
                    A0_2.apply(out=Ellipsis, **compensation['second'])
        else:
            rfield2 = rfield1
            meta2 = meta1

            # monopole of second field is first field
            if 0 in self.attrs['poles']:
                A0_2 = A0_1

        # save the painted density field #2 for later
        density2 = rfield2.copy()

        # initialize the memory holding the Aell terms for
        # higher multipoles (this holds sum of m for fixed ell)
        # NOTE: this will hold FFTs of density field #2
        Aell = ComplexField(pm)

        # the real-space grid
        xgrid = [xx.astype('f8') + offset[ii] for ii, xx in enumerate(density2.slabs.optx)]
        xnorm = np.sqrt(sum(xx**2 for xx in xgrid))
        xgrid = [x/xnorm for x in xgrid]

        # the Fourier-space grid
        kgrid = [kk.astype('f8') for kk in cfield.slabs.optx]
        knorm = np.sqrt(sum(kk**2 for kk in kgrid)); knorm[knorm==0.] = np.inf
        kgrid = [k/knorm for k in kgrid]

        # CHANGE
        # revisited normalization

        if self.attrs['norm'] > 0:
            norm = 1.0 / self.attrs['norm']
            if rank == 0:
                self.logger.info("normalized power spectrum with `norm = %.6f`" % self.attrs['norm'])
        else:
            norm = 1.0
            if rank == 0:
                self.logger.info("normalization of power spectrum is neglected, as no random is provided.")

        # loop over the higher order multipoles (ell > 0)
        start = time.time()
        for iell, ell in enumerate(poles[1:]):

            # clear 2D workspace
            Aell[:] = 0.

            # iterate from m=-l to m=l and apply Ylm
            substart = time.time()
            for Ylm in Ylms[iell]:

                # reset the real-space mesh to the original density #2
                rfield2[:] = density2[:]

                # apply the config-space Ylm
                for islab, slab in enumerate(rfield2.slabs):
                    slab[:] *= Ylm(xgrid[0][islab], xgrid[1][islab], xgrid[2][islab])

                # real to complex of field #2
                rfield2.r2c(out=cfield)

                # apply the Fourier-space Ylm
                for islab, slab in enumerate(cfield.slabs):
                    slab[:] *= Ylm(kgrid[0][islab], kgrid[1][islab], kgrid[2][islab])

                # add to the total sum
                Aell[:] += cfield[:]

                # and this contribution to the total sum
                substop = time.time()
                if rank == 0:
                    self.logger.debug("done term for Y(l=%d, m=%d) in %s" %(Ylm.l, Ylm.m, timer(substart, substop)))

            # apply the compensation transfer function
            if compensation['second'] is not None:
                Aell.apply(out=Ellipsis, **compensation['second'])

            # factor of 4*pi from spherical harmonic addition theorem + volume factor
            Aell[:] *= 4*np.pi*volume

            # log the total number of FFTs computed for each ell
            if rank == 0:
                args = (ell, len(Ylms[iell]))
                self.logger.info('ell = %d done; %s r2c completed' %args)

            # calculate the power spectrum multipoles, slab-by-slab to save memory
            # NOTE: this computes (A0 of field #1) * (Aell of field #2).conj()
            for islab in range(A0_1.shape[0]):
                Aell[islab,...] = norm * A0_1[islab] * Aell[islab].conj()

            # project on to 1d k-basis (averaging over mu=[0,1])
            proj_result, _ = project_to_basis(Aell, edges)
            result['power_%d' %ell][:] = np.squeeze(proj_result[2])

        # summarize how long it took
        stop = time.time()
        if rank == 0:
            self.logger.info("higher order multipoles computed in elapsed time %s" %timer(start, stop))

        # also compute ell=0
        if 0 in self.attrs['poles']:

            # the 3D monopole
            for islab in range(A0_1.shape[0]):
                A0_1[islab,...] = norm*A0_1[islab]*A0_2[islab].conj()

            # the 1D monopole
            proj_result, _ = project_to_basis(A0_1, edges)
            result['power_0'][:] = np.squeeze(proj_result[2])

        # save the number of modes and k
        result['k'][:] = np.squeeze(proj_result[0])
        result['modes'][:] = np.squeeze(proj_result[-1])

        # copy over any painting meta data
        if self.first is self.second:
            copy_meta(self.attrs, meta1)
        else:
            copy_meta(self.attrs, meta1, prefix='first')
            copy_meta(self.attrs, meta2, prefix='second')

        return result
