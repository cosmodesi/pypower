"""
Reimplement nbodykit's algorithm with support cross-correlations.
Normalization is now computed as a sum on the mesh, hence no nbar column is needed.
k-binning can be custom (note one cannot use :meth:`BinnedStatistic.reindex` method).

Most of the code is copy-pasted from `nbodykit
<https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower>`_
Changes w.r.t. original algorithm are highlighted with # CHANGE.
The project may be to have these changes integrated in the original code.

Ultimately we would like to provide a user-interface similar
to that of https://github.com/cosmodesi/pyrecon.
"""

import logging
import warnings
import time

import numpy as np

from pmesh.pm import ComplexField
from nbodykit.utils import attrs_to_dict, timer
from nbodykit.source.catalog import MultipleSpeciesCatalog, ArrayCatalog
from nbodykit.source.mesh import MultipleSpeciesCatalogMesh
from nbodykit.binned_statistic import BinnedStatistic
from nbodykit.algorithms.fftpower import project_to_basis, _find_unique_edges
from nbodykit.algorithms.convpower.catalog import FKPCatalog as _FKPCatalog
from nbodykit.algorithms.convpower.catalogmesh import FKPCatalogMesh as _FKPCatalogMesh
from nbodykit.algorithms.convpower.fkp import ConvolvedFFTPower as _ConvolvedFFTPower
from nbodykit.algorithms.convpower.fkp import _cast_mesh, get_real_Ylm, copy_meta, get_compensation


class ConvolvedFKPFFTPower(_ConvolvedFFTPower):
    """
    Algorithm to compute power spectrum multipoles using FFTs
    for a data survey with non-trivial geometry.

    This extends `nbodykit's algorithm
    <https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py>`_
    to cross-correlations and revisits the normalization.

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
    documentation of :func:`~ConvolvedFKPFFTPower.run`.

    Parameters
    ----------
    first : FKPCatalog, FKPCatalogMesh
        the first source to paint the data/randoms; FKPCatalog is automatically
        converted to a FKPCatalogMesh, using default painting parameters
    poles : list of int
        a list of integer multipole numbers ``ell`` to compute
    second : FKPCatalog, FKPCatalogMesh, default=None
        the second source to paint the data/randoms, for cross-correlations.
    edges : list, array, dict, default=None
        array of wavenumber edges, or a dictionary with (optional) keys:
            - min: the edge of the first wavenumber bin; default is 0
            - max: the limit of the last wavenumber bin; default is None, no limit.
            - step: the spacing in wavenumber to use; if not provided; the fundamental mode of the box is used
    same_noise : string, default=None
        Only considered in the case of cross-correlations; whether the two data (resp. randoms) fields
        are a same Poisson realization (of the same size): set to 'data' (resp. 'randoms').
        If the two data fields *and* the two randoms fields are the same Poisson realization, set to 'both'.
        If ``None``, the two data and randoms fields are assumed to *not* be the same Poisson realization,
        i.e. shot noise is set to 0.

    References
    ----------
    * Hand, Nick et al. `An optimal FFT-based anisotropic power spectrum estimator`, 2017
    * Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
      MNRAS, 2015
    * Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
    """
    logger = logging.getLogger('ConvolvedFKPFFTPower')

    def __init__(self, first, poles,
                 second=None,
                 Nmesh=None,
                 edges=None,
                 same_noise=None):

        first = _cast_mesh(first, Nmesh=Nmesh)
        if second is not None:
            second = _cast_mesh(second, Nmesh=Nmesh)
        else:
            second = first

        # CHANGE
        isauto = second is first

        self.same_noise = {'data':isauto,'randoms':isauto}
        if not isauto:
            if same_noise == 'both':
                self.same_noise['data'] = self.same_noise['randoms'] = True
            elif same_noise:
                self.same_noise[same_noise] = True

        self.first = first
        self.second = second

        # grab comm from first source
        self.comm = first.comm

        # check for comm mismatch
        assert second.comm is first.comm, "communicator mismatch between input sources"

        # CHANGE
        # make a box big enough for both catalogs if they are not equal
        def extent(source):
            return [source.attrs['BoxCenter'] - source.attrs['BoxSize']/2, source.attrs['BoxCenter'] + source.attrs['BoxSize']/2]

        first_extent = extent(first)
        second_extent = extent(second)

        # CHANGE
        if not np.array_equal(first_extent, second_extent):

            # determine max box length along each dimension
            joint_extent = [np.min([first_extend[0],second_extent[0]],axis=0),
                            np.max([first_extend[-1],second_extent[-1]],axis=0)]

            boxsize = joint_extent[1] - joint_extent[0]
            boxcenter = (joint_extent[0] + joint_extent[1])/2.

            # re-center the box
            first.recenter_box(boxsize, boxcenter)
            second.recenter_box(boxsize, boxcenter)

        # make a list of multipole numbers
        if np.ndim(poles) == 0:
            poles = [poles]

        # store meta-data
        self.attrs = {}
        self.attrs['poles'] = poles
        self.attrs['edges'] = edges

        # store BoxSize and BoxCenter from source
        self.attrs['Nmesh'] = self.first.attrs['Nmesh'].copy()
        self.attrs['BoxSize'] = self.first.attrs['BoxSize']
        self.attrs['BoxPad'] = self.first.attrs['BoxPad']
        self.attrs['BoxCenter'] = self.first.attrs['BoxCenter']

        # grab some mesh attrs, too
        self.attrs['mesh.resampler'] = self.first.resampler
        self.attrs['mesh.interlaced'] = self.first.interlaced

        # and run
        self.run()

    def run(self):
        """
        Compute the power spectrum multipoles. This function does not return
        anything, but adds several attributes (see below).

        Attributes
        ----------
        edges : array_like
            the edges of the wavenumber bins
        poles : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object that behaves similar to a structured array, with
            fancy slicing and re-indexing; it holds the measured multipole
            results, as well as the number of modes (``modes``) and average
            wavenumbers values in each bin (``k``)
        attrs : dict
            dictionary holding input parameters and several important quantites
            computed during execution:

            #. data.N, randoms.N :
                the unweighted number of data and randoms objects
            #. data.W, randoms.W :
                the weighted number of data and randoms objects, using the
                column specified as the weights
            #. alpha :
                the ratio of ``data.W`` to ``randoms.W``
            #. norm :
                the normalization of the power spectrum.
            #. data.shotnoise, randoms.shotnoise :
                the shot noise values for the "data" and "random" catalogs;
                See equation 15 of arxiv:1312.4611.
            #. shotnoise :
                the total shot noise for the power spectrum, equal to
                ``data.shotnoise`` + ``randoms.shotnoise``; this should be subtracted from
                the monopole.
            #. BoxSize :
                the size of the Cartesian box used to grid the data and
                randoms objects on a Cartesian mesh.
        """
        pm = self.first.pm

        edges = self.attrs['edges']
        kcoords = None
        if isinstance(edges,dict):
            dk = 2*np.pi/pm.BoxSize.min() if edges.get('step',None) is None else edges['step']
            kmin = edges.get('min',0.)
            kmax = edges.get('max',np.pi*pm.Nmesh.min()/pm.BoxSize.max() + dk/2)
            if dk > 0:
                kedges = np.arange(kmin, kmax, dk)
            else:
                k = pm.create_coords('complex')
                kedges, kcoords = _find_unique_edges(k, 2 * np.pi / pm.BoxSize, kmax, pm.comm)
                if self.comm.rank == 0:
                    self.logger.info('%d unique k values are found' % len(kcoords))
        else:
            kedges = np.array(edges)

        # measure the binned 1D multipoles in Fourier space
        result = self._compute_multipoles(kedges)

        # set all the necessary results
        self.poles = BinnedStatistic(['k'], [kedges], result,
                            fields_to_sum=['modes'],
                            coords=[kcoords],
                            **{key:value for key,value in self.attrs.items() if key != 'edges'})

        self.edges = self.attrs['edges'] = kedges

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
        # clear compensation from the actions
        for source in [self.first, self.second]:
            source.actions[:] = []; source.compensated = False
            assert len(source.actions) == 0

        # compute the compensations
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

        # CHANGE
        # store alpha: ratio of data to randoms
        self.attrs['first.alpha'] = self.attrs['second.alpha'] = meta1['alpha']

        # FFT 1st density field and apply the resampler transfer kernel
        cfield = rfield1.r2c()
        if compensation['first'] is not None:
            cfield.apply(out=Ellipsis, **compensation['first'])
            print('LOOOOOOL',compensation['first'])
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

            self.attrs['second.alpha'] = meta2['alpha']

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
        self.attrs['norm'] = self.normalization()

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

        # compute shot noise
        self.attrs['shotnoise'] = self.shotnoise()

        # copy over any painting meta data
        if self.first is self.second:
            copy_meta(self.attrs, meta1)
        else:
            copy_meta(self.attrs, meta1, prefix='first')
            copy_meta(self.attrs, meta2, prefix='second')

        return result

    def normalization(self):
        # CHANGE
        r"""
        Compute the power spectrum normalization.
        This differs from the original `nbodykit's algorithm
        <https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py>`_.

        The normalization is given by:

        .. math::

            A = dV \frac{\alpha_{2} \sum_{i} n_{d,1}^{i} n_{r,2}^{i} + \alpha_{1} \sum_{i} n_{d,2}^{i} n_{r,1}^{i}}{2}

        :math:`n_{d,1}^{i}` and :math:`n_{r,1}^{i}` are the first data and randoms density, as obtained by
        painting data :math:`w_{d}` and random weights :math:`w_{r}` on the same mesh (of cell volume :math:`dV`) as the FFTs,
        using the Nearest Grid Point assignment scheme. The same applies to second data and randoms density. The sum then runs over the mesh cells.
        :math:`\alpha_{1} = \sum_{i} w_{d,1}^{i} / \sum_{i} w_{r,1}^{i}` and :math:`\alpha_{2} = \sum_{i} w_{d,2}^{i} / \sum_{i} w_{r,2}^{i}`
        where the sum of weights is performed over the catalogs.

        Note
        ----
        In progress.
        """
        # the selection (same for first/second)
        vol_per_cell = (self.attrs['BoxSize']/self.attrs['Nmesh']).prod()
        old_resampler = {}
        for name in ['first','second']:
            old_resampler[name] = getattr(self,name).resampler
            getattr(self,name).resampler = 'nnb'
        norm = self.attrs['second.alpha']*self.first['data'].to_real_field(normalize=False)*self.second['randoms'].to_real_field(normalize=False)
        norm = norm.csum().real/vol_per_cell # meshes are not normalized by cell-volume, so *dV => /dV
        if self.second is not self.first:
            norm2 = self.attrs['first.alpha']*self.second['data'].to_real_field(normalize=False)*self.first['randoms'].to_real_field(normalize=False)
            norm2 = norm2.csum().real/vol_per_cell
            norm = (norm + norm2)/2.
        for name in ['first','second']:
            getattr(self,name).resampler = old_resampler[name]
        return norm

    def _unnormalized_shotnoise(self, name='data'):
        # CHANGE
        r"""
        Compute the power spectrum shot noise, using either the ``data`` or ``randoms`` source.

        If first and second catalog share are a same Poisson realisation, this computes:

        .. math::

            S = \sum_{i} w_{1}^{i} w_{2}^{i}

        else, returns 0.
        """
        Pshot = 0
        if not self.same_noise[name]:
            return Pshot

        # the selection (same for first/second)
        sel = self.first.source.compute(self.first.source[name][self.first.selection])

        # selected first/second meshes for "name" (data or randoms)
        first = self.first.source[name][sel]
        second = self.second.source[name][sel]

        weight1 = first[self.first.weight]
        weight2 = second[self.second.weight]

        Pshot = np.sum(weight1*weight2)

        # reduce sum across all ranks
        Pshot = self.comm.allreduce(first.compute(Pshot))

        # divide by normalization from randoms
        return Pshot

    def shotnoise(self):
        # CHANGE
        r"""
        Compute the (normalized) power spectrum shot noise, summed over data and randoms.
        Set 'data.shotnoise', 'randoms.shotnoise' and 'shotnoise' in :attr:`attrs`.
        """
        Pshot = 0
        for name in ['data', 'randoms']:

            S = self._unnormalized_shotnoise(name) / self.attrs['norm']
            if name == 'randoms':
                alpha2 = self.attrs['first.alpha']*self.attrs['second.alpha']
                S *= alpha2
            self.attrs['%s.shotnoise' % name] = S
            Pshot += S # add to total

        self.attrs['shotnoise'] = Pshot
        return Pshot



class FKPCatalog(_FKPCatalog):
    """
    An interface for simultaneous modeling of a ``data`` CatalogSource and a
    ``randoms`` CatalogSource, in the spirit of
    `Feldman, Kaiser, and Peacock, 1994 <https://arxiv.org/abs/astro-ph/9304022>`_.
    This removes requirement for nbar from `nbodykit's class
    <https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/catalog.py>`_.

    This main functionality of this class is:

    *   provide a uniform interface to accessing columns from the
        ``data`` CatalogSource and ``randoms`` CatalogSource, using
        column names prefixed with "data/" or "randoms/"
    *   compute the shared :attr:`BoxSize` of the source, by
        finding the maximum Cartesian extent of the ``randoms``
    *   provide an interface to a mesh object, which knows how to paint the
        FKP density field from the ``data`` and ``randoms``

    Parameters
    ----------
    data : CatalogSource
        the CatalogSource of particles representing the `data` catalog
    randoms : CatalogSource, default=None
        the CatalogSource of particles representing the `randoms` catalog
        if None is given an empty catalog is used.
    BoxSize : float, 3-vector, optional
        the size of the Cartesian box to use for the unified `data` and
        `randoms`; if not provided, the maximum Cartesian extent of the
        `randoms` defines the box
    BoxPad : float, 3-vector, optional
        optionally apply this additional buffer to the extent of the
        Cartesian box

    References
    ----------
    - `Feldman, Kaiser, and Peacock, 1994 <https://arxiv.org/abs/astro-ph/9304022>`__
    """
    logger = logging.getLogger('FKPCatalog')

    def __repr__(self):
        return "FKPCatalog(species=%s)" %str(self.attrs['species'])

    def __init__(self, data, randoms, BoxSize=None, BoxPad=0.02):
        # CHANGE

        if randoms is None:
            # create an empty catalog.
            randoms = data[:0]

        # init the base class
        MultipleSpeciesCatalog.__init__(self, ['data', 'randoms'], data, randoms)

        self.attrs['BoxSize'] = np.empty(3,dtype='f8')
        self.attrs['BoxSize'][:] = BoxSize
        self.attrs['BoxPad'] = np.empty(3,dtype='f8')
        self.attrs['BoxPad'][:] = BoxSize

    def to_mesh(self, Nmesh=None, BoxSize=None, BoxCenter=None, dtype='c16', interlaced=False,
                compensated=False, resampler='cic', fkp_weight='FKPWeight',
                weight='Weight', selection='Selection',
                position='Position', bbox_from_species=None):
        # CHANGE
        """
        Convert the FKPCatalog to a mesh, which knows how to "paint" the
        FKP density field.

        Additional keywords to the :func:`to_mesh` function include the
        FKP weight column, completeness weight column, and the column
        specifying the number density as a function of redshift.

        Parameters
        ----------
        Nmesh : int, 3-vector, optional
            the number of cells per box side; if not specified in `attrs`, this
            must be provided
        dtype : str, dtype, optional
            the data type of the mesh when painting. dtype='f8' or 'f4' assumes
            Hermitian symmetry of the input field (\delta(x) =
            \delta^{*}(-x)), and stores it as an N x N x N/2+1 real array.
            This speeds evaluation of even multipoles but yields
            incorrect odd multipoles in the presence of the wide-angle effect.
            dtype='c16' or 'c8' stores the field as an N x N x N complex array
            to correctly recover the odd multipoles.
        interlaced : bool, optional
            whether to use interlacing to reduce aliasing when painting the
            particles on the mesh
        compensated : bool, optional
            whether to apply a Fourier-space transfer function to account for
            the effects of the gridding + aliasing
        resampler : str, optional
            the string name of the resampler to use when interpolating the
            particles to the mesh; see ``pmesh.window.methods`` for choices
        weight : str, optional
            the name of the column in the source specifying the total
            weight, e.g. completness weight times FKP weight;
            this weight is applied to the individual fields, either
            ``n_{d}``  or ``n_{r}``.
        selection : str, optional
            the name of the column used to select a subset of the source when
            painting
        position : str, optional
            the name of the column that specifies the position data of the
            objects in the catalog
        bbox_from_species: str, optional
            if given, use the species to infer a bbox.
            if not give, will try random, then data (if random is empty)
        """
        # verify that all of the required columns exist
        for name in self.species:
            for col in [weight]:
                if col not in self[name]:
                    raise ValueError("the '%s' species is missing the '%s' column" %(name, col))

        if Nmesh is None:
            try:
                Nmesh = self.attrs['Nmesh']
            except KeyError:
                raise ValueError("cannot convert FKP source to a mesh; 'Nmesh' keyword is not "
                                 "supplied and the FKP source does not define one in 'attrs'.")

        # first, define the Cartesian box
        if bbox_from_species is not None:
            BoxSize1, BoxCenter1 = self._define_bbox(position, selection, bbox_from_species)
        else:
            if self['randoms'].csize > 0:
                BoxSize1, BoxCenter1 = self._define_bbox(position, selection, "randoms")
            else:
                BoxSize1, BoxCenter1 = self._define_bbox(position, selection, "data")

        if BoxSize is None:
            BoxSize = BoxSize1

        if BoxCenter is None:
            BoxCenter = BoxCenter1

        # log some info
        if self.comm.rank == 0:
            self.logger.info("BoxSize = %s" %str(BoxSize))
            self.logger.info("BoxCenter = %s" %str(BoxCenter))

        # initialize the FKP mesh
        kws = {'Nmesh':Nmesh, 'BoxSize':BoxSize, 'BoxCenter':BoxCenter, 'dtype':dtype, 'selection':selection}
        return FKPCatalogMesh(self,
                              weight=weight,
                              position=position,
                              value='Value',
                              interlaced=interlaced,
                              compensated=compensated,
                              resampler=resampler,
                              **kws)

class FKPCatalogMesh(_FKPCatalogMesh):
    """
    A subclass of
    :class:`~nbodykit.source.catalogmesh.species.MultipleSpeciesCatalogMesh`
    designed to paint a :class:`~nbodykit.source.catalog.fkp.FKPCatalog` to
    a mesh.

    This removes requirement for nbar and weight_fkp from `nbodykit's class
    <https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/catalogmesh.py>`_.

    The multiple species here are ``data`` and ``randoms`` CatalogSource
    objects, where ``randoms`` is a catalog of randomly distributed objects
    with no instrinsic clustering that defines the survey volume.

    The position of the catalogs are re-centered to the ``[-L/2, L/2]``
    where ``L`` is the size of the Cartesian box.

    Parameters
    ----------
    source : CatalogSource
        the input catalog that we wish to interpolate to a mesh
    BoxSize : float, 3-vector
        the size of the box
    BoxCenter : float, 3-vector, default=None
        the box center
    Nmesh : int, 3-vector
        the number of cells per mesh side
    dtype : str
        the data type of the values stored on mesh
    selection : str
        column in ``source`` that selects the subset of particles to grid
        to the mesh
    weight : str
        the weight column name
    position : str, optional
        column in ``source`` specifying the position coordinates; default
        is ``Position``
    interlaced : bool, optional
        whether to use interlacing to reduce aliasing when painting the
        particles on the mesh
    compensated : bool, optional
        whether to apply a Fourier-space transfer function to account for
        the effects of the gridding + aliasing
    resampler : str, optional
        the string name of the resampler to use when interpolating the
        particles to the mesh; see ``pmesh.window.methods`` for choices
    """
    logger = logging.getLogger('FKPCatalogMesh')

    def __init__(self, source, BoxSize, BoxCenter, Nmesh, dtype, selection,
                    weight='Weight', value='Value',
                    position='Position', interlaced=False,
                    compensated=False, resampler='cic'):
        # CHANGE
        if not isinstance(source, FKPCatalog):
            raise TypeError("the input source for FKPCatalogMesh must be a FKPCatalog")

        uncentered_position = position
        position = '_RecenteredPosition'

        self.attrs.update(source.attrs)

        self.recenter_box(BoxSize, BoxCenter)

        MultipleSpeciesCatalogMesh.__init__(self, source=source,
                        BoxSize=BoxSize, Nmesh=Nmesh,
                        dtype=dtype, weight=weight, value=value, selection=selection, position=position,
                        interlaced=interlaced, compensated=compensated, resampler=resampler)

        self._uncentered_position = uncentered_position
        self.weight = weight

    def TotalWeight(self, name):
        # CHANGE
        """The total weight for the mesh."""
        assert name in ['data', 'randoms']
        return self.source[name][self.weight]

    def weighted_total(self, name):
        # CHANGE
        r"""
        Compute the weighted total number of objects, using either the
        ``data`` or ``randoms`` source.

        This is the sum of weights:

        .. math::

            W = \sum w_{i}
        """
        # the selection
        sel = self.source.compute(self.source[name][self.selection])

        # the selected mesh for "name"
        selected = self.source[name][sel]

        # sum up completeness weights
        wsum = self.source.compute(selected[self.weight].sum())
        return self.comm.allreduce(wsum)
