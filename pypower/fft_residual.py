import numpy as np

from . import mpi
from .fft_power import CatalogFFTPower, normalization, unnormalized_shotnoise


class CatalogFFTResidual(CatalogFFTPower):

    r"""Wrapper on :class:`CatalogFFTPower` to estimate :math:`(D - R) R`."""

    def __init__(self, data_positions1, randoms_positions1=None, randoms_positions2=None, shifted_positions1=None,
                 data_weights1=None, randoms_weights1=None, randoms_weights2=None, shifted_weights1=None,
                 D1R2_twopoint_weights=None, mpiroot=None, mpicomm=mpi.COMM_WORLD, **kwargs):
        r"""
        Initialize :class:`CatalogFFTResidual`, i.e. estimate :math:`(D - R) R`.

        Parameters
        ----------
        data_positions1 : list, array
            Positions in the data catalog. Typically of shape (3, N) or (N, 3).

        randoms_positions1 : list, array, default=None
            Optionally, positions of the random catalog representing the first selection function.
            If no randoms are provided, selection function will be assumed uniform.

        randoms_positions2 : list, array, default=None
            Optionally (for cross-correlation), positions in the second randoms catalog. See ``randoms_positions1``.

        shifted_positions1 : array, default=None
            Optionally, in case of BAO reconstruction, positions of the first shifted catalog.

        data_weights1 : array of shape (N,), default=None
            Optionally, weights in the first data catalog.

        randoms_weights1 : array of shape (N,), default=None
            Optionally, weights in the first randoms catalog.

        randoms_weights2 : array of shape (N,), default=None
            Optionally (for cross-correlation), weights in the second randoms catalog.

        shifted_weights1 : array, default=None
            Optionally, weights of the first shifted catalog. See ``data_weights1``.

        D1R2_twopoint_weights : WeightTwoPointEstimator, default=None
            Weights to be applied to each pair of particles between first data catalog and second randoms catalog.
            See ``D1D2_twopoint_weights``.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=mpi.COMM_WORLD
            The MPI communicator.

        kwargs : dict
            Other arguments for :class:`CatalogFFTPower`.
        """
        def is_none(array):
            if mpicomm is None or mpiroot is None:
                return array is None
            return mpicomm.bcast(array is None, root=mpiroot)

        self.same_shotnoise_residual = is_none(randoms_positions2)
        if self.same_shotnoise_residual:
            randoms_positions2 = randoms_positions1
            if is_none(randoms_weights2):
                randoms_weights2 = randoms_weights1
            self.same_shotnoise_residual = is_none(shifted_positions1)
        if is_none(randoms_positions1) or is_none(randoms_positions2):
            raise ValueError('A random catalog must be provided')

        super(CatalogFFTResidual, self).__init__(data_positions1=data_positions1, randoms_positions1=randoms_positions1, data_positions2=randoms_positions2, shifted_positions1=shifted_positions1,
                                                 data_weights1=data_weights1, randoms_weights1=randoms_weights1, data_weights2=randoms_weights2, shifted_weights1=shifted_weights1,
                                                 D1R2_twopoint_weights=D1R2_twopoint_weights, mpiroot=mpiroot, mpicomm=mpicomm, **kwargs)

    def _set_shotnoise(self, shotnoise, shotnoise_nonorm=None, mesh1=None, mesh2=None):
        self.shotnoise = shotnoise
        if shotnoise is None:
            if shotnoise_nonorm is None:
                if self.same_shotnoise_residual:
                    mesh1c = mesh1.clone(data_positions=mesh1.randoms_positions, data_weights=mesh1.randoms_weights, position_type='pos')
                    shotnoise_nonorm = - mesh1.sum_data_weights / mesh1.sum_randoms_weights * unnormalized_shotnoise(mesh1c, mesh2)
                else:
                    shotnoise_nonorm = 0.
            self.shotnoise = shotnoise_nonorm / self.wnorm

    def _set_normalization(self, wnorm, mesh1, mesh2=None):
        # Set :attr:`wnorm`
        self.wnorm = wnorm
        if wnorm is None:
            self.wnorm = np.real(normalization(mesh1, mesh2, fields=[('randoms', 'data')]))
