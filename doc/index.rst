.. title:: pypower docs

***********************************
Welcome to pypower's documentation!
***********************************

.. toctree::
  :maxdepth: 1
  :caption: User documentation

  user/building
  api/api

.. toctree::
  :maxdepth: 1
  :caption: Developer documentation

  developer/documentation
  developer/tests
  developer/contributing
  developer/changes

.. toctree::
  :hidden:

************
Introduction
************

**pypower** is a package for auto and cross power spectrum and associated window function estimation,
for periodic boxes, survey geometry, in the flat-sky or plane-parallel (plus first odd wide-angle corrections) configurations.


A typical auto power spectrum estimation is as simple as:

.. code-block:: python

  import numpy as np
  from pypower import CatalogFFTPower

  kedges = np.linspace(0., 0.2, 11)
  # pass mpiroot=0 if input positions and weights are not MPI-scattered
  result = CatalogFFTPower(data_positions1=data_positions1, data_weights1=data_weights1, randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                           edges=kedges, ells=(0, 2, 4), boxsize=1000., nmesh=512, resampler='tsc', interlacing=2, los=None, position_type='pos')
  # wavenumber array in result.poles.k
  # multipoles in result.poles.power

Example notebooks are provided in :root:`pypower/nb`.

**************
Code structure
**************

The code structure is the following:

  - mesh.py implements methods to paint catalog on mesh
  - fft_power.py implements FFT-based power spectrum estimation
  - direct_power.py implements direct estimation of power spectrum multipoles, i.e. summing over particle pairs (typically for PIP corrections)
  - approx_window.py implements computation of approximate window matrix
  - fft_window.py implements computation of more accurate (estimator-based) window matrix
  - wide_angle.py implements wide-angle corrections for window matrix
  - fftlog.py implements FFTlog algorithm, used in window matrix computation
  - utils.py implements various utilities
  - mpi.py implements MPI-related utilities
  - a module for each two-point counter engine


Changelog
=========

* :doc:`developer/changes`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
