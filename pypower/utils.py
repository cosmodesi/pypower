"""A few utilities."""

import os
import sys
import time
import logging
import traceback
import functools

import numpy as np


def exception_handler(exc_type, exc_value, exc_traceback):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '='*100
    #log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')


def mkdir(dirname):
    """Try to create ``dirnm`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname) # MPI...
    except OSError:
        return


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.

    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.

    filename : string, default=None
        If not ``None`` stream to file name.

    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.

    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level,str):
        level = {'info':logging.INFO,'debug':logging.DEBUG,'warning':logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        def format(self, record):
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s'
            return super(MyFormatter,self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename,mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level,handlers=[handler],**kwargs)
    sys.excepthook = exception_handler


class BaseMetaClass(type):

    """Metaclass to add logging attributes to :class:`BaseClass` derived classes."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        cls.set_logger()
        return cls

    def set_logger(cls):
        """
        Add attributes for logging:

        - logger
        - methods log_debug, log_info, log_warning, log_error, log_critical
        """
        cls.logger = logging.getLogger(cls.__name__)

        def make_logger(level):

            @classmethod
            def logger(cls, *args, **kwargs):
                return getattr(cls.logger, level)(*args, **kwargs)

            return logger

        for level in ['debug','info','warning','error','critical']:
            setattr(cls, 'log_{}'.format(level), make_logger(level))


class BaseClass(object,metaclass=BaseMetaClass):
    """
    Base class that implements :meth:`copy`.
    To be used throughout this package.
    """
    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self, **kwargs):
        new = self.__copy__()
        new.__dict__.update(kwargs)
        return new

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    @property
    def with_mpi(self):
        """Whether to use MPI."""
        return getattr(self, 'mpicomm', None) is not None and self.mpicomm.size > 1

    def save(self, filename):
        """Save to ``filename``."""
        if not self.with_mpi or self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            mkdir(os.path.dirname(filename))
            np.save(filename, self.__getstate__(), allow_pickle=True)
        if self.with_mpi:
            self.mpicomm.Barrier()

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new


def distance(positions):
    """Return cartesian distance, taking coordinates along ``position`` first axis."""
    return np.sqrt(sum(pos**2 for pos in positions))


def cartesian_to_sky(positions, wrap=True, degree=True):
    r"""
    Transform cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    positions : array of shape (3, N), list of 3 arrays
        Positions in cartesian coordinates.

    wrap : bool, default=True
        Whether to wrap RA in :math:`[0, 2 \pi]`.

    degree : bool, default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    Returns
    -------
    rdd : list of 3 arrays
        Right ascension, declination and distance.
    """
    dist = distance(positions)
    ra = np.arctan2(positions[1], positions[0])
    if wrap: ra %= 2.*np.pi
    dec = np.arcsin(positions[2]/dist)
    conversion = np.pi/180. if degree else 1.
    return [ra/conversion, dec/conversion, dist]


def sky_to_cartesian(rdd, degree=True, dtype=None):
    """
    Transform distance, RA, Dec into cartesian coordinates.

    Parameters
    ----------
    rdd : array of shape (3, N), list of 3 arrays
        Right ascension, declination and distance.

    degree : default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    Returns
    -------
    positions : list of 3 arrays
        Positions x, y, z in cartesian coordinates.
    """
    conversion = 1.
    if degree: conversion = np.pi/180.
    ra, dec, dist = rdd
    cos_dec = np.cos(dec*conversion)
    x = dist*cos_dec*np.cos(ra*conversion)
    y = dist*cos_dec*np.sin(ra*conversion)
    z = dist*np.sin(dec*conversion)
    return [x, y, z]


def rebin(array, new_shape, statistic=np.sum):
    """
    Bin an array in all axes based on the target shape, by summing or
    averaging. Number of output dimensions must match number of input dimensions and
    new axes must divide old ones.

    Taken from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    and https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/binned_statistic.html#BinnedStatistic.reindex.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = rebin(m, new_shape=(5,5), statistic=np.sum)
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if array.ndim == 1 and np.ndim(new_shape) == 0:
        new_shape = [new_shape]
    if array.ndim != len(new_shape):
        raise ValueError('Input array dim is {}, but requested output one is {}'.format(array.ndim, len(new_shape)))

    pairs = []
    for d, c in zip(new_shape, array.shape):
        if c % d != 0:
            raise ValueError('New shape should divide current shape, but {:d} % {:d} = {:d}'.format(c, d, c % d))
        pairs.append((d, c//d))

    flattened = [l for p in pairs for l in p]
    array = array.reshape(flattened)

    for i in range(len(new_shape)):
        array = statistic(array, axis=-1*(i+1))

    return array


# Create a lookup table for set bits per byte
_popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def popcount(*arrays):
    """
    Return number of 1 bits in each value of input array.
    Inspired from https://github.com/numpy/numpy/issues/16325.
    """
    #if not np.issubdtype(array.dtype, np.unsignedinteger):
    #    raise ValueError('input array must be an unsigned int dtype')
    toret = _popcount_lookuptable[arrays[0].view((np.uint8, (arrays[0].dtype.itemsize,)))].sum(axis=-1)
    for array in arrays[1:]: toret += popcount(array)
    return toret


def pack_bitarrays(*arrays, dtype=np.uint64):
    """
    Pack bit arrays into a list of integer arrays.
    Inverse operation is :func:`unpack_bitarray`, i.e.
    ``unpack_bitarrays(pack_bitarrays(*arrays, dtype=dtype))``is ``arrays``,
    whatever integer ``dtype`` is.

    Parameters
    ----------
    arrays : bool arrays
        Arrays of integers or booleans whose elements should be packed to bits.

    dtype : string, dtype
        Type of output integer arrays.

    Returns
    -------
    arrays : list
        List of integer arrays of type ``dtype``, representing input boolean arrays.
    """
    if not arrays:
        return []
    return reformat_bitarrays(*np.packbits(arrays, axis=0, bitorder='little'), dtype=dtype)


def unpack_bitarrays(*arrays):
    """
    Unpack integer arrays into a bit array.
    Inverse operation is :func:`pack_bitarray`, i.e.
    ``pack_bitarrays(unpack_bitarrays(*arrays), dtype=arrays.dtype)``is ``arrays``.

    Parameters
    ----------
    arrays : integer arrays
        Arrays of integers whose elements should be unpacked to bits.

    Returns
    -------
    arrays : list
        List of boolean arrays of type ``np.uint8``, representing input integer arrays.
    """
    arrayofbytes = reformat_bitarrays(*arrays, dtype=np.uint8)
    return np.unpackbits(arrayofbytes, axis=0, count=None, bitorder='little')


def reformat_bitarrays(*arrays, dtype=np.uint64):
    """
    Reformat input integer arrays into list of arrays of type ``dtype``.
    If, e.g. 6 arrays of type ``np.uint8`` are input, and ``dtype`` is ``np.uint32``,
    a list of 2 arrays is returned.

    Parameters
    ----------
    arrays : integer arrays
        Arrays of integers to reformat.

    dtype : string, dtype
        Type of output integer arrays.

    Returns
    -------
    arrays : list
        List of integer arrays of type ``dtype``, representing input integer arrays.
    """
    dtype = np.dtype(dtype)
    toret = []
    nremainingbytes = 0
    for array in arrays:
        # first bits are in the first byte array
        arrayofbytes = array.view((np.uint8, (array.dtype.itemsize,)))
        arrayofbytes = np.moveaxis(arrayofbytes, -1, 0)
        for arrayofbyte in arrayofbytes:
            if nremainingbytes == 0:
                toret.append([])
                nremainingbytes = dtype.itemsize
            newarray = toret[-1]
            nremainingbytes -= 1
            newarray.append(arrayofbyte[...,None])
    for iarray,array in enumerate(toret):
        npad = dtype.itemsize - len(array)
        if npad: array += [np.zeros_like(array[0])]*npad
        toret[iarray] = np.squeeze(np.concatenate(array,axis=-1).view(dtype), axis=-1)
    return toret
