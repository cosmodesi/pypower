r"""
Implementation of odd wide-angle matrices:
- :class:`CorrelationFunctionOddWideAngleMatrix` for correlation function
- :class:`PowerSpectrumOddWideAngleMatrix` for power spectrum,
following https://arxiv.org/abs/2106.06324.
"""

import logging
from dataclasses import dataclass

import numpy as np

from . import utils
from .utils import BaseClass


@dataclass(frozen=True)
class Projection(BaseClass):
    """
    Class representing a "projection", i.e. multipole and wide-angle expansion order.

    Attributes
    ----------
    ell : int
        Multipole order.

    wa_order : int, None
        Wide-angle order.
    """
    def __init__(self, ell, wa_order='default', default_wa_order=0):
        """
        Initialize :class:`Projection`.

        Parameters
        ----------
        ell : int
            Multipole order.

        wa_order : int, None, default='default'
            Wide-angle order.
            If 'default', defaults to ``default_wa_order``.

        default_wa_order : int, default=0
            Default wide-angle order to use if ``wa_order`` is 'default'.
        """
        if isinstance(ell, self.__class__):
            self.__dict__.update(ell.__dict__)
            return
        if isinstance(ell, (tuple, list)):
            ell, wa_order = ell
        self.__dict__['ell'] = ell
        self.__dict__['wa_order'] = default_wa_order if wa_order == 'default' else wa_order

    def clone(self, **kwargs):
        """Clone current projection, optionally updating ``ell`` and ``wa_order`` (using ``kwargs``)."""
        return self.__class__(**{**self.__getstate__(), **kwargs})

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for name in ['ell', 'wa_order']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __repr__(self):
        """String representation of current projection."""
        return '{}(ell={}, wa_order={})'.format(self.__class__.__name__, self.ell, self.wa_order)

    def latex(self, inline=False):
        """
        Return latex string for current projection.
        If ``inline`` is ``True``, add surrounding dollar $ signs.
        """
        if self.wa_order is None:
            toret = r'\ell = {:d}'.format(self.ell)
        else:
            toret = r'(\ell, n) = ({:d}, {:d})'.format(self.ell, self.wa_order)
        if inline:
            toret = '${}$'.format(toret)
        return toret

    def __hash__(self):
        return hash((self.ell, self.wa_order))

    def __eq__(self, other):
        """Is current projection equal to ``other``?"""
        try:
            return self.ell == other.ell and self.wa_order == other.wa_order
        except AttributeError:
            return False

    def __gt__(self, other):
        """Is current projection greater than ``other``?"""
        return (self.ell > other.ell) or (self.wa_order is not None and other.wa_order is None) or (other.wa_order is not None and self.wa_order > other.wa_order)

    def __lt__(self, other):
        """Is current projection less than ``other``?"""
        return (self.ell < other.ell) or (other.wa_order is not None and self.wa_order is None) or (self.wa_order is not None and self.wa_order < other.wa_order)


class BaseMatrix(BaseClass):
    """
    Base class to represent a linear transform of the theory model,
    from input projections :attr:`projsin` to output projections :attr:`projsout`.

    Attributes
    ----------
    matrix : array
        2D array representing linear transform.
        First axis is input, second is output.

    xin : list
        List of input "theory" coordinates.

    xout : list
        List of output "theory" coordinates.

    projsin : list
        List of input "theory" projections.

    projsout : list
        List of output "observed" projections.
    """
    weightsin = None
    weightsout = None

    def __init__(self, value, xin, xout, projsin, projsout, weightsin=None, weightsout=None, attrs=None):
        """
        Initialize :class:`BaseMatrix`.

        Parameters
        ----------
        value : array
            2D array representing linear transform.

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

        weightsin : array, list, default=None
            Optionally, list of weights to apply when rebinning input "theory" coordinates.

        weightsout : array, list, default=None
            Optionally, list of weights to apply when rebinning output "observed" coordinates.

        attrs : dict, default=None
            Dictionary of other attributes.
        """
        self.value = np.asarray(value)
        if self.value.ndim != 2:
            raise ValueError('Input matrix must be 2D, not {}D.'.format(self.value.ndim))
        self.projsin = [Projection(proj) for proj in projsin]
        self.projsout = [Projection(proj) for proj in projsout]
        self._set_xw(shape=self.value.shape, xin=xin, xout=xout, weightsin=weightsin, weightsout=weightsout)
        self.attrs = attrs or {}

    def _set_xw(self, shape=None, **kwargs):
        for iaxis, axis in enumerate(['in', 'out']):
            projsname = 'projs{}'.format(axis)
            projs = getattr(self, projsname)
            for name in ['x', 'weights']:
                name = '{}{}'.format(name, axis)
                arrays = kwargs.get(name, None)
                if arrays is not None:
                    if np.ndim(arrays[0]) == 0: arrays = [np.asarray(arrays) for proj in projs]
                    else: arrays = [np.asarray(array) for array in arrays]
                    setattr(self, name, arrays)
                    if len(arrays) != len(projs):
                        raise ValueError('Input {} should be a list of arrays of same length as {}'.format(name, projsname))
                    size = sum(len(array) for array in arrays)
                    if shape is not None and size != shape[iaxis]:
                        raise ValueError('Given input {} and {}, input matrix should be of size {:d} along axis {:d}'.format(name, projsname, size, iaxis))

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in ['value', 'xin', 'xout', 'weightsin', 'weightsout', 'attrs']:
            if hasattr(self, key): state[key] = getattr(self, key)
        for key in ['projsin', 'projsout']:
            state[key] = [proj.__getstate__() for proj in getattr(self, key)]
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(BaseMatrix, self).__setstate__(state)
        for key in ['projsin', 'projsout']:
            setattr(self, key, [Projection.from_state(state) for state in getattr(self, key)])

    def dot(self, array, unpack=False):
        """
        Apply linear transform to input array.
        If ``unpack`` is ``True``, return "unpacked" array,
        i.e. a list of arrays corresponding to ``projsout``.
        """
        array = np.dot(np.asarray(array).flat, self.value)
        if unpack:
            toret = []
            nout = 0
            for xout in self.xout:
                sl = slice(nout, nout+len(xout))
                toret.append(array[sl])
                nout = sl.stop
            return toret
        return array

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return len(self.shape)

    def pack(self, matrix):
        """
        Set :attr:`matrix` from "unpacked" matrix, i.e. from a list of lists of matrices,
        where block for output projection ``projout`` and input projection ``projin``
        is obtained through ``matrix[self.projsout.index(projout)][self.projsin.index(projin)]``.
        See :meth:`unpacked`.
        """
        self.value = np.bmat(matrix).A

    def unpacked(self, axis=None):
        """
        Return unpacked matrix, a list of lists of matrices
        where block for output projection ``projout`` and input projection ``projin``
        is obtained through ``matrix[self.projsout.index(projout)][self.projsin.index(projin)]``.
        """
        matrix = []
        nin = 0
        for xin in self.xin:
            slin = slice(nin, nin + len(xin))
            line = []
            nout = 0
            for xout in self.xout:
                slout = slice(nout, nout+len(xout))
                line.append(self.value[slin, slout])
                nout = slout.stop
            nin = slin.stop
            matrix.append(line)
        if axis == 'in':
            matrix = [np.concatenate(m, axis=-1) for m in matrix]
        if axis == 'out':
            matrix = [np.concatenate([matrix[iin][iout] for iin in range(len(self.xin))], axis=0) for iout in range(len(self.xout))]
        return matrix

    def select_proj(self, projsin=None, projsout=None, **kwargs):
        """
        Restrict current instance to provided projections.

        Parameters
        ----------
        projsin : list, default=None
            List of input projections to restrict to.
            Defaults to :attr:`projsin`.
            If one projection is not in :attr:`projsin`, add a new column to :attr:`matrix`,
            setting a diagonal matrix where input and output projection match (if the case);
            see ``xin``.

        projsout : list, default=None
            List of output projections to restrict to.
            Defaults to :attr:`projsout`.
            If one projection is not in :attr:`projsout`, add a new row to :attr:`matrix`,
            setting a diagonal matrix where input and output projection match (if the case);
            see ``xout``.

        kwargs : dict
            In case a new input/output projection must be added,
            :attr:`xin`/:attr:`xout` for this projection.
        """
        old_matrix = self.unpacked()
        old_projs = {}

        for axis in ['in', 'out']:
            name = 'projs{}'.format(axis)
            old_projs[axis] = getattr(self, name)
            projs = locals()[name]
            if projs is None:
                projs = old_projs[axis]
            else:
                if not isinstance(projs, list): projs = [projs]
                projs = [Projection(proj) for proj in projs]
            setattr(self, name, projs)

            for name in ['x', 'weights']:
                name = '{}{}'.format(name, axis)
                old = getattr(self, name, None)
                if old is not None:
                    new = []
                    for proj in projs:
                        if proj in old_projs[axis]:
                            new.append(old[old_projs[axis].index(proj)])
                        else:
                            new.append(kwargs.get(name, old[0]))
                    setattr(self, name, new)

        self.value = []
        for iin, projin in enumerate(self.projsin):
            line = []
            for iout, projout in enumerate(self.projsout):
                if projin in old_projs['in'] and projout in old_projs['out']:
                    tmp = old_matrix[old_projs['in'].index(projin)][old_projs['out'].index(projout)]
                else:
                    shape = (len(self.xin[iin]), len(self.xout[iout]))
                    if projout == projin:
                        if shape[1] != shape[0]:
                            raise ValueError('Cannot set diagonal matrix for ({}, {}) as expected shape is {}'.format(projin, projout, shape))
                        tmp = np.eye(shape[0], dtype=self.dtype)
                    else:
                        tmp = np.zeros(shape, dtype=self.dtype)
                line.append(tmp)
            self.value.append(line)
        self.value = np.bmat(self.value).A

    def __getitem__(self, slices):
        """Call :meth:`slice_x`."""
        new = self.copy()
        if isinstance(slices, tuple):
            new.slice_x(*slices)
        else:
            new.slice_x(slices)
        return new

    def slice_x(self, slicein=None, sliceout=None, projsin=None, projsout=None):
        """
        Slice matrix in place. If slice step is not 1, use :meth:`rebin`.

        Parameters
        ----------
        slicein : slice, default=None
            Slicing to apply to input coordinates, defaults to ``slice(None)``.

        sliceout : slice, default=None
            Slicing to apply to output coordinates, defaults to ``slice(None)``.

        projsin : list, default=None
            List of input projections to apply slicing to.
            Defaults to :attr:`projsin`.

        projsout : list, default=None
            List of output projections to apply slicing to.
            Defaults to :attr:`projsout`.
        """
        self.value = self.unpacked() # unpack first, as based on :attr:`xin`, :attr:`xout`

        inprojs, masks, factors = {}, {}, {}
        for axis in ['in', 'out']:
            name = 'projs{}'.format(axis)
            projs = locals()[name]
            selfprojs = getattr(self, name)
            if projs is None:
                projs = selfprojs
            else:
                if not isinstance(projs, list): projs = [projs]
                projs = [Projection(proj) for proj in projs]
            inprojs[axis] = projs

            masks[axis] = []
            x = getattr(self, 'x{}'.format(axis))
            sl = locals()['slice{}'.format(axis)]
            if sl is None: sl = slice(None)
            start, stop, step = sl.start, sl.stop, sl.step
            if start is None: start = 0
            if step is None: step = 1
            factors[axis] = step

            for proj in projs:
                selfii = selfprojs.index(proj)
                indices = np.arange(len(x[selfii]))[slice(start, stop, 1)]
                if indices.size:
                    stopii = indices[-1] + 1 # somewhat hacky, but correct!
                else:
                    stopii = 0
                masks[axis].append(np.arange(start, stopii, 1))

            for name in ['x', 'weights']:
                arrays = getattr(self, '{}{}'.format(name, axis))
                if arrays is not None:
                    for ii, proj in enumerate(projs):
                        selfii = selfprojs.index(proj)
                        arrays[selfii] = arrays[selfii][masks[axis][ii]]

        for iin, projin in enumerate(inprojs['in']):
            selfiin = self.projsin.index(projin)
            for iout, projout in enumerate(inprojs['out']):
                selfiout = self.projsout.index(projout)
                self.value[selfiin][selfiout] = self.value[selfiin][selfiout][np.ix_(masks['in'][iin], masks['out'][iout])]

        self.value = np.bmat(self.value).A
        if not all(f == 1 for f in factors.values()):
            self.rebin_x(factorin=factors['in'], factorout=factors['out'], projsin=inprojs['in'], projsout=inprojs['out'])

    def select_x(self, xinlim=None, xoutlim=None, projsin=None, projsout=None):
        """
        Restrict current instance to provided coordinate limits in place.

        Parameters
        ----------
        xinlim : tuple, default=None
            Restrict input coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        xoutlim : tuple, default=None
            Restrict output coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        projsin : list, default=None
            List of input projections to apply limits to.
            Defaults to :attr:`projsin`.

        projsout : list, default=None
            List of output projections to apply limits to.
            Defaults to :attr:`projsout`.
        """
        # One could also set the slices, and call slice_x, but this is inefficient in case x is different for each projection
        self.value = self.unpacked() # unpack first, as based on :attr:`xin`, :attr:`xout`

        inprojs, masks = {}, {}
        for axis in ['in', 'out']:
            name = 'projs{}'.format(axis)
            projs = locals()[name]
            selfprojs = getattr(self, name)
            if projs is None:
                projs = selfprojs
            else:
                if not isinstance(projs, list): projs = [projs]
                projs = [Projection(proj) for proj in projs]
            inprojs[axis] = projs

            masks[axis] = []
            x = getattr(self, 'x{}'.format(axis))
            lim = locals()['x{}lim'.format(axis)]
            if lim is None: lim = (-np.inf, np.inf)
            for proj in projs:
                selfii = selfprojs.index(proj)
                tmp = (x[selfii] >= lim[0]) & (x[selfii] <= lim[1])
                masks[axis].append(np.all(tmp, axis=tuple(range(1, tmp.ndim))))

            for name in ['x', 'weights']:
                arrays = getattr(self, '{}{}'.format(name, axis))
                if arrays is not None:
                    for ii, proj in enumerate(projs):
                        selfii = selfprojs.index(proj)
                        arrays[selfii] = arrays[selfii][masks[axis][ii]]

        for iin, projin in enumerate(inprojs['in']):
            selfiin = self.projsin.index(projin)
            for iout, projout in enumerate(inprojs['out']):
                selfiout = self.projsout.index(projout)
                self.value[selfiin][selfiout] = self.value[selfiin][selfiout][np.ix_(masks['in'][iin], masks['out'][iout])]

        self.value = np.bmat(self.value).A

    def rebin_x(self, factorin=1, factorout=1, projsin=None, projsout=None, statistic=None):
        """
        Rebin current instance.
        Internal weights :attr:`weightsin`, :attr:`weightsout`, if not ``None``, are applied.

        Parameters
        ----------
        factorin : int, default=1
            Rebin matrix along input coordinates by this factor.

        factorout : int, default=1
            Rebin matrix along output coordinates by this factor.

        projsin : list, default=None
            List of input projections to apply rebinning to.
            Defaults to :attr:`projsin`.

        projsout : list, default=None
            List of output projections to apply rebinning to.
            Defaults to :attr:`projsout`.

        statistic : string, callable, default=None
            Operation to apply when performing rebinning.
            Defaults to average along input coordinates and sum along output coordinates.
        """
        if statistic is None:
            self.rebin_x(factorin=1, factorout=factorout, projsin=projsin, projsout=projsout, statistic=np.mean)
            self.rebin_x(factorin=factorin, factorout=1, projsin=projsin, projsout=projsout, statistic=np.sum)
            return

        self.value = self.unpacked() # unpack first, as based on :attr:`xin`, :attr:`xout`

        inprojs, old_weights, new_weights = {}, {}, {}
        for axis in ['in', 'out']:
            name = 'projs{}'.format(axis)
            projs = locals()[name]
            selfprojs = getattr(self, name)
            if projs is None:
                projs = selfprojs
            else:
                if not isinstance(projs, list): projs = [projs]
                projs = [Projection(proj) for proj in projs]
            inprojs[axis] = projs
            factorname = 'factor{}'.format(axis)
            factor = locals()[factorname]
            arrays = getattr(self, 'weights{}'.format(axis))
            if arrays is not None:
                old_weights[axis], new_weights[axis] = [], []
                for proj in projs:
                    selfii = selfprojs.index(proj)
                    old_weights[axis].append(arrays[selfii])
                    if len(arrays[selfii]) % factor:
                        raise ValueError('Rebinning factor {} must divide size along axis {}'.format(factorname, axis))
                    arrays[selfii] = utils.rebin(arrays[selfii], len(arrays[selfii])//factor, statistic=np.sum)
                    new_weights[axis].append(arrays[selfii])
            arrays = getattr(self, 'x{}'.format(axis))
            for ii, proj in enumerate(projs):
                selfii = selfprojs.index(proj)
                if old_weights:
                    arrays[selfii] = utils.rebin(arrays[selfii]*old_weights[axis][ii], len(arrays[selfii])//factor, statistic=np.sum)/new_weights[axis][ii]
                else:
                    arrays[selfii] = utils.rebin(arrays[selfii], len(arrays[selfii])//factor, statistic=np.mean)

        for iin, projin in enumerate(inprojs['in']):
            selfiin = self.projsin.index(projin)
            for iout, projout in enumerate(inprojs['out']):
                selfiout = self.projsout.index(projout)
                tmp = self.value[selfiin][selfiout]
                new_shape = tuple(s//f for s,f in zip(tmp.shape, (factorin, factorout)))
                oweights, nweights = 1., 1.
                for iaxis, (axis, ii) in enumerate(zip(['in', 'out'], [iin, iout])):
                    if axis in old_weights:
                        oweights = oweights * np.expand_dims(old_weights[axis][ii], axis=1-iaxis)
                        nweights = nweights * np.expand_dims(new_weights[axis][ii], axis=1-iaxis)
                self.value[selfiin][selfiout] = utils.rebin(self.value[selfiin][selfiout]*oweights, new_shape, statistic=statistic)/nweights
        self.value = np.bmat(self.value).A

    @classmethod
    def concatenate_proj(cls, *others, axis='in'):
        """
        Concatenate input matrices along projection axis ``axis``.

        Parameters
        ----------
        others : BaseMatrix
            Matrices to concatenate.

        axis : string, default='in'
            Should be either 'in' (to stack input projections)
            or 'out' (to stack output projections).

        Returns
        -------
        matrix : BaseMatrix
            New matrix, of same type as ``others[0]``.
        """
        new = others[0].copy()
        axis = axis.lower()
        iaxis = ['in', 'out'].index(axis)
        for name in ['projs', 'x', 'weights']:
            name = '{}{}'.format(name, axis)
            if getattr(others[0], name) is not None:
                arrays = []
                for other in others: arrays += getattr(other, name)
                setattr(new, name, arrays)
        new.value = np.concatenate([other.value for other in others], axis=iaxis)
        return new

    @classmethod
    def concatenate_x(cls, *others, axis='in'):
        """
        Concatenate input matrices along x-axis ``axis``.

        Parameters
        ----------
        others : BaseMatrix
            Matrices to concatenate.

        axis : string, default='in'
            Should be either 'in' (to stack input x)
            or 'out' (to stack output x).

        Returns
        -------
        matrix : BaseMatrix
            New matrix, of same type as ``others[0]``.
        """
        new = others[0].copy()
        axis = axis.lower()
        iaxis = ['in', 'out'].index(axis)
        for name in ['x', 'weights']:
            name = '{}{}'.format(name, axis)
            if getattr(new, name) is not None:
                arrays = []
                for ii in range(len(getattr(new, name))):
                    arrays.append(np.concatenate([getattr(other, name)[ii] for other in others], axis=0))
                setattr(new, name, arrays)
        new.value = []
        others = [other.unpacked() for other in others]
        for iin, projin in enumerate(new.projsin):
            line = []
            for iout, projout in enumerate(new.projsout):
                line.append(np.concatenate([m[iin][iout] for m in others], axis=iaxis))
            new.value.append(line)
        new.value = np.bmat(new.value).A
        return new

    @staticmethod
    def join(*others):
        """
        Join input matrices, i.e. dot them,
        optionally selecting input and output projections such that they match.
        """
        new = BaseMatrix.copy(others[-1])
        for first, second in zip(others[-2::-1], others[::-1]):
            first = first.copy()
            first.select_proj(projsout=second.projsin)
            if first.shape[1] != second.shape[0]:
                raise ValueError('Input matrices do not have same shape')
            new.value = first.value @ new.value
            for name in ['projs', 'x', 'weights']:
                name = '{}in'.format(name)
                tmp = getattr(first, name)
                if tmp is not None: tmp = tmp.copy()
                setattr(new, name, tmp)
        return new

    def __copy__(self):
        new = super(BaseMatrix, self).__copy__()
        new.attrs = self.attrs.copy()
        for axis in ['in', 'out']:
            for name in ['projs', 'x', 'weights']:
                name = '{}{}'.format(name, axis)
                tmp = getattr(new, name)
                if tmp is not None: tmp = tmp.copy()
                setattr(new, name, tmp)
        return new

    @property
    def nx(self):
        """Tuple of list of length of input and output coordinates."""
        return ([len(x) for x in self.xin], [len(x) for x in self.xout])

    @property
    def nprojs(self):
        """Number of input, output projections."""
        return (len(self.projsin), len(self.projsout))

    def prod_proj(self, array, axes=('in', 0), projs=None):
        """
        Multiply current matrix by input ``array`` along input ``axes``, projection-wise,
        i.e. a same operation is applied for all coordinates of a given (input projection, output projection) block.

        Parameters
        ----------
        array : 1D or 2D array
            Array to multiply matrix with.

        axes : string, tuple
            Tuple of axes to sum over (axis in current matrix ("in" or "out")), axis in input ``array``).
            If ``array`` is 1D, one can just provide the axis in current matrix ("in" or "out").
        """
        array = np.asarray(array)
        if array.ndim == 1:
            if np.ndim(axes) == 0: axes = (axes, 0)
            array = np.diag(array)
        elif array.ndim != 2:
            raise ValueError('Input array should be 1D or 2D')
        axes = tuple(axes)
        if len(axes) != 2:
            raise ValueError('Please provide a tuple for axes to sum over: (axis in self - in or out, axis in input array)')
        unpacked = self.unpacked(axis=axes[0])
        projsname = 'projs{}'.format(axes[0])
        if projs is None: projs = getattr(self, projsname)
        else: projs = [Projection(proj) for proj in projs]
        iaxis = ['in', 'out'].index(axes[0])
        if not all(nx == self.nx[iaxis][0] for nx in self.nx[iaxis]):
            raise ValueError('Coordinates do not have same length along input axis {}'.format(axes[0]))
        reverse = axes[1] % 2 == 0 # we want to sum over second axis of array
        if reverse: array = array.T
        shape = (len(projs), len(unpacked))
        if array.shape != shape:
            raise ValueError('Given input projs, input array is expected to be a matrix of shape {}'.format(shape[::-1] if reverse else shape))
        matrix = []
        for iout in range(shape[0]):
            tmp = sum(c * unpacked[iin] for iin, c in enumerate(array[iout]))
            matrix.append(tmp)
        self.value = np.concatenate(matrix, axis=iaxis)
        setattr(self, projsname, projs)
        for name in ['x', 'weights']:
            name = '{}{}'.format(name, axes[0])
            tmp = getattr(self, name)
            if tmp is not None and len(tmp) != len(projs):
                tmp = [tmp[0].copy() for _ in projs]
            setattr(self, name, tmp)


def odd_wide_angle_coefficients(ell, wa_order=1, los='firstpoint'):
    r"""
    Compute coefficients of odd wide-angle expansion, i.e.:

    .. math::

        - \frac{\ell \left(\ell - 1\right)}{2 \ell \left(2 \ell - 1\right)}, \frac{\left(\ell + 1\right) \left(\ell + 2\right)}{2 \ell \left(2 \ell + 3\right)}

    For the first point line-of-sight. See https://fr.overleaf.com/read/hpgbwqzmtcxn.
    A minus sign is applied on both factors if ``los`` is 'endpoint'.

    Parameters
    ----------
    ell : int
        (Odd) multipole order.

    wa_order : int, default=1
        Wide-angle expansion order.
        So far only order 1 is supported.

    los : string
        Choice of line-of-sight, either:

        - 'firstpoint': the separation vector starts at the end of the line-of-sight
        - 'endpoint': the separation vector ends at the end of the line-of-sight.

    Returns
    -------
    ells : list
        List of multipole orders of correlation function.

    coeffs : list
        List of coefficients to apply to correlation function multipoles corresponding to output ``ells``.
    """
    if wa_order != 1:
        raise ValueError('Only wide-angle order 1 supported')

    if ell % 2 == 0:
        raise ValueError('Wide-angle order 1 produces only odd poles')

    if los not in ['firstpoint', 'endpoint']:
        raise ValueError('Only "firstpoint" and "endpoint" line-of-sight supported')

    def coefficient(ell):
        return ell*(ell+1)/2./(2*ell+1)

    sign = (-1)**(los == 'endpoint')
    if ell == 1:
        return [ell + 1], [sign * coefficient(ell+1)]
    return [ell-1, ell+1], [- sign * coefficient(ell-1), sign * coefficient(ell+1)]


class CorrelationFunctionOddWideAngleMatrix(BaseMatrix):

    """Class computing matrix for odd wide-angle expansion of the correlation function."""

    def __init__(self, sep, projsin, projsout=None, wa_orders=1, los='firstpoint', attrs=None):
        """
        Initialize :class:`CorrelationFunctionOddWideAngleMatrix`.

        Parameters
        ----------
        k : array
            Input (and ouput) separations.

        projsin : list
            Input projections.

        projsout : list, default=None
            Output projections. Defaults to ``propose_out(projsin, wa_orders=wa_orders)``.
            If output projections have :attr:`Projection.wa_order` ``None``, wide-angle orders are summed over.

        wa_orders : int, list
            Wide-angle expansion orders.
            So far order 1 only is supported.

        los : string
            Choice of line-of-sight, either:

            - 'firstpoint': the separation vector starts at the end of the line-of-sight
            - 'endpoint': the separation vector ends at the end of the line-of-sight.

        attrs : dict, default=None
            Dictionary of other attributes.
        """
        self.wa_orders = wa_orders
        if np.ndim(wa_orders) == 0:
            self.wa_orders = [wa_orders]
        if not np.allclose(self.wa_orders, [1]):
            raise NotImplementedError('Only wide-angle order wa_order = 1 supported')
        self.los = los.lower()

        self.projsin = [Projection(proj) for proj in projsin]
        if projsout is None:
            self.projsout = self.propose_out(projsin, wa_orders=self.wa_orders)
        else:
            self.projsout = [Projection(proj) for proj in projsout]
        if any(proj.wa_order is None for proj in self.projsin):
            raise ValueError('Input projections must have wide-angle order wa_order specified')
        self._set_xw(xin=sep, xout=sep)
        self.attrs = attrs or {}
        self.run()

    def run(self):
        r"""
        Set matrix:

        .. math::

            M_{\ell\ell^{\prime}}^{(n,n^{\prime})}(s) =
            - \frac{\ell \left(\ell - 1\right)}{2 \ell \left(2 \ell - 1\right)} \delta_{\ell,\ell - 1} \delta_{n^{\prime},0}
            + \frac{\left(\ell + 1\right) \left(\ell + 2\right)}{2 \ell \left(2 \ell + 3\right)} \delta_{\ell,\ell + 1} \delta_{n^{\prime},0}

        if :math:`\ell` is odd and :math:`n = 1`, else:

        .. math::

            M_{\ell\ell^{\prime}}^{(0,n^{\prime})}(s) = \delta_{\ell,ell^{\prime}} \delta_{n^{\prime},0}

        with :math:`\ell` multipole order corresponding to ``projout.ell`` and :math:`\ell^{\prime}` to ``projin.ell``,
        :math:`n` wide angle order corresponding to ``projout.wa_order`` and :math:`n^{\prime}` to ``projin.wa_order``.
        If output ``projout.wa_order`` is ``None``, sum over :math:`n` (correct only if no window convolution is accounted for).
        """
        sep = self.xin[0]
        eye = np.ones(len(sep), dtype=sep.dtype)
        self.projvalue = []
        for projin in self.projsin:
            line = []
            for iprojout, projout in enumerate(self.projsout):
                block = 0.*eye
                if projout.ell == projin.ell and projout.wa_order == projin.wa_order:
                    block = eye
                else:
                    if projout.wa_order is None:
                        wa_orders = self.wa_orders # sum over :math:`n`
                    else:
                        wa_orders = [projout.wa_order] # projout.wa_order is 1
                    for wa_order in wa_orders:
                        if wa_order != 1: continue
                        ells, coeffs = odd_wide_angle_coefficients(projout.ell, wa_order=wa_order, los=self.los)
                        if projin.wa_order == 0 and projin.ell in ells:
                            coeff = coeffs[ells.index(projin.ell)]
                            block += coeff * eye
                line.append(block)
            self.projvalue.append(line)
        self.projvalue = np.array(self.projvalue) # (in, out)

    @staticmethod
    def propose_out(projsin, wa_orders=1):
        """Propose output projections (i.e. multipoles at wide-angle order > 0) that can be computed given proposed input projections ``projsin``."""
        if np.ndim(wa_orders) == 0:
            wa_orders = [wa_orders]

        projsin = [Projection(proj) for proj in projsin]
        ellsin = [proj.ell for proj in projsin if proj.wa_order == 0] # only consider input wa_order = 0 multipoles
        projsout = []
        for wa_order in wa_orders:
            for ellout in range(1, max(ellsin) + 2, 2):
                if any(ell in ellsin for ell in odd_wide_angle_coefficients(ellout, wa_order=wa_order)[0]): # check input multipoles are provided
                    projsout.append(Projection(ell=ellout, wa_order=wa_order))

        return projsout

    @property
    def value(self):
        if getattr(self, '_value', None) is None:
            self._value = np.bmat([[np.diag(tmp) for tmp in line] for line in self.projvalue]).A
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class PowerSpectrumOddWideAngleMatrix(BaseMatrix):
    """
    Class computing matrix for odd wide-angle expansion of the power spectrum.
    Adapted from https://github.com/fbeutler/pk_tools/blob/master/wide_angle_tools.py
    """
    def __init__(self, k, projsin, projsout=None, d=1., wa_orders=1, los='firstpoint', attrs=None):
        """
        Initialize :class:`PowerSpectrumOddWideAngleMatrix`.

        Parameters
        ----------
        k : array
            Input (and ouput) wavenumbers.

        projsin : list
            Input projections.

        projsout : list, default=None
            Output projections. Defaults to ``propose_out(projsin, wa_orders=wa_orders)``.
            If output projections have :attr:`Projection.wa_order` ``None``, wide-angle orders are summed over.

        d : float, default=1
            Distance at the effective redshift. Use :math:`1` if already included in window functions.

        wa_orders : int, list
            Wide-angle expansion orders.
            So far order 1 only is supported.

        los : string
            Choice of line-of-sight, either:

            - 'firstpoint': the separation vector starts at the end of the line-of-sight
            - 'endpoint': the separation vector ends at the end of the line-of-sight.

        attrs : dict, default=None
            Dictionary of other attributes.
        """
        self.d = d
        CorrelationFunctionOddWideAngleMatrix.__init__(self, k, projsin, projsout=projsout, wa_orders=wa_orders, los=los, attrs=attrs)

    def run(self):
        r"""
        Set matrix:

        .. math::

            M_{\ell\ell^{\prime}}^{(n,n^{\prime})}(k) =
            - \frac{\ell \left(\ell - 1\right)}{2 \ell \left(2 \ell - 1\right) d} \delta_{\ell,\ell - 1} \delta_{n^{\prime},0} \left[\frac{\ell - 1}{k} - \partial_{k} \right]
            - \frac{\left(\ell + 1\right) \left(\ell + 2\right)}{2 \ell \left(2 \ell + 3\right) d} \delta_{\ell,\ell + 1} \delta_{n^{\prime},0} \left[ \frac{\ell + 2}{k} + \partial_{k} \right]

        if :math:`\ell` is odd and :math:`n = 1`, else:

        .. math::

            M_{\ell\ell^{\prime}}^{(0,n^{\prime})}(k) = \delta_{\ell,ell^{\prime}} \delta_{n^{\prime},0}

        with :math:`\ell` multipole order corresponding to ``projout.ell`` and :math:`\ell^{\prime}` to ``projin.ell``,
        :math:`n` wide angle order corresponding to ``projout.wa_order`` and :math:`n^{\prime}` to ``projin.wa_order``.
        If output ``projout.wa_order`` is ``None``, sum over :math:`n` (correct only if no window convolution is accounted for).
        Derivatives :math:`\partial_{k}` are computed with finite differences, see arXiv:2106.06324 eq. 3.3.
        """
        k = self.xin[0]
        eye = np.eye(len(k), dtype=k.dtype)
        self.value = []

        for projin in self.projsin:
            line = []
            for iprojout, projout in enumerate(self.projsout):
                block = 0.*eye
                if projout.ell == projin.ell and projout.wa_order == projin.wa_order:
                    block = eye
                else:
                    if projout.wa_order is None:
                        wa_orders = self.wa_orders # sum over :math:`n`
                    else:
                        wa_orders = [projout.wa_order] # projout.wa_order is 1
                    for wa_order in wa_orders:
                        if wa_order != 1: continue
                        ells, coeffs = odd_wide_angle_coefficients(projout.ell, wa_order=wa_order, los=self.los)
                        if projin.wa_order == 0 and projin.ell in ells:
                            # - \frac{\ell \left(\ell - 1\right)}{2 \ell \left(2 \ell - 1\right) d} (if projin.ell == projout.ell - 1)
                            # or \frac{\left(\ell + 1\right) \left(\ell + 2\right)}{2 \ell \left(2 \ell + 3\right) d} (if projin.ell == projout.ell + 1)
                            coeff = coeffs[ells.index(projin.ell)]/self.d
                            if projin.ell == projout.ell + 1:
                                coeff_spherical_bessel = - (projin.ell + 1)
                            else:
                                coeff_spherical_bessel = projin.ell
                            # K 'diag' terms
                            # tmp is - \frac{\ell \left(\ell - 1\right)}{2 \ell \left(2 \ell - 1\right) d} \frac{\ell - 1}{k} (if projin.ell == projout.ell - 1)
                            # or - \frac{\left(\ell + 1\right) \left(\ell + 2\right)}{2 \ell \left(2 \ell + 3\right) d} \frac{\ell + 2}{k} (if projin.ell == projout.ell + 1)
                            tmp = np.diag(coeff_spherical_bessel * coeff / k)
                            deltak = 2. * np.diff(k)
                            # derivative - :math:`\partial_{k}`
                            tmp += np.diag(coeff / deltak, k=-1) - np.diag(coeff / deltak, k=1)

                            # taking care of corners
                            tmp[0,0] += 2.*coeff / deltak[0]
                            tmp[0,1] = -2.*coeff / deltak[0]
                            tmp[-1,-1] -= 2.*coeff / deltak[-1]
                            tmp[-1,-2] = 2.*coeff / deltak[-1]
                            block += tmp.T # (in, out)
                line.append(block)
            self.value.append(line)
        self.value = np.bmat(self.value).A # (in, out)

    propose_out = CorrelationFunctionOddWideAngleMatrix.propose_out
