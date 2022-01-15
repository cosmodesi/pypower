import os

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo import Cosmology
from mockfactory import Catalog

from pypower import Projection, BaseMatrix, PowerSpectrumOddWideAngleMatrix, setup_logging


def test_deriv():
    n = 5
    m = np.zeros((n,n), dtype='f8')
    m += np.diag(np.ones(n-1), k=1) - np.diag(np.ones(n-1), k=-1)
    m[0,0] = -0.5
    m[0,1] = 0.5
    m[-1,-1] = 0.5
    m[-1,-2] = -0.5

    ref = np.zeros((n,n), dtype='f8')
    for index1 in range(n):
        index2 = index1

        if index2 > 0 and index2 < n-1:
            pre_factor = 1.
        else:
            pre_factor = 0.5

        if index2 > 0:
            ref[index1, index2-1] = -pre_factor
        else:
            ref[index1, index2] += -pre_factor

        if index2 < n-1:
            ref[index1, index2+1] = pre_factor
        else:
            ref[index1, index2] += pre_factor

    assert np.allclose(m, ref)


def test_projection():

    tu = (2, 1)
    proj = Projection(tu)
    assert proj.ell == tu[0] and proj.wa_order == tu[1]
    proj = Projection(*tu)
    assert proj.ell == tu[0] and proj.wa_order == tu[1]
    proj = Projection(ell=tu[0], wa_order=tu[1])
    assert proj.ell == tu[0] and proj.wa_order == tu[1]
    assert proj.latex() == '(\\ell, n) = (2, 1)'
    assert proj.latex(inline=True) == '$(\\ell, n) = (2, 1)$'


def test_matrix():

    projsin = [Projection(0, 0), Projection(2, 1), Projection(4, 0)]
    projsout = [Projection(2, 0), Projection(1, 1), Projection(2, 0)]
    xin = xout = np.arange(20)
    matrix = np.eye(len(projsin) * len(xin))
    matrix = BaseMatrix(matrix, xin, xout, projsin, projsout, weightsin=np.linspace(0.2, 1., len(xin)), weightsout=np.linspace(0.2, 1., len(xout)))
    matrix.select_projs(projsout=projsout[:-1])
    assert matrix.projsout == projsout[:-1]
    assert len(matrix.xout) == len(matrix.projsout)
    assert matrix.shape == (len(projsin) * len(xin), len(matrix.projsout) * len(xin))
    matrix.select_x(xinlim=(2, np.inf))
    assert len(matrix.xin[0]) == len(xin) - 2
    assert matrix.shape == (len(matrix.projsin) * len(matrix.xin[0]), len(matrix.projsout) * len(matrix.xout[0]))
    matrix.rebin_x(factorout=2)
    assert len(matrix.xout[0]) == len(xout)//2
    assert matrix.shape == (len(matrix.projsin) * len(matrix.xin[0]), len(matrix.projsout) * len(matrix.xout[0]))

    projsin = [Projection(5, 1)]
    matrix2 = BaseMatrix(matrix.value[:len(matrix.xin[0])*len(projsin),:], matrix.xin[0], matrix.xout, projsin, matrix.projsout, weightsin=matrix.weightsin[0], weightsout=matrix.weightsout)
    matrix = matrix.concatenate_proj(matrix, matrix2, axis='in')
    assert matrix.projsin[-1] == projsin[0]
    assert matrix.shape == (len(matrix.projsin) * len(matrix.xin[0]), len(matrix.projsout) * len(matrix.xout[0]))
    matrix2 = BaseMatrix(matrix.value[:,:2*len(matrix.projsout)], matrix.xin[0], matrix.xout[0][:2], matrix.projsin, matrix.projsout, weightsin=matrix.weightsin[0], weightsout=matrix.weightsout[0][:2])
    matrix = matrix.concatenate_x(matrix, matrix2, axis='out')
    assert len(matrix.xout[0]) == len(xout)//2 + 2
    assert matrix.shape == (len(matrix.projsin) * len(matrix.xin[0]), len(matrix.projsout) * len(matrix.xout[0]))

    matrix.prod_proj([0, 1.], axes='out')
    assert np.allclose(matrix.value[:,:len(matrix.xout[0])], 0.)
    matrix.prod_proj([[0, 1.], [0., 1.]], axes=('out', 0))
    matrix.prod_proj([0, 1., 2., 1.], axes='in')
    assert matrix.shape == (len(matrix.projsin) * len(matrix.xin[0]), len(matrix.projsout) * len(matrix.xout[0]))

    matrix.prod_proj([[0, 1.]]*3, axes=('out', -1), projs=matrix.projsout + [matrix.projsout[-1]])
    assert len(matrix.xout) == len(matrix.projsout) == 3


def test_power_spectrum_odd_wideangle():
    ells = [0, 2, 4]
    kmin, kmax = 0., 0.2
    nk = 10
    dk = (kmax - kmin)/nk
    k = np.array([i*dk + dk/2. for i in range(nk)])
    d = 1.
    projsin = [Projection(ell=ell, wa_order=0) for ell in ells]
    projsout = [Projection(ell=ell, wa_order=ell % 2) for ell in range(ells[-1]+1)]
    PowerSpectrumOddWideAngleMatrix.propose_out(ells, wa_orders=1)
    wa = PowerSpectrumOddWideAngleMatrix(k, projsin, projsout=projsout, d=1., wa_orders=1, los='firstpoint')

    from wide_angle_tools import get_end_point_LOS_M
    ref = get_end_point_LOS_M(d, Nkth=nk, kmin=kmin, kmax=kmax)
    assert np.allclose(wa.value.T, ref)

    assert wa.projsout != wa.projsin
    wa.select_projs(projsout=projsin)
    assert wa.projsout == wa.projsin

    shape = wa.value.shape
    wa.rebin_x(factorout=2)
    assert wa.value.shape == (shape[0], shape[1]//2)
    assert np.allclose(wa.xout[0], (k[::2] + k[1::2])/2.)

    klim = (0., 0.15)
    mask = (k >= klim[0]) & (k <= klim[1])
    assert not np.all(mask)
    wa.select_x(xinlim=klim)
    assert np.allclose(wa.xin[0], k[mask])


if __name__ == '__main__':

    setup_logging()

    test_deriv()
    test_matrix()
    test_projection()
    test_power_spectrum_odd_wideangle()
