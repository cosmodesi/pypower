import numpy as np
from mockfactory import Catalog

from pypower.fft_power import normalization
from pypower import MeshFFTPower, CatalogMesh, CatalogFFTResidual, setup_logging

from test_fft_power import data_fn, randoms_fn


def test_residual():
    boxsize = 1000.
    nmesh = 128
    kedges = np.linspace(0., 0.3, 6)
    ells = (0, 1, 2, 3, 4)
    resampler = 'tsc'
    interlacing = 2
    boxcenter = np.array([3000., 0., 0.])[None, :]
    los = None
    data = Catalog.read(data_fn)
    randoms = Catalog.read(randoms_fn)
    weight_value = 2.
    for catalog in [data, randoms]:
        catalog['Position'] += boxcenter
        catalog['Weight'] = weight_value * catalog.ones()

    def get_ref_residual(data, randoms, randoms2, weights=('data', 'randoms', 'randoms2')):
        data_weights = data['Weight'] if 'data' in weights else None
        randoms_weights = randoms['Weight'] if 'randoms' in weights else None
        randoms_weights2 = randoms2['Weight'] if 'randoms2' in weights else None
        mesh = CatalogMesh(data_positions=data['Position'], data_weights=data_weights, randoms_positions=randoms['Position'], randoms_weights=randoms_weights,
                           boxsize=boxsize, boxcenter=boxcenter, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos')
        mesh2 = CatalogMesh(data_positions=randoms2['Position'], data_weights=randoms_weights2,
                            boxsize=boxsize, boxcenter=boxcenter, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos')
        return MeshFFTPower(mesh, mesh2=mesh2, ells=ells, los=los, edges=kedges, wnorm=normalization(mesh, mesh2, fields=[('randoms', 'data')])).poles

    def get_residual(data, randoms=None, randoms2=None, shifted=None, weights=('data', 'randoms', 'randoms2', 'shifted')):
        data_weights = data['Weight'] if 'data' in weights else None
        randoms_weights = randoms['Weight'] if 'randoms' in weights and randoms is not None else None
        shifted_weights = shifted['Weight'] if 'shifted' in weights and shifted is not None else None
        randoms_weights2 = randoms2['Weight'] if 'randoms2' in weights and randoms2 is not None else None
        return CatalogFFTResidual(data_positions1=data['Position'], data_weights1=data_weights, randoms_positions1=randoms['Position'] if randoms is not None else None, randoms_weights1=randoms_weights,
                                  randoms_positions2=randoms2['Position'] if randoms2 is not None else None, randoms_weights2=randoms_weights2,
                                  shifted_positions1=shifted['Position'] if shifted is not None else None, shifted_weights1=shifted_weights,
                                  boxsize=boxsize, boxcenter=boxcenter, nmesh=nmesh, resampler=resampler, interlacing=interlacing, ells=ells, los=los, edges=kedges, position_type='pos').poles

    ref = get_ref_residual(data, randoms, randoms)

    test = get_residual(data, randoms=randoms, randoms2=randoms)
    assert np.allclose(test.power, ref.power)
    boxcenter = None
    ref = get_residual(data, randoms)
    assert ref.shotnoise != 0.

    test = get_residual(data, randoms=randoms, randoms2=randoms, shifted=randoms)
    assert test.shotnoise == 0.
    assert np.allclose(test.power, ref.get_power(remove_shotnoise=False))

    test = get_residual(data, randoms=randoms, shifted=randoms)
    assert test.shotnoise == 0.
    assert np.allclose(test.power, ref.get_power(remove_shotnoise=False))


if __name__ == '__main__':

    setup_logging('debug')

    test_residual()
