import os

import numpy as np
from matplotlib import pyplot as plt
from nbodykit import setup_logging
from nbodykit.transform import SkyToCartesian
from nbodykit.utils import ScatterArray, GatherArray
from nbodykit.lab import cosmology, LogNormalCatalog, UniformCatalog


base_dir = '_catalog'
data_fn = os.path.join(base_dir,'lognormal_data.fits')
randoms_fn = os.path.join(base_dir,'lognormal_randoms.fits')
data_masked_fn = os.path.join(base_dir,'lognormal_masked_data.fits')
randoms_masked_fn = os.path.join(base_dir,'lognormal_masked_randoms.fits')

cosmo = cosmology.Planck15
nbar = 1e-3


def generate_lognormal():
    redshift = 1.
    Plin = cosmology.LinearPower(cosmo,redshift,transfer='CLASS')
    BoxSize = 600.
    bias = 2.0
    #Nmesh = 256
    Nmesh = 64
    seed = 42
    data = LogNormalCatalog(Plin=Plin,nbar=nbar,BoxSize=BoxSize,Nmesh=Nmesh,bias=bias,seed=seed,unitary_amplitude=True)
    offset = cosmo.comoving_distance(redshift) - BoxSize/2.
    data['Position'][:,0] += offset
    los = data['Position']/np.sqrt(np.sum(data['Position']**2,axis=-1))[:,None]
    data['RSDPosition'] = data['Position'] + np.sum(data['VelocityOffset']*los,axis=-1)[:,None]*los
    data['Weight'] = np.ones(data.size)
    data['NZ'] = np.ones(data.size)*nbar

    randoms = UniformCatalog(10.*nbar,BoxSize,seed=42,dtype='f8')
    randoms['Position'][:,0] += offset
    randoms['Weight'] = np.ones(randoms.size)
    randoms['NZ'] = np.ones(randoms.size)*nbar

    return data, randoms


def test_fkp_power():
    BoxSize = 1000.
    Nmesh = 128
    dk = 0.01
    ells = [0,2,4]
    data, randoms = generate_lognormal()

    def get_ref_power(data, randoms):
        from nbodykit.lab import FKPCatalog, ConvolvedFFTPower
        fkp = FKPCatalog(data,randoms,nbar='NZ')
        mesh = fkp.to_mesh(position='Position',comp_weight='Weight',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
        return ConvolvedFFTPower(mesh,poles=ells,dk=dk)

    def get_fkp_power(data, randoms):
        from pypower import FKPCatalog, ConvolvedFKPFFTPower
        fkp = FKPCatalog(data,randoms)
        mesh = fkp.to_mesh(position='Position',weight='Weight',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
        return ConvolvedFKPFFTPower(mesh,poles=ells,edges={'step':dk})

    ref_power = get_ref_power(data,randoms)
    power = get_fkp_power(data,randoms)
    ref_norm = ref_power.attrs['randoms.norm']
    norm = power.attrs['norm']
    print(norm/ref_norm)
    for ell in ells:
        #print(power.poles['power_{}'.format(ell)].real*norm/ref_norm,ref_power.poles['power_{}'.format(ell)].real)
        assert np.allclose(power.poles['power_{}'.format(ell)].real*norm/ref_norm,ref_power.poles['power_{}'.format(ell)].real)

    def get_fkp_power_cross(data, randoms):
        from pypower import FKPCatalog, ConvolvedFKPFFTPower
        fkp = FKPCatalog(data,randoms)
        mesh = fkp.to_mesh(position='Position',weight='Weight',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
        second = fkp.to_mesh(position='Position',weight='Weight',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
        return ConvolvedFKPFFTPower(mesh,second=second,poles=ells,edges={'step':dk})

    cross = get_fkp_power_cross(data,randoms)
    for ell in ells:
        assert np.allclose(power.poles['power_{}'.format(ell)].real,cross.poles['power_{}'.format(ell)].real)


def test_mesh_power():
    BoxSize = 1000.
    Nmesh = 128
    dk = 0.01
    ells = [0]
    data, randoms = generate_lognormal()

    def get_ref_power(data, randoms):
        from nbodykit.lab import FKPCatalog, ConvolvedFFTPower
        fkp = FKPCatalog(data,randoms,nbar='NZ')
        mesh = fkp.to_mesh(position='Position',comp_weight='Weight',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
        return ConvolvedFFTPower(mesh,poles=ells,dk=dk)

    def get_mesh_power(data, randoms):
        from pypower import FKPCatalog, ConvolvedMeshFFTPower
        fkp = FKPCatalog(data,randoms)
        mesh = fkp.to_mesh(position='Position',weight='Weight',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True,compensated=True)
        return ConvolvedMeshFFTPower(mesh,poles=ells,edges={'step':dk})

    ref_power = get_ref_power(data,randoms)
    power = get_mesh_power(data,randoms)
    ref_norm = ref_power.attrs['randoms.norm']
    norm = power.attrs['norm']
    for ell in ells:
        #print(power.poles['power_{}'.format(ell)].real*norm/ref_norm,ref_power.poles['power_{}'.format(ell)].real)
        assert np.allclose(power.poles['power_{}'.format(ell)].real*norm/ref_norm,ref_power.poles['power_{}'.format(ell)].real)

    def get_mesh_power_cross(data, randoms):
        from pypower import FKPCatalog, ConvolvedMeshFFTPower
        fkp = FKPCatalog(data,randoms)
        mesh = fkp.to_mesh(position='Position',weight='Weight',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True,compensated=True)
        second = fkp.to_mesh(position='Position',weight='Weight',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True,compensated=True)
        return ConvolvedMeshFFTPower(mesh,second=second,poles=ells,edges={'step':dk})

    cross = get_mesh_power_cross(data,randoms)
    for ell in ells:
        assert np.allclose(power.poles['power_{}'.format(ell)].real,cross.poles['power_{}'.format(ell)].real)


if __name__ == '__main__':

    setup_logging()
    test_fkp_power()
    test_mesh_power()
