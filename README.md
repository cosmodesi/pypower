# pypower

**pypower** is a package for auto and cross power spectrum and associated window function estimation,
for periodic boxes, survey geometry, in the flat-sky or plane-parallel (plus first odd wide-angle corrections) configurations.

A typical auto power spectrum estimation is as simple as:
```
import numpy as np
from pypower import CatalogFFTPower

kedges = np.linspace(0., 0.2, 11)
# pass mpiroot=0 if input positions and weights are not MPI-scattered
result = CatalogFFTPower(data_positions1=data_positions1, data_weights1=data_weights1, randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                         edges=kedges, ells=(0, 2, 4), boxsize=1000., nmesh=512, resampler='tsc', interlacing=2, los=None, position_type='pos')
# wavenumber array in result.poles.k
# multipoles in result.poles.power
```

Example notebooks presenting most use cases are provided in directory nb/.

# Requirements

Only strict requirements are:

  - numpy
  - scipy
  - pmesh

To enable faster spherical harmonics computation:

  - sympy
  - numexpr

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/adematti/pypower
```

### git

First:
```
git clone https://github.com/adematti/pypower.git
```
To install the code:
```
python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately)::
```
python setup.py develop --user
```

## License

**pypower** is free software distributed under a GPLv3 license. For details see the [LICENSE](https://github.com/adematti/pypower/blob/main/LICENSE).

## Credits

[nbodykit](https://github.com/bccp/nbodykit) for recipe and most routines for power spectrum estimation.
Patrick McDonald for [recipe](https://github.com/cosmodesi/GC_derivations) for window function matrix estimation.
