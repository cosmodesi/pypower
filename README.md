# pypower

**pypower** is a wrapper for power spectrum estimation.

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
To install the code::
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
