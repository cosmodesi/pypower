.. _user-building:

Building
========

Requirements
------------
Only strict requirements are:

  - numpy
  - scipy
  - pmesh

To enable faster spherical harmonics computation:

  - sympy
  - numexpr

pip
---
To install **pypower**, simply run::

  python -m pip install git+https://github.com/cosmodesi/pypower

To install sympy, numexpr::

  python -m pip install git+https://github.com/cosmodesi/pypower#egg=pypower[extras]

git
---
First::

  git clone https://github.com/cosmodesi/pypower.git

To install the code::

  python setup.py install --user

Or in development mode (any change to Python code will take place immediately)::

  python setup.py develop --user
