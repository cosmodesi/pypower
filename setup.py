import os
import sys
from setuptools import setup


package_basename = 'pypower'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='cosmodesi',
      author_email='',
      description='Estimation of power spectrum and window function',
      license='BSD3',
      url='http://github.com/cosmodesi/pypower',
      install_requires=['numpy', 'scipy'],  # just to get the data types
      extras_require={'extras': ['pmesh @ git+https://github.com/MP-Gadget/pmesh', 'sympy', 'numexpr']},
      packages=[package_basename])
