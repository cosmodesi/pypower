from setuptools import setup


setup(name='pypower',
      version='0.0.1',
      author='cosmodesi',
      author_email='',
      description='Estimation of power spectrum and window function',
      license='GPL3',
      url='http://github.com/cosmodesi/pypower',
      install_requires=['numpy', 'scipy', 'pmesh'],
      extras_require={'extras':['sympy', 'numexpr']},
      packages=['pypower']
)
