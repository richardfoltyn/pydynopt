from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(name='pydynopt',
      packages=['pydynopt.common', 'pydynopt.utils', 'pydynopt.interpolate'],
      ext_modules=cythonize(['pydynopt/common/*.pyx',
            'pydynopt/utils/*.pyx',
            'pydynopt/interpolate/*.pyx']))