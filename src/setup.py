from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(name='pydynopt',
      packages=['pydynopt.common', 'pydynopt.utils'],
      ext_modules=cythonize(['pydynopt/common/*.pyx',
            'pydynopt/utils/*.pyx']))