from distutils.core import setup, Extension
from Cython.Build import cythonize


files = ['pydynopt/common/*.pyx', 'pydynopt/utils/*.pyx',
         'pydynopt/interpolate/*.pyx', 'pydynopt/optimize/*.pyx']

exclude = ['scratch/*', 'pydynopt/utils/utils_*', '**/test.pyx']
packages = ['pydynopt.common', 'pydynopt.utils', 'pydynopt.interpolate',
            'pydynopt.optimize']


setup(name='pydynopt',
      packages=packages,
      ext_modules=cythonize(files, exclude=exclude))