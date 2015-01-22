from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import Extension, build_ext

import numpy as np

cdirectives_default = {
    'wraparound': False,
    'cdivision': True
}

cdirectives_debug = {
    'boundscheck': True,
    'overflowcheck': True,
    'nonecheck': True
}
cdirectives_debug.update(cdirectives_default)

cdirectives_profile = {
    'linetrace': True
}

exclude = ['scratch/*', '**/test*.pyx']
packages = ['pydynopt.common', 'pydynopt.utils', 'pydynopt.interpolate',
            'pydynopt.optimize']

ext = [Extension('*', ['**/*.pyx'],
                 cython_directives=cdirectives_debug)]

gdb = True

setup(name='pydynopt',
      packages=packages,
      ext_modules=cythonize(ext, exclude=exclude, gdb_debug=gdb,
                            include_path=[np.get_include()]))