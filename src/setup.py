from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import Extension, build_ext

import numpy as np

cdir_default = {
    'wraparound': False,
    'cdivision': True,
    'boundscheck': False
}

cdir_debug = {
    'boundscheck': True,
    'overflowcheck': True,
    'wraparound': False,
    'cdivision': True,
    'nonecheck': True
}


cdir_profile = {
    # 'linetrace': True
    'profile': True,
    'wraparound': False,
    'cdivision': True,
    'boundscheck': False
}

exclude = ['scratch/*', '**/test*.pyx']
packages = ['pydynopt.common', 'pydynopt.utils', 'pydynopt.interpolate',
            'pydynopt.optimize']

ext = [Extension('*', ['**/*.pyx'])]

gdb = False
annotate = True

setup(name='pydynopt',
      packages=packages,
      ext_modules=cythonize(ext, exclude=exclude, gdb_debug=gdb,
                            include_path=[np.get_include()],
                            compiler_directives=cdir_default,
                            annotate=annotate))