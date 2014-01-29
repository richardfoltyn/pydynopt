from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

extensions = [
    Extension("cutils", ["cutils.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    name="pydynopt utils",
    ext_modules=cythonize(extensions),
    )