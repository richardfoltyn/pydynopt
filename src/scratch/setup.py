__author__ = 'Richard Foltyn'

from distutils.core import setup, Extension
from Cython.Build import cythonize


ext_module = Extension(
    "memviewtest",
    ["memviewtest.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(name='scratch',
      ext_modules=cythonize([ext_module],
                            include_path=['..']))