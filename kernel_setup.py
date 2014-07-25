from distutils.core import setup
from Cython.Build import cythonize

setup(name="distarray.examples.julia_set.kernel",
      ext_modules=cythonize("distarray/examples/julia_set/kernel.pyx"))
