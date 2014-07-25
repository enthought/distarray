from distutils.core import setup
from Cython.Build import cythonize

setup(name="kernel", ext_modules=cythonize("kernel.pyx"))
