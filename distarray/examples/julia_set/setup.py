from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension(name="kernel", sources=["kernel.pyx"])

setup(name="kernel",
      ext_modules=cythonize(ext))
