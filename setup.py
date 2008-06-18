# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

from distutils.core import setup
from mpidistutils import Distribution, Extension, Executable
from mpidistutils import config, build, build_ext
from mpidistutils import build_exe, install_exe, clean_exe
import mpi4py

#--------- -------------------------------------------------------------------
# Metadata
#----------------------------------------------------------------------------

metadata = {
    'name'             : 'ipythondistarray',
    'version'          : '0.1',
    'description'      : 'Distributed Memory Arrays for Python',
    'keywords'         : 'parallel mpi distributed array',
    'license'          : 'New BSD',
    'author'           : 'Brian E. Granger',
    'author_email'     : 'ellisonbg@gmail.com',
    }

#----------------------------------------------------------------------------
# Extension modules
#----------------------------------------------------------------------------

def find_ext_modules():
    import sys
    
    maps = Extension(
        name='ipythondistarray.core.maps_fast',
        sources=['ipythondistarray/core/maps_fast.c']
    )
    # This extension shows how to call mpi4py's C layer using Cython
    mpi_test = Extension(
        name='ipythondistarray.mpi.tests.helloworld',
        sources=['ipythondistarray/mpi/tests/helloworld.c'],
        include_dirs = [mpi4py.get_include()]
    )
    allext = [maps, mpi_test]
    return allext

def find_headers():
    # allheaders = ['mpi/ext/libmpi.h']
    return []

def find_executables():
    return []

def find_packages():
    return    ['ipythondistarray',
              'ipythondistarray.tests',
              'ipythondistarray.core',
              'ipythondistarray.core.tests',
              'ipythondistarray.mpi',
              'ipythondistarray.mpi.tests',
              'ipythondistarray.random',
              'ipythondistarray.random.tests',
              'ipythondistarray.linalg',
              'ipythondistarray.linalg.tests',
              'ipythondistarray.fft',
              'ipythondistarray.fft.tests']


#----------------------------------------------------------------------------
# Setup
#----------------------------------------------------------------------------


def main():
    setup(packages = find_packages(),
          package_data = {'ipythondistarray' : ['include/*.pxi']},
          headers = find_headers(),
          ext_modules = find_ext_modules(),
          executables = find_executables(),
          distclass = Distribution,
          cmdclass = {'config'      : config,
                      'build'       : build,
                      'build_ext'   : build_ext,
                      'build_exe'   : build_exe,
                      'clean_exe'   : clean_exe,
                      'install_exe' : install_exe,
                      },
          **metadata)

if __name__ == '__main__':
    # hack distutils.sysconfig to eliminate debug flags
    from distutils import sysconfig
    cvars = sysconfig.get_config_vars()
    cflags = cvars.get('OPT')
    if cflags:
        cflags = cflags.split()
        for flag in ('-g', '-g3'):
            if flag in cflags:
                cflags.remove(flag)
        cvars['OPT'] = str.join(' ', cflags)
    # and now call main
    main()
