# encoding: utf-8

__docformat__ = "restructuredtext en"

from setuptools import setup


metadata = {
    'name'             : 'distarray',
    'version'          : '0.1',
    'description'      : 'Distributed Memory Arrays for Python',
    'keywords'         : 'parallel mpi distributed array',
    'license'          : 'New BSD',
    'author'           : 'Brian E. Granger',
    'author_email'     : 'ellisonbg@gmail.com',
    'url'              : 'https://github.com/bgrant/ennui'
    }


packages = [
    'distarray',
    'distarray.tests',
    'distarray.core',
    'distarray.core.tests',
    'distarray.random',
    'distarray.random.tests',
    'distarray.mpi',
    'distarray.mpi.tests',
    ]

install_requires = [
    'mpi4py',
    'ipython'
    ]

setup(packages=packages, **metadata)
