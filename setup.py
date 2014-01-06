# encoding: utf-8

__docformat__ = "restructuredtext en"

from setuptools import setup, find_packages


install_requires = [
    'ipython',
    'numpy',
    'six',
    'mpi4py'
]

metadata = {
    'name': 'distarray',
    'version': '0.1',
    'description': 'Distributed Memory Arrays for Python',
    'keywords': 'parallel mpi distributed array',
    'license': 'New BSD',
    'author': 'Brian E. Granger',
    'author_email': 'ellisonbg@gmail.com',
    'url': 'https://github.com/enthought/distarray',
    'packages': find_packages(),
    'install_requires': install_requires
}

setup(**metadata)
