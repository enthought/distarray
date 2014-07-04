# encoding: utf-8
# -----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# -----------------------------------------------------------------------------

from setuptools import setup, find_packages


def parse_readme(filename='README.rst', skiplines=6):
    """Return file `filename` as a string, skipping some lines."""
    with open(filename, 'r') as fp:
        for line in range(skiplines):
            fp.readline()
        return fp.read()


install_requires = [
    'ipython',
    'numpy',
    'mpi4py'
]

metadata = {
    'name': 'distarray',
    'version': '0.4.0-dev',
    'description': 'Distributed Memory Arrays for Python',
    'keywords': 'parallel mpi distributed array',
    'license': 'New BSD',
    'author': 'IPython Development Team and Enthought, Inc.',
    'author_email': 'ksmith@enthought.com',
    'url': 'https://github.com/enthought/distarray',
    'packages': find_packages(),
    'install_requires': install_requires,
    'long_description': parse_readme(),
    'platforms': ["Linux", "Mac OS-X"],
    'entry_points': {'console_scripts': ['dacluster = '
                                         'distarray.apps.dacluster:main']},
    'classifiers': [c.strip() for c in """\
        Development Status :: 2 - Pre-Alpha
        Intended Audience :: Developers
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Operating System :: MacOS
        Operating System :: OS Independent
        Operating System :: POSIX
        Operating System :: Unix
        Programming Language :: Python
        Topic :: Scientific/Engineering
        Topic :: Software Development
        Topic :: Software Development :: Libraries
        """.splitlines() if len(c.strip()) > 0],
}

setup(**metadata)
