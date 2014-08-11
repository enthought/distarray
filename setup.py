# encoding: utf-8
# -----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# -----------------------------------------------------------------------------

from setuptools import setup, find_packages
from distarray.__version__ import __version__


def parse_readme(filename='README.rst', sentinel="README"):
    """
    Return file `filename` as a string.

    Skips lines until it finds a comment that contains `sentinel` text.  This
    effectively strips off any badges or other undesirable content.
    """
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    for idx, line in enumerate(lines):
        if line.startswith('..') and sentinel in line:
            break
    return "".join(lines[idx+1:])


if __name__ == "__main__":

    install_requires = [
        'ipython',
        'numpy',
        'mpi4py'
    ]

    metadata = {
        'name': 'distarray',
        'version': __version__,
        'description': 'Distributed Memory Arrays for Python',
        'keywords': 'parallel mpi distributed array',
        'license': 'New BSD',
        'author': 'IPython Development Team and Enthought, Inc.',
        'maintainer': "DistArray Developers",
        'maintainer_email': "distarray@googlegroups.com",
        'url': 'https://github.com/enthought/distarray',
        'packages': find_packages(),
        'install_requires': install_requires,
        'long_description': parse_readme(),
        'platforms': ["Linux", "Mac OS-X"],
        'entry_points': {'console_scripts': ['dacluster = '
                                             'distarray.apps.dacluster:main']},
        'classifiers': [c.strip() for c in """\
            Development Status :: 3 - Alpha
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
