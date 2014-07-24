# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
The single IPython entry point.
"""

try:
    from IPython.parallel import Client as IPythonClient
except (ImportError, IOError):
    IPythonClient = None
