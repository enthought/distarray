# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Create a distarray.
"""

from distarray.context import Context
from distarray.client_map import Distribution


c = Context()
d = Distribution.from_shape(c, (10, 10, 10), dist=('b', 'n', 'c'))
a = c.zeros(d, dtype='int32')
