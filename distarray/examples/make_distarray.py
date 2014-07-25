# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Create a distarray.
"""

from distarray.dist import Context, Distribution


c = Context()
d = Distribution(c, (10, 10, 10), dist=('b', 'n', 'c'))
a = c.zeros(d, dtype='int32')
