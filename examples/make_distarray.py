# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Create a distarray, then plot its array distribution.
"""

import distarray


c = distarray.Context()
a = c.zeros((10, 10, 10), dtype='int32', dist=('b', 'n', 'c'))
