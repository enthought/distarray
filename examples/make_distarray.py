# encoding: utf-8
# Copyright (c) 2008-2014, IPython Development Team and Enthought, Inc.
"""
Create a distarray, then plot its array distribution.
"""

__docformat__ = "restructuredtext en"

import distarray


c = distarray.Context()
a = c.zeros((10, 10, 10), dtype='int32', dist=('b', 'n', 'c'))
