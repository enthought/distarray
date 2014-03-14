# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Plot distributions for some distarrays for the protocol documentation.
"""

import distarray
from distarray import plotting


# Use 2 processors to match examples.
c = distarray.Context(targets=[0, 1])

# Block-Nondistributed.
a = c.zeros((2, 10), dist=('b', 'n'))

plotting.plot_array_distribution_2d(a, draw_legend=True)
plotting.show(title='Block-Nondistributed\n', filename='plot_block_nondist.png')
