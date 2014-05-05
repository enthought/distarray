# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Create a distarray, then plot its array distribution.
"""

from matplotlib import pyplot

import distarray
from distarray import plotting
from distarray.client_map import Distribution


c = distarray.Context()
d = Distribution.from_shape(c, (64, 64), dist=('n', 'b'))
a = c.zeros(d, dtype='int32')
process_coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
plotting.plot_array_distribution(a, process_coords, cell_label=False,
                                 legend=True)
pyplot.show()
