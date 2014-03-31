# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from __future__ import print_function

import distarray


a = distarray.zeros((16,16))
print(a.comm_rank, a.global_limits(0))
print(a.comm_rank, a.global_limits(1))
