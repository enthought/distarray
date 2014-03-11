# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------


def plot_dist_matrix(name, mec):
    mec.execute('_dm = %s.get_dist_matrix()' % name, 0)
    _dm = mec.zip_pull('_dm', 0)
    import pylab
    pylab.ion()
    pylab.matshow(_dm)
    pylab.colorbar()
    pylab.show()
