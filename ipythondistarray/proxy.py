from ipython1.kernel import client
import numpy as np

def plot_dist_matrix(name, mec):
    mec.execute('_dm = %s.get_dist_matrix()' % name, 0)
    _dm = mec.zip_pull('_dm', 0)
    import pylab
    pylab.ion()
    pylab.matshow(_dm)
    pylab.colorbar()
    pylab.show()