import ipythondistarray as ipda

a = ipda.DistArray((64,64), dtype='int32', dist=(None,'b'))
a.plot_dist_matrix()