import distarray as ipda

a = ipda.LocalArray((64,64), dtype='int32', dist=(None,'b'))
a.plot_dist_matrix()