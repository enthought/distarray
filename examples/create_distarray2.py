import distarray as ipda

a = ipda.RemoteArray((64,64), dtype='int32', dist=(None,'b'))
a.plot_dist_matrix()
