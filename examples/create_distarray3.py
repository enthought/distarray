import distarray as ipda

a = ipda.RemoteArray((64,64), dist=('b','b'))
a.plot_dist_matrix()
