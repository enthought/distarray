import distarray as ipda

a = ipda.RemoteArray((8,64), dist=('b','b'))
a.plot_dist_matrix()
