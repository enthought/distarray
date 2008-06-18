import distarray as ipda

a = ipda.DistArray((64,64), dist=('b','b'))
a.plot_dist_matrix()


