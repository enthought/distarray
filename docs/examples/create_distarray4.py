import ipythondistarray as ipda

a = ipda.DistArray((64,64), dist=('b','c'))
a.plot_dist_matrix()


