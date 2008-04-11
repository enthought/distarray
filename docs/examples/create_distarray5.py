import ipythondistarray as ipda

a = ipda.DistArray((8,64), dist=('b','b'))
a.plot_dist_matrix()


