import distarray as da
a = da.LocalArray((64,64), dist=('b','c'))
a.plot_dist_matrix()


