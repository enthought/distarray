import distarray

a = distarray.zeros((16,16))
print a.comm_rank, a.global_corners(0)
print a.comm_rank, a.global_corners(1)

b = distarray.zeros((16,16),dist=('c','b'))
print b.comm_rank, b.global_corners(0)