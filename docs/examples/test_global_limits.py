import distarray

a = distarray.zeros((16,16))
print a.comm_rank, a.global_limits(0)
print a.comm_rank, a.global_limits(1)