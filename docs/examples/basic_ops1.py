import ipythondistarray as ipda

a = ipda.random.rand((10,100,100), dist=(None,'b','c'))
b = ipda.random.rand((10,100,100), dist=(None,'b','c'))
c = 0.5*ipda.sin(a) + 0.5*ipda.cos(b)
print c.sum(), c.mean(), c.std(), c.var()