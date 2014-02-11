import distarray as ipda

a = ipda.RemoteArray((8,64,64), dist=('b',None,'c'))
