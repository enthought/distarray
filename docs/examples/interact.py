from IPython.kernel import client
from distarray import client as daclient
import numpy as np
import distarray as da
mec = client.MultiEngineClient()
mec.execute('import distarray as da')
dac = daclient.DistArrayContext(mec)
mec.activate()
