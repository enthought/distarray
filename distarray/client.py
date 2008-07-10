# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

class DistArrayContext(object):
    
    def __init__(self, multiengine_client):
        self.multiengine_client = multiengine_client
        self.multiengine_client.execute('import distarray as da')
    
    def todistarray(self, key, arr):
        dtype_name = key+'_dtype'
        shape_name = key+'_shape'
        local_name = key+'_local'
        self.multiengine_client.push({dtype_name:arr.dtype, shape_name:arr.shape})
        self.multiengine_client.scatter(local_name,arr)
        self.multiengine_client.execute('%s = da.DistArray(%s,dtype=%s,buf=%s)' % (key,shape_name,dtype_name,local_name))
    
    def fromdistarray(self, key):
        local_name = key+'_local'
        local_shape = key+'_shape'
        self.multiengine_client.execute('%s = %s.local_view(); %s = %s.shape' % (local_name,key,local_shape,key))
        shape = self.multiengine_client.pull(local_shape,0)[0]
        arr = self.multiengine_client.gather(local_name)
        arr.shape = shape
        return arr
    
    
    