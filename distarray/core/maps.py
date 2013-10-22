# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

import inspect
from distarray.core.error import InvalidMapCodeError


class Map(object):

    def __init__(self, shape, grid_shape):
        self.shape = shape
        self.grid_shape = grid_shape
        self.local_shape = self.shape // self.grid_shape
        if self.shape % self.grid_shape > 0:
            self.local_shape += 1

    def owner(self, i):
        raise NotImplemented("Implement in subclass.")

    def local_index(self, i):
        raise NotImplemented("Implement in subclass.")

    def global_index(self, owner, p):
        raise NotImplemented("Implement in subclass.")


class BlockMap(Map):

    def owner(self, i):
        return i // self.local_shape

    def local_index(self, i):
        return i % self.local_shape

    def global_index(self, owner, p):
        return owner * self.local_shape + p


class CyclicMap(Map):

    def owner(self, i):
        return i % self.grid_shape

    def local_index(self, i):
        return i // self.grid_shape

    def global_index(self, owner, p):
        return owner + p * self.grid_shape


class BlockCyclicMap(Map):
    # http://netlib.org/scalapack/slug/node76.html

    def __init__(self, shape, grid_shape, block_size):
        super(BlockCyclicMap, self).__init__(shape, grid_shape)
        self.block_size = block_size

    def owner(self, i):
        return (i // self.block_size) % self.grid_shape

    def local_index(self, i):
        local_block_number = i // (self.block_size * self.grid_shape)
        offset = i % self.block_size
        return local_block_number * self.block_size + offset

    def global_index(self, owner, p):
        local_block_number = p // self.block_size
        offset = p % self.block_size
        return ((local_block_number*self.grid_shape + owner) *
                self.block_size + offset)


class MapRegistry(object):

    def __init__(self):
        self.maps = {}

    def register_map(self, code, m):
        if inspect.isclass(m):
            if issubclass(m, Map):
                self.maps[code] = m
            else:
                raise TypeError("Must register a Map subclass.")
        else:
            raise TypeError("Must register a class.")

    def get_map_class(self, code):
        m = self.maps.get(code)
        if m is None:
            if inspect.isclass(code):
                if issubclass(code, Map):
                    return code
                else:
                    msg = "Not a Map subclass or a valid map code: %s." % code
                    raise InvalidMapCodeError(msg)
            else:
                msg = "Not a Map subclass or a valid map code: %s." % code
                raise InvalidMapCodeError(msg)
        else:
            return m


_map_registry = MapRegistry()
register_map = _map_registry.register_map
get_map_class = _map_registry.get_map_class

register_map('b', BlockMap)
register_map('c', CyclicMap)
register_map('bc', BlockCyclicMap)

# bp1 = BlockMap(16, 2)
# bp2 = CyclicMap(16, 2)
#
# import numpy
# result = numpy.empty((16,16),dtype='int32')
#
# grid = numpy.arange(4, dtype='int32')
# grid.shape=(2,2)
#
# for i in range(16):
#     for j in range(16):
#         # print bp1.owner(i), bp2.owner(j)
#         result[i,j] = grid[bp1.owner(i), bp2.owner(j)]
