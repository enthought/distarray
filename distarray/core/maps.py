# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

import inspect
from abc import ABCMeta, abstractproperty, abstractmethod
from distarray.core.error import InvalidMapCodeError


class ILocalMap:

    __metaclass__ = ABCMeta

    @abstractproperty
    def local_shape(self):
        pass

    @abstractmethod
    def owner(self, i):
        pass

    @abstractmethod
    def local_index(self, i):
        pass

    @abstractmethod
    def global_index(self, owner, p):
        pass


def regular_local_shape(shape, grid_shape):
    local_shape = shape // grid_shape
    if shape % grid_shape > 0:
        local_shape += 1
    return local_shape


class BlockCyclicMap(ILocalMap):

    def __init__(self, shape, grid_shape, block_size):
        self.shape = shape
        self.grid_shape = grid_shape
        self.block_size = block_size

    @property
    def local_shape(self):
        return regular_local_shape(self.shape, self.grid_shape)

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


class BlockMap(BlockCyclicMap):

    def __init__(self, shape, grid_shape):
        super(BlockMap, self).__init__(shape, grid_shape,
                block_size=regular_local_shape(shape, grid_shape))


class CyclicMap(BlockCyclicMap):

    def __init__(self, shape, grid_shape):
        super(CyclicMap, self).__init__(shape, grid_shape, block_size=1)


class MapRegistry(object):

    def __init__(self):
        self.maps = {}

    def register_map(self, code, m):
        if inspect.isclass(m):
            if issubclass(m, ILocalMap):
                self.maps[code] = m
            else:
                raise TypeError("Must register a Map subclass.")
        else:
            raise TypeError("Must register a class.")

    def get_map_class(self, code):
        m = self.maps.get(code)
        if m is None:
            if inspect.isclass(code):
                if issubclass(code, ILocalMap):
                    return code
                else:
                    msg = "Not an ILocalMap subclass or a valid map code: %s." % code
                    raise InvalidMapCodeError(msg)
            else:
                msg = "Not a ILocalMap subclass or a valid map code: %s." % code
                raise InvalidMapCodeError(msg)
        else:
            return m


_map_registry = MapRegistry()
register_map = _map_registry.register_map
get_map_class = _map_registry.get_map_class

register_map('b', BlockMap)
register_map('c', CyclicMap)
register_map('bc', BlockCyclicMap)
