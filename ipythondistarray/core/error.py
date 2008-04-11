#----------------------------------------------------------------------------
# Exports
#----------------------------------------------------------------------------

__all__ = ['InvalidBaseCommError',
    'InvalidGridShapeError',
    'GridShapeError',
    'DistError',
    'DistMatrixError',
    'IncompatibleArrayError',
    'NullCommError']

#----------------------------------------------------------------------------
# Exceptions
#----------------------------------------------------------------------------

class DistArrayError(Exception):
    pass

class InvalidBaseCommError(DistArrayError):
    pass

class InvalidGridShapeError(DistArrayError):
    pass

class GridShapeError(DistArrayError):
    pass

class DistError(DistArrayError):
    pass

class DistMatrixError(DistArrayError):
    pass

class IncompatibleArrayError(DistArrayError):
    pass

class NullCommError(DistArrayError):
    pass
