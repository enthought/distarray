from ipythondistarray.core.error import DistArrayError

__all__ = [
    'MPIDistArrayError',
    'InvalidCommSizeError',
    'InvalidRankError']

class MPIDistArrayError(DistArrayError):
    pass

class InvalidCommSizeError(MPIDistArrayError):
    pass

class InvalidRankError(MPIDistArrayError):
    pass

