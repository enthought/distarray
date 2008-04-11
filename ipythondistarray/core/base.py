
__all__ = ['BaseDistArray',
    'arecompatible']

class BaseDistArray(object):
    
    def compatibility_hash(self):
        raise NotImplementedError("subclasses must implement compatibility_hash")


def arecompatible(a, b):
    """
    Do these arrays have the same compatibility hash?
    """
    return a.compatibility_hash() == b.compatibility_hash()
    