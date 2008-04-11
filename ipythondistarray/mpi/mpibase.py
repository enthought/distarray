from mpi4py import MPI

from ipythondistarray.mpi.error import *

__all__ = [
    'COMM_PRIVATE',
    'MPI',
    'create_comm_of_size',
    'create_comm_with_list']


COMM_PRIVATE = MPI.COMM_WORLD.Clone()

def create_comm_of_size(size=4):
    """
    Create a subcommunicator of COMM_PRIVATE of given size.
    """
    group = COMM_PRIVATE.Get_group()
    comm_size = COMM_PRIVATE.Get_size()
    if size > comm_size:
        raise InvalidCommSizeError("requested size (%i) is bigger than the comm size (%i)" % (size, comm_size))
    else:
        subgroup = group.Incl(range(size))
        newcomm = COMM_PRIVATE.Create(subgroup)
        return newcomm

def create_comm_with_list(nodes):
    """
    Create a subcommunicator of COMM_PRIVATE with a list of ranks.
    """
    group = COMM_PRIVATE.Get_group()
    comm_size = COMM_PRIVATE.Get_size()
    size = len(nodes)
    if size > comm_size:
        raise InvalidCommSizeError("requested size (%i) is bigger than the comm size (%i)" % (size, comm_size))
    for i in nodes:
        if not i in range(comm_size):
            raise InvalidRankError("rank is not valid: %r" % i)
    subgroup = group.Incl(nodes)
    newcomm = COMM_PRIVATE.Create(subgroup)
    return newcomm    
    
    