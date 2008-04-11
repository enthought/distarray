from math import sqrt

def divisors(n):
    i = 2
    while i<n:
        if n % i == 0:
            yield i
        i += 1

def multi_for(iterables):
    if not iterables:
        yield ()
    else:
        for item in iterables[0]:
            for rest_tuple in multi_for(iterables[1:]):
                yield (item,) + rest_tuple

def create_factors(n, size=2):
    divs = list(divisors(n))
    factors = []
    for indices in multi_for( [xrange(p) for p in size*[len(divs)]] ):
        total = 1
        for i in indices:
            total = total*divs[i]
        if n == total:
            factor = [divs[i] for i in indices]
            factor.sort()
            factor = tuple(factor)
            if factor not in factors:
                factors.append(factor)
    return factors


def divisors_minmax(n, dmin, dmax):
    """Find the divisors of n in the interval (dmin,dmax]."""
    i = dmin+1
    while i<=dmax:
        if n % i == 0:
            yield i
        i += 1
        
def list_or_tuple(seq):
    return isinstance(seq, (list, tuple))
        
def flatten(seq, to_expand=list_or_tuple):
    """Flatten a nested sequence."""
    for item in seq:
        if to_expand(item):
            for subitem in flatten(item, to_expand):
                yield subitem
        else:
            yield item

def mult_partitions(n, s):
    """Compute the multiplicative partitions of n of size s
    
    >>> mult_partitions(52,3)
    [(2, 2, 13)]
    >>> mult_partitions(52,2)
    [(2, 26), (4, 13)]
    """
    return [tuple(flatten(p)) for p in mult_partitions_recurs(n,s)]

def mult_partitions_recurs(n, s, pd=1):
    if s == 1:
        return [n]        
    divs = divisors_minmax(n, pd, int(sqrt(n)))
    fs = []
    for d in divs:
        fs.extend([(d,f) for f in mult_partitions_recurs(n/d, s-1, pd)])
        pd = d
    return fs


def mirror_sort(seq, ref_seq):
    """Sort s2 into the order that s1 is in.
    
    >>> mirror_sort(range(5),[1,5,2,4,3])
    [0, 4, 1, 3, 2]
    """
    assert len(seq)==len(ref_seq), "Sequences must have the same length"
    shift = zip(range(len(ref_seq)),ref_seq)
    shift.sort(key=lambda x:x[1])
    shift = [s[0] for s in shift]
    newseq = len(ref_seq)*[0]
    for s_index in range(len(shift)):
        newseq[shift[s_index]] = seq[s_index]
    return newseq

def outer_zip(seqa, seqb):
    """An outer product, but using zip rather than *.
    
    >>> a = 2*range(4)
    >>> b = range(8)
    >>> outer_zip(a,b)
    [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)], [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)], [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)], [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)], [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)], [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)], [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)], [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]]
    
    """
    return [[(i,j) for j in seqb] for i in seqa]

def _raise_nie():
    raise NotImplementedError("This has not yet been implemented for distributed arrays")
