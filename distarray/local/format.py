# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Define a simple format for saving LocalArrays to disk with full information
about them.  This format, ``.dnpy``, draws heavily from the ``.npy`` format
specification from NumPy and from the data structure defined in the Distributed
Array Protocol.

Version numbering
-----------------

The version numbering of this format is independent of DistArray's and
the Distributed Array Protocol's version numberings.

Format Version 1.0
------------------

The first 6 bytes are a magic string: exactly ``\\x93DARRY``.

The next 1 byte is an unsigned byte: the major version number of the file
format, e.g. ``\\x01``.

The next 1 byte is an unsigned byte: the minor version number of the file
format, e.g. ``\\x00``. Note: the version of the file format is not tied
to the version of the DistArray package.

The next 2 bytes form a little-endian unsigned short int: the length of
the header data HEADER_LEN.

The next HEADER_LEN bytes form the header data describing the
distribution of this chunk of the LocalArray.  It is an ASCII string
which contains a Python literal expression of a dictionary. It is
terminated by a newline (``\\n``) and padded with spaces (``\\x20``) to
make the total length of ``magic string + 4 + HEADER_LEN`` be evenly
divisible by 16 for alignment purposes.

The dictionary contains two keys, both described in the Distributed
Array Protocol:

    "__version__" : str
        Version of the Distributed Array Protocol used in this header.
    "dim_data" : tuple of dict
        One dictionary per array dimension; see the Distributed Array
        Protocol for the details of this data structure.

For repeatability and readability, the dictionary keys are sorted in
alphabetic order. This is for convenience only. A writer SHOULD implement
this if possible. A reader MUST NOT depend on this.

Following this header is the output of ``numpy.save`` for the underlying
data buffer.  This contains the full output of ``save``, beginning with
the magic number for ``.npy`` files, followed by the ``.npy`` header and
array data.

The ``.npy`` format, including reasons for creating it and a comparison
of alternatives, is described fully in the "npy-format" NEP and in the
module docstring for ``numpy.lib.format``.

"""

import io
from distarray.externals import six

import numpy as np
from numpy.lib.format import write_array_header_1_0
from numpy.lib.utils import safe_eval
from numpy.compat import asbytes

from distarray.utils import _raise_nie


MAGIC_PREFIX = asbytes('\x93DARRY')
MAGIC_LEN = len(MAGIC_PREFIX) + 2


# This is only copied from numpy/lib/format.py because the numpy version
# doesn't allow one to set the MAGIC_PREFIX
def magic(major, minor, prefix=MAGIC_PREFIX):
    """Return the magic string for the given file format version.

    Parameters
    ----------
    major : int in [0, 255]
    minor : int in [0, 255]

    Returns
    -------
    magic : str

    Raises
    ------
    ValueError
        if the version cannot be formatted.

    """
    if major < 0 or major > 255:
        raise ValueError("Major version must be 0 <= major < 256.")
    if minor < 0 or minor > 255:
        raise ValueError("Minor version must be 0 <= minor < 256.")
    if six.PY2:
        return prefix + chr(major) + chr(minor)
    elif six.PY3:
        return prefix + bytes([major, minor])
    else:
        raise _raise_nie()


def write_localarray(fp, arr, version=(1, 0)):
    """
    Write a LocalArray to a .dnpy file, including a header.

    The ``__version__`` and ``dim_data`` keys from the Distributed Array
    Protocol are written to a header, then ``numpy.save`` is used to write the
    value of the ``buffer`` key.

    Parameters
    ----------
    fp : file_like object
        An open, writable file object, or similar object with a ``.write()``
        method.
    arr : LocalArray
        The array to write to disk.
    version : (int, int), optional
        The version number of the file format.  Default: (1, 0)

    Raises
    ------
    ValueError
        If the array cannot be persisted.
    Various other errors
        If the underlying numpy array contains Python objects as part of its
        dtype, the process of pickling them may raise various errors if the
        objects are not picklable.

    """
    if version != (1, 0):
        msg = "Only version (1, 0) is supported, not %s."
        raise ValueError(msg % (version,))

    fp.write(magic(*version))

    distbuffer = arr.__distarray__()
    metadata = {'__version__': distbuffer['__version__'],
                'dim_data': distbuffer['dim_data'],
                }

    write_array_header_1_0(fp, metadata)
    np.save(fp, distbuffer['buffer'])


def read_magic(fp):
    """Read the magic string to get the version of the file format.

    Parameters
    ----------
    fp : filelike object

    Returns
    -------
    major : int
    minor : int
    """
    magic_str = _read_bytes(fp, MAGIC_LEN, "magic string")
    if magic_str[:-2] != MAGIC_PREFIX:
        msg = "the magic string is not correct; expected %r, got %r"
        raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
    if six.PY2:
        major, minor = map(ord, magic_str[-2:])
    elif six.PY3:
        major, minor = magic_str[-2:]
    else:
        raise _raise_nie()
    return major, minor


def read_array_header_1_0(fp):
    """
    Read an array header from a filelike object using the 1.0 file format
    version.

    This will leave the file object located just after the header.

    Parameters
    ----------
    fp : filelike object
        A file object or something with a `.read()` method like a file.

    Returns
    -------
    __version__ : str
        Version of the Distributed Array Protocol used.
    dim_data : tuple
        A tuple containing a dictionary for each dimension of the underlying
        array, as described in the Distributed Array Protocol.

    Raises
    ------
    ValueError
        If the data is invalid.

    """
    # Read an unsigned, little-endian short int which has the length of the
    # header.
    import struct
    hlength_str = _read_bytes(fp, 2, "Array header length")
    header_length = struct.unpack('<H', hlength_str)[0]
    header = _read_bytes(fp, header_length, "Array header")

    # The header is a pretty-printed string representation of a literal Python
    # dictionary with trailing newlines padded to a 16-byte boundary. The keys
    # are strings.
    try:
        d = safe_eval(header)
    except SyntaxError as e:
        msg = "Cannot parse header: %r\nException: %r"
        raise ValueError(msg % (header, e))
    if not isinstance(d, dict):
        msg = "Header is not a dictionary: %r"
        raise ValueError(msg % d)
    keys = sorted(d.keys())
    if keys != ['__version__', 'dim_data']:
        msg = "Header does not contain the correct keys: %r"
        raise ValueError(msg % (keys,))

    # TODO: Sanity check with the DAP validator

    return d['__version__'], d['dim_data']


def read_localarray(fp):
    """
    Read a LocalArray from an .dnpy file.

    Parameters
    ----------
    fp : file_like object
        If this is not a real file object, then this may take extra memory
        and time.

    Returns
    -------
    distbuffer : dict
        The Distributed Array Protocol structure created from the data on disk.

    Raises
    ------
    ValueError
        If the data is invalid.

    """
    version = read_magic(fp)
    if version != (1, 0):
        msg = "only support version (1,0) of file format, not %r"
        raise ValueError(msg % (version,))

    __version__, dim_data = read_array_header_1_0(fp)

    buf = np.load(fp)

    distbuffer = {
        '__version__': __version__,
        'dim_data': dim_data,
        'buffer': buf,
        }

    return distbuffer


# This is only copied from numpy/lib/format.py because importing it doesn't
# work
def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on would-block,
        # python2 file will truncate, probably nothing can be done about that.
        # note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data
