# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
`Context` objects contain the information required for `DistArray`\s to
communicate with `LocalArray`\s.
"""

from __future__ import absolute_import

import atexit
import collections
import types
from abc import ABCMeta, abstractmethod

from functools import wraps

import numpy

from distarray.externals import six
from distarray.globalapi import ipython_cleanup
from distarray.globalapi.distarray import DistArray
from distarray.globalapi.maps import Distribution, asdistribution

from distarray.globalapi.ipython_utils import IPythonClient
from distarray.utils import uid, DISTARRAY_BASE_NAME, has_exactly_one
from distarray.localapi.proxyize import Proxy

# mpi context
from distarray.mpionly_utils import (make_targets_comm, get_nengines,
                                     get_world_rank, initial_comm_setup,
                                     is_solo_mpi_process, get_comm_world,
                                     mpi, push_function)


@six.add_metaclass(ABCMeta)
class BaseContext(object):

    """
    Context objects manage the setup and communication of the worker processes
    for DistArray objects.  A DistArray object has a context, and contexts have
    an MPI intracommunicator that they use to communicate with worker
    processes.

    Typically there is just one context object that uses all processes,
    although it is possible to have more than one context with a different
    selection of engines.
    """

    _CLEANUP = None

    @abstractmethod
    def __init__(self):
        raise TypeError("The base context class is not meant to be "
                        "instantiated on its own.")

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def make_subcomm(self, new_targets):
        pass

    @abstractmethod
    def apply(self, func, args=None, kwargs=None, targets=None):
        pass

    @abstractmethod
    def push_function(self, key, func):
        pass

    def _setup_context_key(self):
        """
        Create a dict on the engines which will hold everything from
        this context.
        """
        context_key = uid()
        cmd = ("import types, sys;"
               "%s = types.ModuleType('%s');")
        cmd %= (context_key, context_key)
        self._execute(cmd, self.all_targets)
        return context_key

    @staticmethod
    def _key_prefix():
        """ Get the base name for all keys. """
        return DISTARRAY_BASE_NAME

    def _generate_key(self):
        """ Generate a unique key name for this context. """
        key = "%s_%s" % (self.context_key, uid())
        return key

    def _key_and_push(self, *values, **kwargs):
        keys = [self._generate_key() for value in values]
        targets = kwargs.get('targets', self.targets)
        def _local_key_and_push(kvs):
            from distarray.localapi.proxyize import Proxy
            return [Proxy(key, val, '__main__') for key, val in kvs]
        result = self.apply(_local_key_and_push, (zip(keys, values), ), targets=targets)
        return tuple(result[0])

    def delete_key(self, key, targets=None):
        """ Delete the specific key from all the engines. """
        def _local_delete(obj):
            from distarray.localapi.proxyize import Proxy
            from importlib import import_module
            if isinstance(obj, Proxy):
                obj.cleanup()
            else:
                main = import_module('__main__')
                delattr(main, obj)
        targets = targets or self.targets
        with self.view.temp_flags(targets=targets):
            self.view.apply_sync(_local_delete, key)

    def _create_local(self, local_call, shape_or_dist, dtype):
        """Creates LocalArrays with the method named in `local_call`."""

        def create_local(local_call, ddpr, dtype, comm):
            from distarray.localapi.maps import Distribution
            if len(ddpr) == 0:
                dim_data = ()
            else:
                dim_data = ddpr[comm.Get_rank()]
            local_call = eval(local_call)
            distribution = Distribution(comm=comm, dim_data=dim_data)
            rval = local_call(distribution=distribution, dtype=dtype)
            return proxyize(rval)

        distribution = asdistribution(self, shape_or_dist)

        ddpr = distribution.get_dim_data_per_rank()
        args = [local_call, ddpr, dtype, distribution.comm]
        da_key = self.apply(create_local, args=args,
                            targets=distribution.targets)[0]
        return DistArray.from_localarrays(da_key, distribution=distribution,
                                          dtype=dtype)

    def empty(self, shape_or_dist, dtype=float):
        """Create an empty Distarray.

        Parameters
        ----------
        shape_or_dist : shape tuple or Distribution object
        dtype : NumPy dtype, optional (default float)

        Returns
        -------
        DistArray
            A DistArray distributed as specified, with uninitialized values.
        """
        return self._create_local(local_call='distarray.localapi.empty',
                                  shape_or_dist=shape_or_dist, dtype=dtype,)

    def zeros(self, shape_or_dist, dtype=float):
        """Create a Distarray filled with zeros.

        Parameters
        ----------
        shape_or_dist : shape tuple or Distribution object
        dtype : NumPy dtype, optional (default float)

        Returns
        -------
        DistArray
            A DistArray distributed as specified, filled with zeros.
        """
        return self._create_local(local_call='distarray.localapi.zeros',
                                  shape_or_dist=shape_or_dist, dtype=dtype,)

    def ones(self, shape_or_dist, dtype=float):
        """Create a Distarray filled with ones.

        Parameters
        ----------
        shape_or_dist : shape tuple or Distribution object
        dtype : NumPy dtype, optional (default float)

        Returns
        -------
        DistArray
            A DistArray distributed as specified, filled with ones.
        """
        return self._create_local(local_call='distarray.localapi.ones',
                                  shape_or_dist=shape_or_dist, dtype=dtype,)

    def allclose(self, a, b, rtol=1e-05, atol=1e-08):

        adist = a.distribution
        bdist = b.distribution

        if not adist.is_compatible(bdist):
            raise ValueError("%r and %r have incompatible distributions.")

        def local_allclose(la, lb, rtol, atol):
            from numpy import allclose
            return allclose(la.ndarray, lb.ndarray, rtol, atol)

        local_results = self.apply(local_allclose, 
                                  (a.key, b.key, rtol, atol),
                                  targets=a.targets)
        return all(local_results)

    def save_dnpy(self, name, da):
        """
        Save a distributed array to files in the ``.dnpy`` format.

        The ``.dnpy`` file format is a binary format inspired by NumPy's
        ``.npy`` format.  The header of a particular ``.dnpy`` file contains
        information about which portion of a DistArray is saved in it (using
        the metadata outlined in the Distributed Array Protocol), and the data
        portion contains the output of NumPy's `save` function for the local
        array data.  See the module docstring for `distarray.localapi.format` for
        full details.

        Parameters
        ----------
        name : str or list of str
            If a str, this is used as the prefix for the filename used by each
            engine.  Each engine will save a file named ``<name>_<rank>.dnpy``.
            If a list of str, each engine will use the name at the index
            corresponding to its rank.  An exception is raised if the length of
            this list is not the same as the context's communicator's size.
        da : DistArray
            Array to save to files.

        Raises
        ------
        TypeError
            If `name` is an sequence whose length is different from the
            context's communicator's size.

        See Also
        --------
        load_dnpy : Loading files saved with save_dnpy.
        """

        def _local_save_dnpy(local_arr, fname_base):
            from distarray.localapi import save_dnpy
            fname = "%s_%s.dnpy" % (fname_base, local_arr.comm_rank)
            save_dnpy(fname, local_arr)

        def _local_save_dnpy_names(local_arr, fnames):
            from distarray.localapi import save_dnpy
            fname = fnames[local_arr.comm_rank]
            save_dnpy(fname, local_arr)

        if isinstance(name, six.string_types):
            func = _local_save_dnpy
        elif isinstance(name, collections.Sequence):
            if len(name) != len(self.targets):
                errmsg = "`name` must be the same length as `self.targets`."
                raise TypeError(errmsg)
            func = _local_save_dnpy_names
        else:
            errmsg = "`name` must be a string or a list."
            raise TypeError(errmsg)

        self.apply(func, (da.key, name), targets=da.targets)

    def load_dnpy(self, name):
        """
        Load a distributed array from ``.dnpy`` files.

        The ``.dnpy`` file format is a binary format inspired by NumPy's
        ``.npy`` format.  The header of a particular ``.dnpy`` file contains
        information about which portion of a DistArray is saved in it (using
        the metadata outlined in the Distributed Array Protocol), and the data
        portion contains the output of NumPy's `save` function for the local
        array data.  See the module docstring for `distarray.localapi.format` for
        full details.

        Parameters
        ----------
        name : str or list of str
            If a str, this is used as the prefix for the filename used by each
            engine.  Each engine will load a file named ``<name>_<rank>.dnpy``.
            If a list of str, each engine will use the name at the index
            corresponding to its rank.  An exception is raised if the length of
            this list is not the same as the context's communicator's size.

        Returns
        -------
        result : DistArray
            A DistArray encapsulating the file loaded on each engine.

        Raises
        ------
        TypeError
            If `name` is an iterable whose length is different from the
            context's communicator's size.

        See Also
        --------
        save_dnpy : Saving files to load with with load_dnpy.
        """

        def _local_load_dnpy(comm, fname_base):
            from distarray.localapi import load_dnpy
            fname = "%s_%s.dnpy" % (fname_base, comm.Get_rank())
            local_arr = load_dnpy(comm, fname)
            return proxyize(local_arr)

        def _local_load_dnpy_names(comm, fnames):
            from distarray.localapi import load_dnpy
            fname = fnames[comm.Get_rank()]
            local_arr = load_dnpy(comm, fname)
            return proxyize(local_arr)

        if isinstance(name, six.string_types):
            func = _local_load_dnpy
        elif isinstance(name, collections.Sequence):
            if len(name) != len(self.targets):
                errmsg = "`name` must be the same length as `self.targets`."
                raise TypeError(errmsg)
            func = _local_load_dnpy_names
        else:
            errmsg = "`name` must be a string or a list."
            raise TypeError(errmsg)

        da_key = self.apply(func, (self.comm, name), targets=self.targets)
        return DistArray.from_localarrays(da_key[0], context=self)

    def save_hdf5(self, filename, da, key='buffer', mode='a'):
        """
        Save a DistArray to a dataset in an ``.hdf5`` file.

        Parameters
        ----------
        filename : str
            Name of file to write to.
        da : DistArray
            Array to save to a file.
        key : str, optional
            The identifier for the group to save the DistArray to (the default
            is 'buffer').
        mode : optional, {'w', 'w-', 'a'}, default 'a'

            ``'w'``
                Create file, truncate if exists
            ``'w-'``
                Create file, fail if exists
            ``'a'``
                Read/write if exists, create otherwise (default)
        """
        try:
            # this is just an early check,
            # h5py isn't necessary until the local call on the engines
            import h5py
        except ImportError:
            errmsg = "An MPI-enabled h5py must be available to use save_hdf5."
            raise ImportError(errmsg)

        def _local_save_dnpy(filename, local_arr, key, mode):
            from distarray.localapi import save_hdf5
            save_hdf5(filename, local_arr, key, mode)

        self.apply(_local_save_dnpy, (filename, da.key, key, mode),
                   targets=da.targets)

    def load_npy(self, filename, distribution):
        """
        Load a DistArray from a dataset in a ``.npy`` file.

        Parameters
        ----------
        filename : str
            Filename to load.
        distribution: Distribution object

        Returns
        -------
        result : DistArray
            A DistArray encapsulating the file loaded.
        """

        def _local_load_npy(filename, ddpr, comm):
            from distarray.localapi import load_npy
            if len(ddpr):
                dim_data = ddpr[comm.Get_rank()]
            else:
                dim_data = ()
            return proxyize(load_npy(comm, filename, dim_data))

        ddpr = distribution.get_dim_data_per_rank()

        da_key = self.apply(_local_load_npy, (filename, ddpr, distribution.comm),
                            targets=distribution.targets)
        return DistArray.from_localarrays(da_key[0], distribution=distribution)

    def load_hdf5(self, filename, distribution, key='buffer'):
        """
        Load a DistArray from a dataset in an ``.hdf5`` file.

        Parameters
        ----------
        filename : str
            Filename to load.
        distribution: Distribution object
        key : str, optional
            The identifier for the group to load the DistArray from (the
            default is 'buffer').

        Returns
        -------
        result : DistArray
            A DistArray encapsulating the file loaded.
        """
        try:
            import h5py
        except ImportError:
            errmsg = "An MPI-enabled h5py must be available to use load_hdf5."
            raise ImportError(errmsg)

        def _local_load_hdf5(filename, ddpr, comm, key):
            from distarray.localapi import load_hdf5
            if len(ddpr):
                dim_data = ddpr[comm.Get_rank()]
            else:
                dim_data = ()
            return proxyize(load_hdf5(comm, filename, dim_data, key))

        ddpr = distribution.get_dim_data_per_rank()

        da_key = self.apply(_local_load_hdf5, (filename, ddpr, distribution.comm, key),
                   targets=distribution.targets)

        return DistArray.from_localarrays(da_key[0], distribution=distribution)

    def fromndarray(self, arr, distribution=None):
        """Create a DistArray from an ndarray.

        Parameters
        ----------
        distribution : Distribution object, optional
            If a Distribution object is not provided, one is created with
            `Distribution(arr.shape)`.

        Returns
        -------
        DistArray
            A DistArray distributed as specified, using the values and dtype
            from `arr`.
        """
        if distribution is None:
            distribution = Distribution(self, arr.shape)
        out = self.empty(distribution, dtype=arr.dtype)
        try:
            out[...] = arr
        except AttributeError:
            # no slicing for a given map type; do it the slow way
            for index, value in numpy.ndenumerate(arr):
                out[index] = value
        return out

    fromarray = fromndarray

    def fromfunction(self, function, shape, **kwargs):
        """Create a DistArray from a function over global indices.

        Unlike numpy's `fromfunction`, the result of distarray's
        `fromfunction` is restricted to the same Distribution as the
        index array generated from `shape`.

        See numpy.fromfunction for more details.
        """

        self.push_function(function.__name__, function, targets=self.targets)

        def _local_fromfunction(func_name, comm, ddpr, kwargs):
            from distarray.localapi import fromfunction
            from distarray.localapi.maps import Distribution
            from importlib import import_module

            main = import_module('__main__')

            if len(ddpr):
                dim_data = ddpr[comm.Get_rank()]
            else:
                dim_data = ()
            func = getattr(main, func_name)
            dist = Distribution(comm, dim_data=dim_data)
            local_arr = fromfunction(func, dist, **kwargs)
            return proxyize(local_arr)

        dist = kwargs.get('dist', None)
        grid_shape = kwargs.get('grid_shape', None)
        distribution = Distribution(context=self,
                                    shape=shape, dist=dist,
                                    grid_shape=grid_shape)
        ddpr = distribution.get_dim_data_per_rank()
        da_name = self.apply(_local_fromfunction,
                             (function.__name__, distribution.comm, ddpr, kwargs),
                             targets=distribution.targets)
        return DistArray.from_localarrays(da_name[0],
                                          distribution=distribution)

    def register(self, func):
        """Associate a function with this Context.  Allows access to the local
        process and local data associated with each DistArray.

        After registering a function with a context, the function can be called
        as ``context.func(...)``.  Doing so will call the function locally on
        target processes determined from the arguments passed in using
        ``Context.apply(...)``.  The function can take non-proxied Python
        objects, DistArrays, or other proxied objects as arguments.
        Non-proxied Python objects will be broadcasted to all local processes;
        proxied objects will be dereferenced before calling the function on the
        local process.
        """

        if func.__name__ in ("""__init__ cleanup close apply push_function
                             delete_key empty zeros ones save_dnpy load_dnpy
                             save_hdf5 load_npy load_hdf5 fromndarray
                             fromarray fromfunction register""".split()):
            msg = "Function name %s clashes with existing function."
            raise ValueError(msg % func.__name__)
        if func.__name__.startswith("_"):
            msg = "Function name %r starts with underscore."
            raise ValueError(msg % func.__name__)

        self.push_function(func.__name__, func, targets=self.targets)

        @wraps(func)
        def _wrapper(ctx, *args, **kwargs):
            dist = ctx._determine_distribution(list(args) +
                                               list(kwargs.values()))
            targets = dist.targets
            args = [a.key if isinstance(a, DistArray) else a
                    for a in args]
            kwargs = {k: (v.key if isinstance(v, DistArray) else v)
                      for k, v in kwargs.items()}
            results = ctx.apply(func, args=args,
                                kwargs=kwargs, targets=targets,
                                autoproxyize=True)
            return ctx._process_local_results(results, targets)

        setattr(self.__class__, _wrapper.__name__, _wrapper)

    def _determine_distribution(self, objs):
        dists = []
        for o in objs:
            if isinstance(o, DistArray):
                dists.append(o.distribution)
        for d in dists[1:]:
            if not dists[0].is_compatible(d):
                msg = "Distrbution %r is not compatible with %r"
                raise ValueError(msg % (dists[0], d))
        if not dists:
            raise TypeError("Cannot determine a Distribution.")
        return dists[0]

    def _process_local_results(self, results, targets):
        """Figure out what to return on the Client.

        Parameters
        ----------
        key : string
            Key corresponding to wrapped function's return value.

        Returns
        -------
        Varied
            A DistArray (if locally all values are LocalArray), a None (if
            locally all values are None), or else, pull the result back to the
            client and return it.  If all but one of the pulled values is None,
            return that non-None value only.
        """
        def is_NoneType(pxy):
            return pxy.type_str == str(type(None))

        def is_LocalArray(pxy):
            return (isinstance(pxy, Proxy) and 
                    pxy.type_str == "<class 'distarray.localapi.localarray.LocalArray'>")

        if all(is_LocalArray(r) for r in results):
            result = DistArray.from_localarrays(results[0], context=self, targets=targets)
        elif all(r is None for r in results):
            result = None
        else:
            if has_exactly_one(results):
                result = next(x for x in results if x is not None)
            else:
                result = results

        return result


class IPythonContext(BaseContext):

    """
    Context class that uses IPython.parallel.

    See the docstring for  `BaseContext` for more information about Contexts.

    See also
    --------
    BaseContext
    """

    def __init__(self, client=None, targets=None):

        if client is None:
            self.client = IPythonClient()
            self.owns_client = True
        else:
            self.client = client
            self.owns_client = False

        self.view = self.client[:]
        self.nengines = len(self.view)

        self.all_targets = sorted(self.view.targets)
        if targets is None:
            self.targets = self.all_targets
        else:
            self.targets = []
            for target in targets:
                if target not in self.all_targets:
                    raise ValueError("Target %r not registered" % target)
                else:
                    self.targets.append(target)
        self.targets = sorted(self.targets)

        # local imports
        self.view.execute("from functools import reduce; "
                          "from importlib import import_module; "
                          "import distarray.localapi; "
                          "import distarray.localapi.mpiutils; "
                          "import distarray.utils; "
                          "import distarray.localapi.proxyize as proxyize; "
                          "from distarray.localapi.proxyize import Proxy; "
                          "import numpy")

        self.context_key = self._setup_context_key()

        # setup proxyize which is used by context.apply in the rest of the
        # setup.
        cmd = "proxyize = proxyize.Proxyize()"
        self.view.execute(cmd)

        self._base_comm = self._make_base_comm()
        self._comm_from_targets = {tuple(sorted(self.view.targets)): self._base_comm}
        self.comm = self.make_subcomm(self.targets)

        if not BaseContext._CLEANUP:
            BaseContext._CLEANUP = (atexit.register(ipython_cleanup.clear_all),
                                    atexit.register(ipython_cleanup.cleanup_all,
                                                    '__main__',
                                                    DISTARRAY_BASE_NAME))

    def make_subcomm(self, new_targets):

        if new_targets != sorted(new_targets):
            raise ValueError("targets must be in sorted order.")

        try:
            return self._comm_from_targets[tuple(new_targets)]
        except KeyError:
            pass

        def _make_new_comm(rank_list, base_comm):
            import distarray.localapi.mpiutils as mpiutils
            res = mpiutils.create_comm_with_list(rank_list, base_comm)
            return proxyize(res)  # noqa

        new_comm = self.apply(_make_new_comm, (new_targets, self._base_comm),
                              targets=self.view.targets)[0]

        self._comm_from_targets[tuple(new_targets)] = new_comm
        return new_comm

    def _make_base_comm(self):
        """
        Returns a proxy for an MPI communicator that encompasses all targets in
        self.view.targets (not self.targets, which can be a subset).
        """

        def get_rank():
            from distarray.localapi.mpiutils import get_comm_private
            return get_comm_private().Get_rank()

        # self.view's engines must encompass all ranks in the MPI communicator,
        # i.e., everything in rank_map.values().
        def get_size():
            from distarray.localapi.mpiutils import get_comm_private
            return get_comm_private().Get_size()

        # get a mapping of IPython engine ID to MPI rank
        rank_from_target = self.view.apply_async(get_rank).get_dict()
        ranks = [ rank_from_target[target] for target in self.view.targets ]

        comm_size = self.view.apply_async(get_size).get()[0]
        if set(rank_from_target.values()) != set(range(comm_size)):
            raise ValueError('Engines in view must encompass all MPI ranks.')

        # create a new communicator with the subset of ranks. Note that
        # create_comm_with_list() must be called on all engines, not just those
        # involved in the new communicator.  This is because
        # create_comm_with_list() issues a collective MPI operation.
        def _make_new_comm(rank_list):
            import distarray.localapi.mpiutils as mpiutils
            new_comm = mpiutils.create_comm_with_list(rank_list)
            return proxyize(new_comm)  # noqa

        return self.apply(_make_new_comm, args=(ranks,),
                          targets=self.view.targets)[0]

    # Key management routines:
    def cleanup(self):
        """ Delete keys that this context created from all the engines. """
        # TODO: FIXME: cleanup needs updating to work with proxy objects.
        ipython_cleanup.cleanup(view=self.view, module_name='__main__',
                                prefix=self.context_key)

    def close(self):
        self.cleanup()
        def free_subcomm(subcomm):
            subcomm.Free()
        for targets, subcomm in self._comm_from_targets.items():
            self.apply(free_subcomm, (subcomm,), targets=targets)
        if self.owns_client:
            self.client.close()
        self.comm = None

    # End of key management routines.

    def _execute(self, lines, targets):
        return self.view.execute(lines, targets=targets, block=True)

    def _push(self, d, targets):
        return self.view.push(d, targets=targets, block=True)

    def apply(self, func, args=None, kwargs=None, targets=None, autoproxyize=False):
        """
        Analogous to IPython.parallel.view.apply_sync

        Parameters
        ----------
        func : function
        args : tuple
            positional arguments to func
        kwargs : dict
            key word arguments to func
        targets : sequence of integers
            engines func is to be run on.
        autoproxyize: bool, default False
            If True, implicitly return a Proxy object from the function.

        Returns
        -------
            return a list of the results on the each engine.
        """

        def func_wrapper(func, apply_nonce, context_key, args, kwargs, autoproxyize):
            """
            Function which calls the applied function after grabbing all the
            arguments on the engines that are passed in as names of the form
            `__distarray__<some uuid>`.
            """
            from importlib import import_module
            import types
            from distarray.localapi import LocalArray

            main = import_module('__main__')
            main.proxyize.set_state(apply_nonce)

            # Modify func to change the namespace it executes in.
            # but builtins don't have __code__, __globals__, etc.
            if not isinstance(func, types.BuiltinFunctionType):
                # get func's building  blocks first
                func_code = func.__code__
                func_name = func.__name__
                func_defaults = func.__defaults__
                func_closure = func.__closure__

                # build the func's new execution environment
                main.__dict__.update({'context_key': context_key})
                new_func_globals = main.__dict__
                # create the new func
                func = types.FunctionType(func_code, new_func_globals,
                                          func_name, func_defaults,
                                          func_closure)
            # convert args
            args = list(args)
            for i, a in enumerate(args):
                if isinstance(a, main.Proxy):
                    args[i] = a.dereference()
            args = tuple(args)

            # convert kwargs
            for k in kwargs.keys():
                val = kwargs[k]
                if isinstance(val, main.Proxy):
                    kwargs[k] = val.dereference()

            result = func(*args, **kwargs)

            if autoproxyize and isinstance(result, LocalArray):
                return main.proxyize(result)
            else:
                return result

        # default arguments
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        apply_nonce = uid()[13:]
        wrapped_args = (func, apply_nonce, self.context_key, args, kwargs, autoproxyize)

        targets = self.targets if targets is None else targets

        with self.view.temp_flags(targets=targets):
            return self.view.apply_sync(func_wrapper, *wrapped_args)

    def push_function(self, key, func, targets=None):
        targets = targets or self.targets
        self._push({key: func}, targets=targets)


def _shutdown(mpi, intercomm, targets):
    msg = ('kill',)
    for t in targets:
        intercomm.send(msg, dest=t)
    intercomm.Free()
    mpi.Finalize()

class MPIContext(BaseContext):

    """
    Context class that uses MPI only (no IPython.parallel).

    See the docstring for  `BaseContext` for more information about Contexts.

    See also
    --------
    BaseContext
    """

    INTERCOMM = None

    def delete_key(self, key, targets=None):
        msg = ('delete', key)
        targets = targets or self.targets
        if MPIContext.INTERCOMM:
            self._send_msg(msg, targets=targets)

    def __init__(self, targets=None):

        if MPIContext.INTERCOMM is None:
            MPIContext.INTERCOMM = initial_comm_setup()
            assert get_world_rank() == 0

        self.nengines = get_nengines()

        self.all_targets = list(range(self.nengines))
        self.targets = self.all_targets if targets is None else sorted(targets)

        # make/get comms
        # this is the object we want to use with push, pull, etc'
        self._comm_from_targets = {}
        self.comm = self.make_subcomm(self.targets)

        if BaseContext._CLEANUP is None:
            BaseContext._CLEANUP = atexit.register(_shutdown,
                                               mpi,
                                               MPIContext.INTERCOMM,
                                               tuple(self.all_targets))

        # local imports
        self._execute("from functools import reduce; "
                          "from importlib import import_module; "
                          "import distarray.localapi; "
                          "import distarray.localapi.mpiutils; "
                          "import distarray.utils; "
                          "from distarray.localapi.proxyize import Proxyize, Proxy; "
                          "import numpy")

        self.context_key = self._setup_context_key()

        # setup proxyize which is used by context.apply in the rest of the
        # setup.
        cmd = "proxyize = Proxyize()"
        self._execute(cmd)

    # Key management routines:

    def cleanup(self):
        """ Delete keys that this context created from all the engines. """
        # TODO: implement cleanup.
        pass

    def close(self):
        for targets, subcomm in self._comm_from_targets.items():
            if subcomm is MPIContext.INTERCOMM:
                continue
            self._send_msg(('free_comm', subcomm), targets=targets)

    # End of key management routines.

    def _send_msg(self, msg, targets=None):
        targets = self.targets if targets is None else targets
        for t in targets:
            MPIContext.INTERCOMM.send(msg, dest=t)

    def _recv_msg(self, targets=None):
        res = []
        targets = self.targets if targets is None else targets
        for t in targets:
            res.append(MPIContext.INTERCOMM.recv(source=t))
        return res

    def make_subcomm(self, targets):
        if len(targets) > self.nengines:
            msg = ("The number of engines (%s) is less than the number of "
                   "targets you want (%s)." % (self.nengines, len(targets)))
            raise ValueError(msg)

        if targets != sorted(targets):
            raise ValueError("targets must be in sorted order.")

        try:
            return self._comm_from_targets[tuple(targets)]
        except KeyError:
            pass

        msg = ('make_targets_comm', targets)
        self._send_msg(msg, targets=self.all_targets)
        new_comm = make_targets_comm(targets)
        self._comm_from_targets[tuple(targets)] = new_comm
        return new_comm

    def _execute(self, lines, targets=None):
        msg = ('execute', lines)
        return self._send_msg(msg, targets=targets)

    def _push(self, d, targets=None):
        msg = ('push', d)
        return self._send_msg(msg, targets=targets)

    def apply(self, func, args=None, kwargs=None, targets=None, autoproxyize=False):
        """
        Analogous to IPython.parallel.view.apply_sync

        Parameters
        ----------
        func : function
        args : tuple
            positional arguments to func
        kwargs : dict
            keyword arguments to func
        targets : sequence of integers
            engines func is to be run on.
        autoproxyize: bool, default False
            If True, implicitly return a Proxy object from the function.

        Returns
        -------
        list
            result from each engine.
        """
        # default arguments
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        targets = self.targets if targets is None else targets

        apply_nonce = uid()[13:]
        apply_metadata = (apply_nonce, self.context_key)

        if not isinstance(func, types.BuiltinFunctionType):
            # break up the function
            func_code = func.__code__
            func_name = func.__name__
            func_defaults = func.__defaults__
            func_closure = func.__closure__

            func_data = (func_code, func_name, func_defaults, func_closure)

            msg = ('func_call', func_data, args, kwargs, apply_metadata, autoproxyize)

        else:
            msg = ('builtin_call', func, args, kwargs, autoproxyize)

        self._send_msg(msg, targets=targets)
        return self._recv_msg(targets=targets)

    def push_function(self, key, func, targets=None):
        push_function(self, key, func, targets=targets)


class ContextCreationError(RuntimeError):
    pass


def _fire_off_engines(rank):
    if rank:
        from distarray.mpi_engine import Engine
        Engine()

def Context(*args, **kwargs):

    kind = kwargs.pop('kind', '')

    if not kind:
        kind = 'ipython' if is_solo_mpi_process() else 'mpi'

    if kind.lower().startswith('mpi'):

        CW = get_comm_world()
        myrank = CW.rank
        if myrank:
            _fire_off_engines(myrank)
            import sys
            sys.exit()
        else:
            return MPIContext(*args, **kwargs)

    elif kind.lower().startswith('ipython'):
        return IPythonContext(*args, **kwargs)

    else:
        raise ContextCreationError("%s is not a valid Context selector string." % kind)
