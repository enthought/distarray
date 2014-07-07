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

import collections
import atexit

import numpy

import distarray
from distarray.dist import cleanup
from distarray.externals import six
from distarray.dist.distarray import DistArray
from distarray.dist.maps import Distribution

from distarray.dist.ipython_utils import IPythonClient
from distarray.utils import uid, DISTARRAY_BASE_NAME


class Context(object):

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

    def __init__(self, client=None, targets=None):

        if not Context._CLEANUP:
            Context._CLEANUP = (atexit.register(cleanup.clear_all),
                                atexit.register(cleanup.cleanup_all, '__main__', DISTARRAY_BASE_NAME))

        if client is None:
            self.client = IPythonClient()
            self.owns_client = True
        else:
            self.client = client
            self.owns_client = False

        self.view = self.client[:]

        all_targets = sorted(self.view.targets)
        if targets is None:
            self.targets = all_targets
        else:
            self.targets = []
            for target in targets:
                if target not in all_targets:
                    raise ValueError("Engine with id %r not registered" % target)
                else:
                    self.targets.append(target)
        self.targets = sorted(self.targets)

        # local imports
        self.view.execute("from functools import reduce; "
                          "from importlib import import_module; "
                          "import distarray.local; "
                          "import distarray.local.mpiutils; "
                          "import distarray.utils; "
                          "import distarray.local.proxyize as proxyize; "
                          "import numpy")

        self.context_key = self._setup_context_key()

        # setup proxyize which is used by context.apply in the rest of the
        # setup.
        cmd = "proxyize = proxyize.Proxyize('%s')" % (self.context_key,)
        self.view.execute(cmd)

        self._base_comm = self._make_base_comm()
        self._comm_from_targets = {tuple(sorted(self.view.targets)): self._base_comm}  # noqa
        self.comm = self._make_subcomm(self.targets)

    def _setup_context_key(self):
        """
        Create a dict on the engines which will hold everything from
        this context.
        """
        context_key = uid()
        cmd = ("import types, sys;"
               "%s = types.ModuleType('%s');")
        cmd %= (context_key, context_key)
        self._execute(cmd, targets=range(len(self.view)))
        return context_key

    def _make_subcomm(self, new_targets):

        if new_targets != sorted(new_targets):
            raise ValueError("targets must be in sorted order.")

        try:
            return self._comm_from_targets[tuple(new_targets)]
        except KeyError:
            pass

        def _make_new_comm(rank_list, base_comm):
            import distarray.local.mpiutils as mpiutils
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
            from distarray.local.mpiutils import COMM_PRIVATE
            return COMM_PRIVATE.Get_rank()

        # self.view's engines must encompass all ranks in the MPI communicator,
        # i.e., everything in rank_map.values().
        def get_size():
            from distarray.local.mpiutils import COMM_PRIVATE
            return COMM_PRIVATE.Get_size()

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
            import distarray.local.mpiutils as mpiutils
            new_comm = mpiutils.create_comm_with_list(rank_list)
            return proxyize(new_comm)  # noqa

        return self.apply(_make_new_comm, args=(ranks,),
                          targets=self.view.targets)[0]

    # Key management routines:
    @staticmethod
    def _key_prefix():
        """ Get the base name for all keys. """
        return DISTARRAY_BASE_NAME

    def _generate_key(self):
        """ Generate a unique key name for this context. """
        key = "%s.%s" % (self.context_key, uid())
        return key

    def _key_and_push(self, *values, **kwargs):
        keys = [self._generate_key() for value in values]
        targets = kwargs.get('targets', self.targets)
        self._push(dict(zip(keys, values)), targets=targets)
        return tuple(keys)

    def delete_key(self, key, targets=None):
        """ Delete the specific key from all the engines. """
        cmd = ('try: del %s\n'
               'except NameError: pass') % key
        targets = targets or self.targets
        self._execute(cmd, targets=targets)

    def cleanup(self):
        """ Delete keys that this context created from all the engines. """
        cleanup.cleanup(view=self.view, module_name='__main__', prefix=self.context_key)

    def close(self):
        self.cleanup()
        if self.owns_client:
            self.client.close()
        self._base_comm = None
        self.comm = None

    # End of key management routines.

    def _execute(self, lines, targets):
        return self.view.execute(lines, targets=targets, block=True)

    def _push(self, d, targets):
        return self.view.push(d, targets=targets, block=True)

    def _pull(self, k, targets):
        return self.view.pull(k, targets=targets, block=True)

    def _create_local(self, local_call, distribution, dtype):
        """Creates LocalArrays with the method named in `local_call`."""
        def create_local(local_call, ddpr, dtype, comm):
            from distarray.local.maps import Distribution
            if len(ddpr) == 0:
                dim_data = ()
            else:
                dim_data = ddpr[comm.Get_rank()]
            local_call = eval(local_call)
            distribution = Distribution(comm=comm, dim_data=dim_data)
            rval = local_call(distribution=distribution, dtype=dtype)
            return proxyize(rval)

        ddpr = distribution.get_dim_data_per_rank()
        args = [local_call, ddpr, dtype, distribution.comm]
        da_key = self.apply(create_local, args=args,
                            targets=distribution.targets)[0]
        return DistArray.from_localarrays(da_key, distribution=distribution,
                                          dtype=dtype)

    def empty(self, distribution, dtype=float):
        """Create an empty Distarray.

        Parameters
        ----------
        distribution : Distribution object
        dtype : NumPy dtype, optional (default float)

        Returns
        -------
        DistArray
            A DistArray distributed as specified, with uninitialized values.
        """
        return self._create_local(local_call='distarray.local.empty',
                                  distribution=distribution, dtype=dtype)

    def zeros(self, distribution, dtype=float):
        """Create a Distarray filled with zeros.

        Parameters
        ----------
        distribution : Distribution object
        dtype : NumPy dtype, optional (default float)

        Returns
        -------
        DistArray
            A DistArray distributed as specified, filled with zeros.
        """
        return self._create_local(local_call='distarray.local.zeros',
                                  distribution=distribution, dtype=dtype)

    def ones(self, distribution, dtype=float):
        """Create a Distarray filled with ones.

        Parameters
        ----------
        distribution : Distribution object
        dtype : NumPy dtype, optional (default float)

        Returns
        -------
        DistArray
            A DistArray distributed as specified, filled with ones.
        """
        return self._create_local(local_call='distarray.local.ones',
                                  distribution=distribution, dtype=dtype,)

    def save_dnpy(self, name, da):
        """
        Save a distributed array to files in the ``.dnpy`` format.

        The ``.dnpy`` file format is a binary format inspired by NumPy's
        ``.npy`` format.  The header of a particular ``.dnpy`` file contains
        information about which portion of a DistArray is saved in it (using
        the metadata outlined in the Distributed Array Protocol), and the data
        portion contains the output of NumPy's `save` function for the local
        array data.  See the module docstring for `distarray.local.format` for
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
            from distarray.local import save_dnpy
            fname = "%s_%s.dnpy" % (fname_base, local_arr.comm_rank)
            save_dnpy(fname, local_arr)

        def _local_save_dnpy_names(local_arr, fnames):
            from distarray.local import save_dnpy
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
        array data.  See the module docstring for `distarray.local.format` for
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
            from distarray.local import load_dnpy
            fname = "%s_%s.dnpy" % (fname_base, comm.Get_rank())
            local_arr = load_dnpy(comm, fname)
            return proxyize(local_arr)

        def _local_load_dnpy_names(comm, fnames):
            from distarray.local import load_dnpy
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
            from distarray.local import save_hdf5
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
            from distarray.local import load_npy
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
            from distarray.local import load_hdf5
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

        def _local_fromfunction(func, comm, ddpr, kwargs):
            from distarray.local import fromfunction
            from distarray.local.maps import Distribution
            if len(ddpr):
                dim_data = ddpr[comm.Get_rank()]
            else:
                dim_data = ()
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
                             (function, distribution.comm, ddpr, kwargs),
                             targets=distribution.targets)
        return DistArray.from_localarrays(da_name[0], distribution=distribution)

    def apply(self, func, args=None, kwargs=None, targets=None):
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

        Returns
        -------
            return a list of the results on the each engine.
        """

        def func_wrapper(func, apply_nonce, context_key, args, kwargs):
            """
            Function which calls the applied function after grabbing all the
            arguments on the engines that are passed in as names of the form
            `__distarray__<some uuid>`.
            """
            from importlib import import_module
            import types

            main = import_module('__main__')
            prefix = main.distarray.utils.DISTARRAY_BASE_NAME
            main.proxyize.set_state(apply_nonce)

            # Modify func to change the namespace it executes in.
            # but builtins don't have __code__, __globals__, etc.
            if not isinstance(func, types.BuiltinFunctionType):
                # get func's building  blocks first
                func_code = func.__code__
                func_globals = func.__globals__  # noqa we don't need these.
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
                if (isinstance(a, str) and a.startswith(prefix)):
                    args[i] = main.reduce(getattr, [main] + a.split('.'))
            args = tuple(args)

            # convert kwargs
            for k in kwargs.keys():
                val = kwargs[k]
                if (isinstance(val, str) and val.startswith(prefix)):
                    kwargs[k] = main.reduce(getattr, [main] + val.split('.'))

            return func(*args, **kwargs)

        # default arguments
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        apply_nonce = uid()[13:]
        wrapped_args = (func, apply_nonce, self.context_key, args, kwargs)

        targets = self.targets if targets is None else targets

        return self.view._really_apply(func_wrapper, args=wrapped_args,
                                       targets=targets, block=True)
