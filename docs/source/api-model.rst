Module level API:
-----------------

::

    distarray
    ├── Common Math functions: sin, dot, etc.
    ├── Distarray creation functions: empty, ones, etc.
    ├── decorators: local, vector
    ├── world: the default context used default
    |   └── set_context: used for specifying a global context
    └── Random
        └── Random Array creation functions: randn, normal, etc.
                These should take an optional Context as a kwarg, by
                default use distarray.world.

DistArray object API:
---------------------

::

    DistArray
    ├── DistArray objects are created with the global context unless
    |       otherwise noted.
    ├── Common Math functions: sin, dot, etc.
    ├── global array metadata should have no prefix. i.e. shape is the
    |       global shape.
    ├── local array metadata is prefixed by "local_" and return an
    |       array of local metadata. i.e local_shapes returns an array
    |       where each element is the shape on the corresponding engine.
    └── maps - provide an inteface for mapping indicies between global
            and local arrays. Bob could flush this out better.

LocalArray object API:
----------------------

::

    LocalArray
    ├── Common Math functions: sin, dot, etc.
    ├── local array metadata has no prefix. i.e. shape is the local
    |       shape. The exception is local_array, which is a numpy array
    |       of the local data.
    ├── global array metadata should have a "global_" prefix. i.e.
    |       shape is the global shape.
    ├── global indexing can be done by indexing LocalArray.global.
    └── indexing - indexing a LocalArray object should index the local
            array with local indices.

Context object API:
-------------------

::

    Context
    └── indexing should return another context limited to the scope of
            the engines which were indexed.
