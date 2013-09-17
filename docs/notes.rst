Redistribution of Data
======================


2. Walk over local indices of local_array
3. Using old_dist compute global_inds
4. Using new_dist compute new_owner, append to array for that owner
5. Everyone now knows where data needs to be sent

Next Steps
----------

    * Begin to work on operator overloading.
    * Begin to work on other simple functions and methods.
    * Write example scripts.

Here is an example of how this might worK::

for i in rows:
    for j in cols:
        a.prefetch(i,j)
a.sync()

for p in stuff:
    for q in things:
        b[p,q] = a[p,q]
b.sync()


Design Notes
------------

Currently, in our client, we end up creating lots of temp variable names for
things that get pushed to or created on the engines.  I am now using uuid's
for these so they are unique.  However, we are not deleting these temp
variables, so they will pollute the namespace on the engine.  What to do?

It would be nice to be able to pull expressions, such as 'foo.bar' or
'foo.bar()' to make it easier to create proxy objects.

When proxying, there is a difference between methods that get called on all
engine and those that just get called on one.  These correspond to different
types of proxies.

When you take an ndarray, scatter it to the engines, take the fft and then
gather the result back, the first block is the last, so the result is
not right.  Hmmmmm.

We need a real and imag attribute.

fromfunction works differently from numpy.fromfunction
