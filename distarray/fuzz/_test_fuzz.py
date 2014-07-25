#!/usr/bin/env python

import distarray.dist as da
from distarray.fuzz.make_rand_distarray import rand_distarray
from distarray.fuzz.fuzzy_test import fuzzy_test

c = da.Context()
for _ in range(2):
    fuzzy_test(rand_distarray(c))
