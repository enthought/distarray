# encoding: utf-8

__docformat__ = "restructuredtext en"


import pickle


#MAGIC_PREFIX = asbytes('\x93DARRY')
#MAGIC_LEN = len(MAGIC_PREFIX) + 2


def write_localarray(fh, arr, version=(1, 0)):
    pickle.dump(arr.__distarray__(), fh)


def read_localarray(fh):
    return pickle.load(fh)
