from distarray.context import Context
from distarray.utils import is_local

if not is_local():
    WORLD = Context()
else:
    WORLD = None
