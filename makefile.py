import os, sys

@target
def cython(*args, **kwargs):
    config = kwargs.get('config')
    files = all_files('ipythondistarray', '*.pyx')
    for full_file in files:
        (a, b) = os.path.split(full_file)
        xsys('cd %s && cython %s' % (a, b))

@target
def clean(*args, **kwargs):
    files = all_files('ipythondistarray', '*.so;*.pyc')
    for f in files:
        xsys('rm %s' % f)
    if os.path.isdir('build'):
        xsys('rm -rf build')

@target
def inplace(*args, **kwargs):
    xsys('python setup.py build_ext --inplace')
