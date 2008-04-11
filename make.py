#!/usr/bin/env python

import re
import sys, os, fnmatch
from optparse import OptionParser

def all_files(root, patterns='*', single_level=False, yield_folders=False):
    patterns = patterns.split(';')
    for path, subdirs, files in os.walk(root):
        if yield_folders:
            files.extend(subdirs)
        files.sort()
        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    yield os.path.join(path, name)
                    break
        if single_level:
            break

def xsys(cmd):
    os.system(cmd)



class Maker(object):
    
    def __init__(self, filename):
        self.filename = filename
        self.targets = {}
        self.config = {}
        self.namespace = {}
        self._setup()
        # self._debug()
        
    def _setup(self):
        self._seed_namespace()
        self._loadfile()

    def _seed_namespace(self):
        self.namespace['target'] = self.target
        self.namespace['all_files'] = all_files
        self.namespace['xsys'] = xsys
        self.namespace['x'] = xsys
        
    def _debug(self):
        for k, v in self.targets.iteritems():
            print k, v
        
    def _loadfile(self):
        execfile(self.filename, globals(), self.namespace)
        
    def target(self, t):
        print '[make.py] Registering target: ', t.__name__
        self.targets[t.__name__] = t
        self.config[t.__name__] = {}
        return t

    def build(self, target_name):
        print '[make.py] Building target: ', target_name
        t = self.targets.get(target_name)
        if t is not None:
            t(self.config)
        else:
            print "[make.py] Error: target %s not found" % target_name

    def run(self):
        parser = OptionParser()
        (options, args) = parser.parse_args()
        for a in args:
            self.build(a)
            
if __name__ == '__main__':
    cwd = os.getcwd()
    try_file = os.path.join(cwd, 'makefile.py')
    if os.path.isfile(try_file):
        m = Maker(try_file)
        m.run()
    else:
        print "[make.py] Error: no makefile.ipy could be found in the current working dir."