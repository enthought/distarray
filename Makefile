.PHONY: clean setup_cluster test test_travis teardown_cluster

PYTHON = python
MPIEXEC = mpiexec

develop:
	${PYTHON} setup.py develop

install:
	${PYTHON} setup.py install

setup_cluster:
	-${PYTHON} distarray/tests/run_ipcluster.py
	-sleep 15  # wait for ipcluster

test:
	mpiexec -n 12 nosetests -w distarray/core/tests -i 'paralleltest'
	mpiexec -n 4 nosetests -w distarray/random/tests -i 'paralleltest'
	nosetests

teardown_cluster:
	-kill $(shell ps -ax | grep 'ipcluster start' | grep -v 'grep' | awk '{ print $$1; }' )

clean:
	${PYTHON} setup.py clean --all
	-${RM} `find . -name '*.py[co]'`
	-${RM} `find . -name '*.so'`
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist distarray.egg-info
