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
	${PYTHON} -m unittest discover
	${MPIEXEC} -n 12 ${PYTHON} -m unittest discover -s distarray/core/tests -p 'paralleltest*.py' 
	${MPIEXEC} -n 4 ${PYTHON} -m unittest discover -s distarray/random/tests -p 'paralleltest*.py' 

teardown_cluster:
	-kill $(shell ps -ax | grep 'ipcluster start' | grep -v 'grep' | awk '{ print $$1; }' )

clean:
	${PYTHON} setup.py clean --all
	-${RM} `find . -name '*.py[co]'`
	-${RM} `find . -name '*.so'`
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist distarray.egg-info
