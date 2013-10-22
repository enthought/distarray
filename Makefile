.PHONY: clean setup_cluster test test_travis teardown_cluster

PYTHON = python

develop:
	${PYTHON} setup.py develop

install:
	${PYTHON} setup.py install

setup_cluster:
	-${PYTHON} distarray/tests/run_ipcluster.py
	-sleep 15  # wait for ipcluster

test:
	(nosetests)
	(cd distarray/core/tests && mpiexec -n 12 nosetests -i 'paralleltest_\w+')
	(cd distarray/random/tests && mpiexec -n 4 nosetests -i 'paralleltest_\w+')

teardown_cluster:
	-kill $(shell ps -ax | grep 'ipcluster start' | grep -v 'grep' | awk '{ print $$1; }' )

clean:
	${PYTHON} setup.py clean --all
	-${RM} `find . -name '*.py[co]'`
	-${RM} `find . -name '*.so'`
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist distarray.egg-info
