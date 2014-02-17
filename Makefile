.PHONY: clean setup_cluster test test_travis teardown_cluster

PYTHON = python
MPIEXEC = mpiexec
COVERAGE = coverage

develop:
	${PYTHON} setup.py develop

install:
	${PYTHON} setup.py install

setup_cluster:
	-${PYTHON} distarray/tests/ipcluster.py 'start()'

test:
	${COVERAGE} run -m unittest discover
	${MPIEXEC} -n 12 ${PYTHON} -m unittest discover -s distarray/local/tests -p 'paralleltest*.py' 

report:
	${COVERAGE} html

teardown_cluster:
	-${PYTHON} distarray/tests/ipcluster.py 'stop()'

clean:
	${PYTHON} setup.py clean --all
	-${RM} `find . -name '*.py[co]'`
	-${RM} `find . -name '*.so'`
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist distarray.egg-info .coverage
	-${RM} -rf coverage_report

