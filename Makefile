.PHONY: clean setup_cluster test test_travis teardown_cluster test_client test_engines

PYTHON = python
MPIEXEC = mpiexec
COVERAGE = coverage

develop:
	${PYTHON} setup.py develop

install:
	${PYTHON} setup.py install

setup_cluster:
	-${PYTHON} distarray/tests/ipcluster.py 'start()'

test_client:
	${PYTHON} -m unittest discover

test_client_with_coverage:
	${COVERAGE} run -m unittest discover

test_engines:
	${MPIEXEC} -n 12 ${PYTHON} -m unittest discover -s distarray/local/tests -p 'paralleltest*.py' 

test: test_client test_engines

test_with_coverage: test_client_with_coverage test_engines

report:
	${COVERAGE} html

teardown_cluster:
	-${PYTHON} distarray/tests/ipcluster.py 'stop()'

clean:
	-${PYTHON} setup.py clean --all
	-find . \( -iname '*.py[co]' -or -iname '*.so' -or -iname '__pycache__' \) -exec ${RM} -r '{}' +
	-${RM} -r build MANIFEST dist distarray.egg-info coverage_report
