.PHONY: clean setup_cluster test test_travis teardown_cluster

PYTHON = python
MPIEXEC = mpiexec
COVERAGE = coverage

develop:
	${PYTHON} setup.py develop

install:
	${PYTHON} setup.py install

setup_cluster:
	-${PYTHON} distarray/tests/ipcluster.py start

test:
	${COVERAGE} run -m unittest discover
	${MPIEXEC} -n 12 ${PYTHON} -m unittest discover -s distarray/local/tests -p 'paralleltest*.py' 

report:
	${COVERAGE} html

teardown_cluster:
	-${PYTHON} distarray/tests/ipcluster.py stop

clean:
	-${PYTHON} setup.py clean --all
	-find . \( -iname '*.py[co]' -or -iname '*.so' -or -iname '__pycache__' \) -exec ${RM} '{}' +
	-${RM} -r build MANIFEST dist distarray.egg-info coverage_report
