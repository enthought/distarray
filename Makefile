PYTHON = python
MPIEXEC = mpiexec
COVERAGE = coverage

develop:
	${PYTHON} setup.py develop
.PHONY: develop

install:
	${PYTHON} setup.py install
.PHONY: install

setup_cluster:
	-${PYTHON} distarray/tests/ipcluster.py 'start()'
.PHONY: setup_cluster

test_client:
	${PYTHON} -m unittest discover -v
.PHONY: test_client

test_client_with_coverage:
	${COVERAGE} run -pm unittest discover -v
.PHONY: test_client_with_coverage

test_engines:
	${MPIEXEC} -n 12 ${PYTHON} -m unittest discover -s distarray/local/tests -p 'paralleltest*.py'
.PHONY: test_engines

test_engines_with_coverage:
	${MPIEXEC} -n 12 ${COVERAGE} run -pm unittest discover -s distarray/local/tests -p 'paralleltest*.py'
.PHONY: test_engines_with_coverage

test: test_client test_engines
.PHONY: test

test_with_coverage: test_client_with_coverage test_engines_with_coverage
.PHONY: test_with_coverage

coverage_report:
	${COVERAGE} combine
	${COVERAGE} html
.PHONY: coverage_report

teardown_cluster:
	-${PYTHON} distarray/tests/ipcluster.py 'stop()'
.PHONY: teardown_cluster

restart_cluster:
	-${PYTHON} distarray/tests/ipcluster.py 'restart()'
.PHONY: restart_cluster

clean:
	-${PYTHON} setup.py clean --all
	-find . \( -iname '*.py[co]' -or -iname '*.so' -or -iname '__pycache__' \) -exec ${RM} -r '{}' +
	-${RM} -r build MANIFEST dist distarray.egg-info coverage_report
.PHONY: clean
