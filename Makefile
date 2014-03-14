#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

PYTHON := python
PYTHON_VERSION := $(shell ${PYTHON} --version 2>&1 | cut -f 2 -d ' ')

COVERAGE := coverage

MPIEXEC := mpiexec

PARALLEL_UNITTEST_ARGS := -m unittest discover -s distarray/local/tests -p 'paralleltest*.py'
PARALLEL_TEST_REGULAR := ${PYTHON} ${PARALLEL_UNITTEST_ARGS}
PARALLEL_TEST_COVERAGE := ${COVERAGE} run -p ${PARALLEL_UNITTEST_ARGS}

MPI_OUT_PREFIX := unittest-${PYTHON_VERSION}.out
MPIEXEC_ARGS := --output-filename ${MPI_OUT_PREFIX} -n 12

# Inside MPI_EXEC_CMD, PARALLEL_TEST is meant to be substituted with either
# PARALLEL_TEST_REGULAR or PARALLEL_TEST_COVERAGE from above.  See the
# `test_engines` and `test_engines_with_coverage` targets.
MPI_EXEC_CMD = (${MPIEXEC} ${MPIEXEC_ARGS} ${PARALLEL_TEST} ; OUT=$$? ; \
			   for f in ${MPI_OUT_PREFIX}* ; do echo "====> " $$f ; cat $$f ; done ; \
			   exit $$OUT)

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
	-${RM} ${MPI_OUT_PREFIX}*
	$(eval PARALLEL_TEST := ${PARALLEL_TEST_REGULAR})
	${MPI_EXEC_CMD}
.PHONY: test_engines

test_engines_with_coverage:
	-${RM} ${MPI_OUT_PREFIX}*
	$(eval PARALLEL_TEST := ${PARALLEL_TEST_COVERAGE})
	${MPI_EXEC_CMD}

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
	-${RM} -r build MANIFEST dist distarray.egg-info coverage_report ${MPI_OUT_PREFIX}*
.PHONY: clean
