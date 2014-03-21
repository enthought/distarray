#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

PYTHON := python
PYTHON_VERSION := $(shell ${PYTHON} --version 2>&1 | cut -f 2 -d ' ')

COVERAGE := coverage

MPIEXEC := mpiexec

NPROCS := 12

PARALLEL_OUT_DIR := .parallel_out

PARALLEL_UNITTEST_ARGS := -m unittest discover -s distarray/local/tests -p 'paralleltest*.py'
PARALLEL_TEST_REGULAR := ${PYTHON} ${PARALLEL_UNITTEST_ARGS}
PARALLEL_TEST_COVERAGE := ${COVERAGE} run -p ${PARALLEL_UNITTEST_ARGS}

MPI_OUT_BASE := unittest.out
MPI_OUT_PREFIX := ${PARALLEL_OUT_DIR}/${PYTHON_VERSION}-${MPI_OUT_BASE}
MPIEXEC_ARGS := --output-filename ${MPI_OUT_PREFIX} -n ${NPROCS}

# Inside MPI_EXEC_CMD, PARALLEL_TEST is meant to be substituted with either
# PARALLEL_TEST_REGULAR or PARALLEL_TEST_COVERAGE from above.  See the
# `test_engines` and `test_engines_with_coverage` targets.
MPI_EXEC_CMD = (${MPIEXEC} ${MPIEXEC_ARGS} ${PARALLEL_TEST} ; OUT=$$? ; \
			   for f in ${MPI_OUT_PREFIX}* ; do echo "====> " $$f ; cat $$f ; done ; \
			   exit $$OUT)

IPCLUSTER_SCRIPT := distarray/tests/ipcluster.py
PURGE_SCRIPT := distarray/tests/purge_cluster.py

# default number of engines to use.
NENGINES := 4

# ----------------------------------------------------------------------------
#  Installation targets.
# ----------------------------------------------------------------------------

develop:
	${PYTHON} setup.py develop
.PHONY: develop

install:
	${PYTHON} setup.py install
.PHONY: install

# ----------------------------------------------------------------------------
#  Testing-related targets.
# ----------------------------------------------------------------------------

test_client:
	${PYTHON} -m unittest discover -v
.PHONY: test_client

test_client_with_coverage:
	${COVERAGE} run -pm unittest discover -v
.PHONY: test_client_with_coverage

${PARALLEL_OUT_DIR} :
	mkdir ${PARALLEL_OUT_DIR}

test_engines: ${PARALLEL_OUT_DIR}
	@-${RM} ${MPI_OUT_PREFIX}*
	$(eval PARALLEL_TEST := ${PARALLEL_TEST_REGULAR})
	@echo "Running '${PARALLEL_TEST}' on each engine..."
	@${MPI_EXEC_CMD}
.PHONY: test_engines

test_engines_with_coverage: ${PARALLEL_OUT_DIR}
	@-${RM} ${MPI_OUT_PREFIX}*
	$(eval PARALLEL_TEST := ${PARALLEL_TEST_COVERAGE})
	@echo "Running '${PARALLEL_TEST}' on each engine..."
	@${MPI_EXEC_CMD}
.PHONY: test_engines_with_coverage

test: test_client test_engines
.PHONY: test

test_with_coverage: test_client_with_coverage test_engines_with_coverage
.PHONY: test_with_coverage

coverage_report:
	${COVERAGE} combine
	${COVERAGE} html
.PHONY: coverage_report

# ----------------------------------------------------------------------------
#  Cluster management
# ----------------------------------------------------------------------------

setup_cluster:
	${PYTHON} ${IPCLUSTER_SCRIPT} start ${NENGINES}
.PHONY: setup_cluster

restart_cluster:
	${PYTHON} ${IPCLUSTER_SCRIPT} restart ${NENGINES}
.PHONY: restart_cluster

teardown_cluster:
	${PYTHON} ${IPCLUSTER_SCRIPT} stop
.PHONY: teardown_cluster

clear_distarray:
	${PYTHON} ${IPCLUSTER_SCRIPT} reset
.PHONY: clear_distarray

purge_cluster:
	${PYTHON} ${PURGE_SCRIPT} purge
.PHONY: purge_cluster

dump_cluster:
	${PYTHON} ${PURGE_SCRIPT} dump
.PHONY: dump_cluster

# ----------------------------------------------------------------------------
#  Cleanup.
# ----------------------------------------------------------------------------

clean:
	-${PYTHON} setup.py clean --all
	-find . \( -iname '*.py[co]' -or -iname '*.so' -or -iname '__pycache__' \) -exec ${RM} -r '{}' +
	-${RM} -r ${PARALLEL_OUT_DIR} build MANIFEST dist distarray.egg-info coverage_report
.PHONY: clean
