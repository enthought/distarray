.PHONY: clean inplace srcclean

CYTHON = cython
PYTHON = python
MPI4PY_INCLUDE = ${shell ${PYTHON} -c 'import mpi4py; print( mpi4py.get_include() )'}


src: distarray/core/maps.c distarray/mpi/tests/helloworld.c

srcclean:
	-${RM} distarray/core/maps.c
	-${RM} distarray/mpi/tests/helloworld.c

clean:
	${PYTHON} setup.py clean --all
	-${RM} `find . -name '*.py[co]'`
	-${RM} `find . -name '*.so'`
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist distarrayegg-info

distarray/core/maps.c: distarray/core/maps.pyx
	${CYTHON} -I. -I${MPI4PY_INCLUDE} distarray/core/maps.pyx

distarray/mpi/tests/helloworld.c: distarray/mpi/tests/helloworld.pyx
	${CYTHON} -I. -I${MPI4PY_INCLUDE} distarray/mpi/tests/helloworld.pyx

inplace: src
	${PYTHON} setup.py build_ext --inplace

