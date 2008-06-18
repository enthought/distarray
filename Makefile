.PHONY: clean inplace srcclean

CYTHON = cython
PYTHON = python
MPI4PY_INCLUDE = ${shell ${PYTHON} -c 'import mpi4py; print( mpi4py.get_include() )'}

src: distarray/core/maps_fast.c distarray/mpi/tests/helloworld.c

srcclean:
	-${RM} distarray/core/maps_fast.c
	-${RM} distarray/mpi/tests/helloworld.c

clean:
	${PYTHON} setup.py clean --all
	-${RM} `find . -name '*.py[co]'`
	-${RM} `find . -name '*.so'`
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist distarrayegg-info

distarray/core/maps_fast.c: distarray/core/maps_fast.pyx
	cd distarray/core && ${CYTHON} -I${MPI4PY_INCLUDE} maps_fast.pyx -o maps_fast.c

distarray/mpi/tests/helloworld.c: distarray/mpi/tests/helloworld.pyx
	cd distarray/mpi/tests && ${CYTHON} -I${MPI4PY_INCLUDE} helloworld.pyx -o helloworld.c


inplace: src
	${PYTHON} setup.py build_ext --inplace

