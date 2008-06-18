.PHONY: clean inplace srcclean

CYTHON = cython
PYTHON = python
MPI4PY_INCLUDE = ${shell ${PYTHON} -c 'import mpi4py; print( mpi4py.get_include() )'}

src: ipythondistarray/core/maps_fast.c ipythondistarray/mpi/tests/helloworld.c

srcclean:
	-${RM} ipythondistarray/core/maps_fast.c
	-${RM} ipythondistarray/mpi/tests/helloworld.c

clean:
	${PYTHON} setup.py clean --all
	-${RM} `find . -name '*.py[co]'`
	-${RM} `find . -name '*.so'`
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist ipythondistarray.egg-info

ipythondistarray/core/maps_fast.c: ipythondistarray/core/maps_fast.pyx
	cd ipythondistarray/core && ${CYTHON} -I${MPI4PY_INCLUDE} maps_fast.pyx -o maps_fast.c

ipythondistarray/mpi/tests/helloworld.c: ipythondistarray/mpi/tests/helloworld.pyx
	cd ipythondistarray/mpi/tests && ${CYTHON} -I${MPI4PY_INCLUDE} helloworld.pyx -o helloworld.c


inplace: src
	${PYTHON} setup.py build_ext --inplace

