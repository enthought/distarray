.PHONY: cython

CYTHON = cython
PYTHON = python
MPI4PY_INCLUDE = ${shell ${PYTHON} -c 'import mpi4py; print( mpi4py.get_include() )'}

src: ipythondistarray/core/maps_fast.c

clean:
	-${RM} ipythondistarray/core/maps_fast.c

ipythondistarray/core/maps_fast.c: ipythondistarray/core/maps_fast.pyx
	cd ipythondistarray/core && ${CYTHON} -I${MPI4PY_INCLUDE} maps_fast.pyx -o maps_fast.c

inplace: src
	${PYTHON} setup.py build_ext --inplace

