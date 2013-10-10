.PHONY: clean

PYTHON = python

develop:
	${PYTHON} setup.py develop

install:
	${PYTHON} setup.py install

test:
	mpiexec -n 12 nosetests -w distarray/core/tests
	mpiexec -n 4 nosetests -w distarray/random/tests
	nosetests

clean:
	${PYTHON} setup.py clean --all
	-${RM} `find . -name '*.py[co]'`
	-${RM} `find . -name '*.so'`
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist distarray.egg-info
