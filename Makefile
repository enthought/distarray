.PHONY: clean

PYTHON = python

develop:
	${PYTHON} setup.py develop

install:
	${PYTHON} setup.py install

test:
	nosetests

clean:
	${PYTHON} setup.py clean --all
	-${RM} `find . -name '*.py[co]'`
	-${RM} `find . -name '*.so'`
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist distarray.egg-info
