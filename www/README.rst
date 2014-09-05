DistArray Website
=================

To build, install `pelican`_, with ``pip`` and try::

    make html

You can then see the generated website locally by doing::

    make serve

and checking out ``localhost:8000`` in a web browser.

To deploy the website to our Github-hosted gh-pages, install `ghp-import`_ with
``pip`` and try::

    make github

To add content, add ``.rst`` or ``.md`` files to the directory ``www/content``.
Check out the other files there for examples (as well as the `pelican`_
documentation).

If you want to change major features or styling of the site, you can change
settings in ``pelicanconf.py``.  We're currently using the
``pelican-bootstrap3`` theme for Pelican, and the ``flatly`` theme for
Bootstrap.  The easist way to change the look of the site is to choose another
Bootstrap theme from `bootswatch`_, but there are also other `pelican themes`_
available 


.. _pelican: http://blog.getpelican.com/
.. _pelican themes: https://github.com/getpelican/pelican-themes``.
.. _bootswatch: http://bootswatch.com/
.. _ghp-import: https://pypi.python.org/pypi/ghp-import
