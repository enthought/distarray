#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'IPython development team and Enthought, Inc.'
SITENAME = u'DistArray'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'America/Chicago'

DEFAULT_LANG = u'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None

# Blogroll
LINKS = (('PyPI', 'https://pypi.python.org/pypi/distarray'),
         ('Docs', 'http://distarray.readthedocs.org/'),
         ('Distributed Array Protocol', 'http://distributed-array-protocol.readthedocs.org'),
         ('SciPy', 'http://www.scipy.org/'),
         ('IPython', 'http://ipython.org/'),
         ('Enthought', 'http://www.enthought.com/'),
        )

# Social widget
SOCIAL = (('github', 'https://github.com/enthought/distarray'),)

DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

THEME = "./pelican-bootstrap3/"
