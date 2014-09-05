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
LINKS = (
    ('DistArray Docs', 'http://distarray.readthedocs.org/'),
    ('Distributed Array Protocol', 'http://distributed-array-protocol.readthedocs.org'),
    ('Mailing List', 'https://groups.google.com/forum/#!forum/distarray'),
    ('SciPy', 'http://www.scipy.org/'),
    ('IPython', 'http://ipython.org/'),
    ('Enthought', 'http://www.enthought.com/'),
)

# Social widget
SOCIAL = (
    ('github', 'https://github.com/enthought/distarray'),
    ('twitter', 'https://twitter.com/enthought'),
)

DEFAULT_PAGINATION = False

SHOW_ARTICLE_AUTHOR = False
DISPLAY_ARTICLE_INFO_ON_INDEX = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

BOOTSTRAP_THEME = 'Flatly'
THEME = "./pelican-bootstrap3/"
