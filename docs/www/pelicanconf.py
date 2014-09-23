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

MENUITEMS = (
    ('HOME', '/'),
    ('FEATURES', '/pages/features.html'),
    ('RELEASE NOTES', '/category/release-notes.html'),
    ('TALKS', '/category/talks.html'),
    ('CONTACT', '/pages/contact.html'),
)


LINKS = (
    ('DistArray Docs', 'http://distarray.readthedocs.org/'),
    ('Distributed Array Protocol', 'http://distributed-array-protocol.readthedocs.org'),
    ('Mailing List', 'https://groups.google.com/forum/#!forum/distarray'),
    ('SciPy', 'http://www.scipy.org/'),
    ('IPython', 'http://ipython.org/'),
    ('Enthought', 'http://www.enthought.com/'),
)

SOCIAL = (
    ('github', 'https://github.com/enthought/distarray'),
    ('twitter', 'https://twitter.com/enthought'),
)

DEFAULT_PAGINATION = False
CACHE_CONTENT = False
SLUGIFY_SOURCE = 'basename'
PYGMENTS_STYLE = 'default'

SITELOGO = 'images/distarray-logo.png'
SITELOGO_SIZE = 230
HIDE_SITENAME = True

DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_PAGES_ON_MENU = False
DISPLAY_TAGS_ON_SIDEBAR = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

# pelican-bootstrap3 settings
THEME = "./pelican-bootstrap3/"
BOOTSTRAP_THEME = "enthought_dark"
SHOW_ARTICLE_AUTHOR = False
DISPLAY_ARTICLE_INFO_ON_INDEX = False

PLUGIN_PATHS = ['./pelican-plugins']
PLUGINS = ['liquid_tags.notebook']
