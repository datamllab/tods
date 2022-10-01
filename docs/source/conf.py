# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('./_ext'))
sys.path.append(os.path.abspath('../../tods'))
sys.path.insert(0, os.path.abspath('../sphinxext'))
from github_link import make_linkcode_resolve

# sys.path.append(os.path.abspath('../'))

# -- Auto-doc Skip --------------------
def skip_member(app, what, name, obj, skip, opts):
 # we can document otherwise excluded entities here by returning False
 # or skip otherwise included entities by returning True
    if what == "logger":
        return True
    return None

def setup(app):
    app.connect('autodoc-skip-member', skip_member)

# autodoc_inherit_docstrings = False

# -- Project information -----------------------------------------------------

project = 'TODS'
copyright = '2022, DataLab@Rice University'
author = 'DataLab@Rice University'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_thebe",
    #'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinxcontrib.bibtex',
    'sphinx_togglebutton',
]

# Add bib file
suppress_warnings = ["bibtex"]
bibtex_bibfiles = ['refs.bib']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "img/tods_menu_logo.png"

html_theme_options = {
    "repository_url": "https://github.com/datamllab/tods",
    "use_repository_button": True,

}
# html_theme_options = {
#     'logo_only': True,
#     'display_version': False,

# }
# html_sidebars = {
#    '**': ['fulltoc.html', 'sourcelink.html', 'searchbox.html', 'srclink.html']
# }
linkcode_resolve = make_linkcode_resolve('tods',
                                         'https://github.com/datamllab/'
                                         'tods/blob/dev/'
                                         '{package}/{path}#L{lineno}')
# https://github.com/datamllab/tods/tree/master/todstods/data_processing/CategoricalToBinary.py#L119


# https://github.com/datamllab/tods/tree/master/todstods/data_processing/CategoricalToBinary.py#L119