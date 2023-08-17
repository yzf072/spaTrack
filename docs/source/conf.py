import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
import spaTrack
project = 'spaTrack'
copyright = '2023, BGI'
author = 'BGI'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'nbsphinx',
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

napoleon_numpy_docstring = True
napoleon_use_rtype = True

intersphinx_mapping={
    'numpy': ('https://numpy.org/doc/stable/',None),
    'anndata': ('https://anndata.readthedocs.io/en/latest/',None),
    'python': ('https://docs.python.org/3/',None),
    'pandas': ('https://pandas.pydata.org/docs/',None),
}

# -- Options for HTML output -------------------------------------------------
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']