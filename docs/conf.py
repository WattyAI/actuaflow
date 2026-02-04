import os
import sys
# This allows Sphinx to find your code in the parent directory
sys.path.insert(0, os.path.abspath('..'))

project = 'ActuaFlow'
author = 'Michael Watson'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',       # Pulls docstrings from code
    'sphinx.ext.autosummary',   # Generates summary tables
    'sphinx.ext.napoleon',      # Supports Google/NumPy style docstrings
    'sphinx_autodoc_typehints', # Shows types in docs
    'myst_parser',              # Supports Markdown (.md)
    'sphinx.ext.viewcode',      # Adds links to source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Support both .rst and .md source files (MyST for Markdown)
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc = 'index'
html_theme = 'sphinx_rtd_theme'

# HTML options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Avoid failures when building docs in minimal environments by mocking heavy optional imports
autodoc_mock_imports = ['statsmodels', 'polars', 'matplotlib', 'sklearn', 'numpy', 'pandas']

# MyST parser options (optional)
myst_enable_extensions = [
    'colon_fence',
]

# Autosummary: generate summary stubs for documented modules
autosummary_generate = True

