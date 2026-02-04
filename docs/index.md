# ActuaFlow Documentation

Welcome to the ActuaFlow documentation. This site contains the API reference, conceptual guides, and runnable examples.

## Sections

```{toctree}
:maxdepth: 2

api/index.md
guide/index.md
examples/index.md
```

## Quick build

Install the docs dependencies (recommended):

```
pip install -e .[docs]
pip install sphinx-rtd-theme myst-parser
```

Build HTML (Linux/macOS):

```
make html
```

Build HTML (Windows):

```
make.bat html
```
