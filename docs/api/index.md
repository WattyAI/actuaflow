# API Reference

This folder is intended for the API reference. You can generate API pages automatically using `sphinx-apidoc` or the `sphinx.ext.autodoc` directives.

To auto-generate stubs with sphinx-apidoc (from project root):

```
sphinx-apidoc -o docs/api actuaflow
```

After generation, build the docs as shown in the project root: `make html` or `make.bat html`.

Auto-generated files will appear alongside this index. For manual additions, add Markdown files here and reference them from `docs/index.md`.
