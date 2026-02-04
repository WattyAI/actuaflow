@echo off
setlocal
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1"=="" (
  %SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html
) else (
  %SPHINXBUILD% -b %1 %SOURCEDIR% %BUILDDIR%/%1
)

endlocal
