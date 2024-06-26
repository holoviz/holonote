#!/usr/bin/env bash

set -euxo pipefail

PACKAGE="holonote"

python -m build -w .
VERSION=$(python -c "import $PACKAGE; print($PACKAGE._version.__version__)")
export VERSION

# Update lockfiles
rm -rf node_modules
npm install .
node update_lock.js
python patch_lock.py
rm node_modules/pyodide/*.whl

jupyter lite build

cp -r node_modules/pyodide/ ../../jupyterlite/pyodide
mv pyodide-lock.json ../../jupyterlite/pyodide/pyodide-lock.json
mv ../../dist/* ../../jupyterlite/pyodide/
