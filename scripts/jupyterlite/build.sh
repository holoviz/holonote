#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Make new wheel
hatch build --clean -t wheel
VERSION=$(hatch version)
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
