#!/usr/bin/env bash

set -euxo pipefail

git status

hatch build --clean -t wheel

VERSION="$(hatch version)"
export VERSION
conda build scripts/conda/recipe --no-anaconda-upload --no-verify
