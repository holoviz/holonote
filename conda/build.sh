#!/usr/bin/env bash

set -euxo pipefail

git status

hatch build --clean

VERSION="$(hatch version)"
export VERSION
conda build conda/recipe --no-anaconda-upload --no-verify
