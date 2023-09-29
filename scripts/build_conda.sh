#!/usr/bin/env bash

set -euxo pipefail

git status

hatch build --clean

export VERSION="$(echo "$(ls dist/*.whl)" | cut -d- -f2)"
conda build conda/recipe --no-test --no-anaconda-upload --no-verify
