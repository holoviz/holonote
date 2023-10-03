#!/usr/bin/env bash

set -euxo pipefail

git status

hatch build --clean

export VERSION="$(hatch version)"
conda build conda/recipe --no-test --no-anaconda-upload --no-verify
