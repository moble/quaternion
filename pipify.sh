#! /bin/bash

set -e
set -x

export package_version=$(git log -1 --format=%cd --date=format:'%Y.%-m.%-d.%-H.%-M.%-S' || date +"%Y.%-m.%-d.%-H.%-M.%-S")
echo "Building version '${package_version}'"

# Rebuild and install locally, then test trivial action, to ensure there are no warnings
/bin/rm -rf build __pycache__ dist numpy_quaternion.egg-info
python setup.py install
python -c 'import numpy as np; import quaternion; tmp = quaternion.quaternion(1,2,3,4)'

# Create and upload a pure source pip package
pip install --quiet --upgrade twine
/bin/rm -rf build __pycache__ dist numpy_quaternion.egg-info
python setup.py sdist
twine upload dist/*
