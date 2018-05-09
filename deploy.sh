#! /bin/bash

set -e

export package_version=$(git log -1 --format=%cd --date=format:'%Y.%-m.%-d.%-H.%-M.%-S' || date +"%Y.%-m.%-d.%-H.%-M.%-S")
echo "Building version '${package_version}'"

# Rebuild and install locally, then test trivial action, to ensure there are no warnings
/bin/rm -rf build __pycache__
CFLAGS='-Werror -Wall -Wextra' python setup.py install
python -c 'import numpy as np; import quaternion; tmp = quaternion.quaternion(1,2,3,4)'

# Create a pure source pip package
python setup.py sdist upload

# Create all the osx binary pip packages
./deploy/build_macosx_wheels.sh "${package_version}"

# Start docker for the linux packages
open --hide --background -a Docker
while ! (docker ps > /dev/null 2>&1); do
    echo "Waiting for docker to start..."
    sleep 1
done

# Create all the linux binary pip packages on centos 5
docker run -i -t \
    -v ${HOME}/.pypirc:/root/.pypirc:ro \
    -v `pwd`:/code \
    -v `pwd`/deploy/build_manylinux_wheels.sh:/build_manylinux_wheels.sh \
    quay.io/pypa/manylinux1_x86_64 /build_manylinux_wheels.sh "${package_version}"
