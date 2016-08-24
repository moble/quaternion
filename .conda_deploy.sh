#!/bin/bash
# this script uses the ANACONDA_TOKEN env var.
# to create a token:
# >>> anaconda login
# >>> anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/moble/quaternion --scopes "api:write api:read"
# then go to the Travis-CI settings page for this package, and create an environment variable with that token.

set -e

if [[ "${CONDA}" != "true" ]]; then
    exit 0
fi

conda config --set anaconda_upload no;

echo "Building package"
conda build .

echo "Converting conda package..."
conda convert --platform all $HOME/miniconda/conda-bld/linux-64/quaternion-*.tar.bz2 --output-dir conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload conda-bld/**/quaternion-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
exit 0
