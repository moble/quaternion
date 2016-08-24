#!/bin/bash
# this script uses the ANACONDA_TOKEN env var.
# to create a token:
# >>> anaconda login
# >>> anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/moble/quaternion --scopes "api:write api:read"
# then go to the Travis-CI settings page for this package, and create an environment variable with that token.

# I stole these ideas from Yoav Ram <https://gist.github.com/yoavram/05a3c04ddcf317a517d5>

echo "Trying to run deploy script"

set -e

if [[ "${CONDA}" == "true" ]]; then

    conda config --set anaconda_upload no;

    conda install -n root conda-build anaconda-client

    echo "Building package"
    conda build .

    echo "Converting conda package..."
    conda convert -f --platform osx-64 $HOME/miniconda/conda-bld/linux-64/quaternion-*.tar.bz2 --output-dir conda-bld/
    conda convert -f --platform linux-32 $HOME/miniconda/conda-bld/linux-64/quaternion-*.tar.bz2 --output-dir conda-bld/
    conda convert -f --platform linux-64 $HOME/miniconda/conda-bld/linux-64/quaternion-*.tar.bz2 --output-dir conda-bld/

    echo "Deploying to Anaconda.org..."
    anaconda -t $ANACONDA_TOKEN upload conda-bld/**/quaternion-*.tar.bz2

    echo "Successfully deployed to Anaconda.org."

else

    echo "Skipping deployment"

fi


exit 0
