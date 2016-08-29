#!/bin/bash
set -ev

# Determine which python version we will install
case "${PYTHON_VERSION:-3}" in
    2*)
        Miniconda="Miniconda2"
        ;;
    3*)
        Miniconda="Miniconda3"
        ;;
    *)
        echo "Unrecognized python version: '${PYTHON_VERSION:-3}'"
        exit 1
        ;;
esac

# Determine which OS we should install
case "${CONDA_INSTALLER_OS:-linux}" in
    linux)
        CondaOS='Linux'
        ;;
    osx)
        CondaOS='MacOSX'
        ;;
    windows)
        CondaOS='Windows'
        ;;
    *)
        echo "Unrecognized OS in conda installer: '${CONDA_INSTALLER_OS:-linux}'"
        exit 1
        ;;
esac

# Download the appropriate miniconda installer
wget http://repo.continuum.io/miniconda/${Miniconda}-latest-${CondaOS}-x86_64.sh -O miniconda.sh;

# if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
#     if [[ "$TRAVIS_PYTHON_VERSION" == "2."* ]]; then
#         wget http://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh -O miniconda.sh;
#     else
#         wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
#     fi
# else
#     if [[ "$TRAVIS_PYTHON_VERSION" == "2."* ]]; then
#         wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh;
#     else
#         wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
#     fi
# fi

# Run the installer
bash miniconda.sh -b -p $HOME/miniconda

# Update and activate a new conda environment
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda info -a
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda create -q -n test-environment python="${PYTHON_VERSION:-3}" $*
source activate test-environment
pip install --upgrade pip setuptools
