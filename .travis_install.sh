#!/bin/bash
set -ev

if [[ "${CONDA}" == "true" ]]; then

  if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh;
  else
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
  fi
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  hash -r
  conda info -a
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy scipy numba pip pytest anaconda-client
  source activate test-environment
  conda config --add channels moble
  conda install quaternion

else

  python setup.py install

fi