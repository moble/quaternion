#! /usr/bin/env sh
set -e

PWD=`pwd`
DIR=`basename $PWD`

anaconda-build trigger moble/${DIR}

export CONDA_NPY=19
CONDA_PYs=( 27 34 )

for CONDA_PY in "${CONDA_PYs[@]}"
do

    echo CONDA_PY=${CONDA_PY} CONDA_NPY=${CONDA_NPY}
    export CONDA_PY
    conda build --no-binstar-upload ${1:-.}
    conda server upload --force `conda build ${1:-.} --output`

done
