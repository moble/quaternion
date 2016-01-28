#! /usr/bin/env sh
set -e

function usage {
    echo
    echo "Usage: update_anaconda_org.sh [conda_dir [conda_conversions [update_pypi]]]"
    echo
    echo '    conda_dir: Directory in which .binstar.yml is found, and any conversions are stored.  Defaults to `.`.'
    echo '    conda_conversions: {osx-64,linux-32,linux-64,win-32,win-64,all}, passed to `conda convert`.  Defaults to `all`.'
    echo '    update_pypi: If given as `--pypi`, run `python setup.py register sdist upload`.  Defaults to `--no-pypi`.'
    exit 1
}
if [[ $# == 1 && ( $1 == "-h" || $1 == "--help" ) ]]; then
    usage;
fi


CONDA_DIR=${1:-.}

CONVERSIONS=${2:-all}

PYPI=${3:---no-pypi}

CONDA_PYs=( 27 35 )
for CONDA_PY in "${CONDA_PYs[@]}"
do

    echo CONDA_PY=${CONDA_PY}
    export CONDA_PY
    conda build --no-binstar-upload ${CONDA_DIR}
    mkdir -p ${CONDA_DIR}/conversions
    conda convert -p ${CONVERSIONS} -o ${CONDA_DIR}/conversions `conda build ${CONDA_DIR} --output`

done
conda server upload --force ${CONDA_DIR}/conversions/*/*tar.bz2

if [ "$PYPI" == "--pypi" ]; then
    python setup.py register sdist upload
fi
