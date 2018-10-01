#! /bin/bash
set -e
set -x

. ~/.continuum/anaconda3/etc/profile.d/conda.sh
conda activate base

export package_version="${1:-$(date +'%Y.%-m.%-d.%-H.%-M.%-S')}"
echo "Building macosx wheels, version '${package_version}'"

temp_dir="${HOME}/Research/Temp"
wheelhouse="${temp_dir}/wheelhouse"
code_dir="${PWD}"

PYTHON_VERSIONS=( 2.7 3.5 3.6 3.7 )

/bin/rm -rf "${wheelhouse}"
mkdir -p "${wheelhouse}"

pip install --upgrade wheel
pip install --upgrade pipenv


# Loop through python versions, building wheels
for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
    if [ "$PYTHON_VERSION" -gt "3.6" ]; then
        requirements_build_txt="requirements-build-114.txt"
    else
        requirements_build_txt="requirements-build.txt"
    fi
    conda activate "py${PYTHON_VERSION}"
    /bin/rm -rf "${temp_dir}/quaternion-pipenv"
    mkdir -p "${temp_dir}/quaternion-pipenv"
    pushd "${temp_dir}/quaternion-pipenv"
    pipenv --python "${PYTHON_VERSION}"
    pipenv run pip install -r "${code_dir}/${requirements_build_txt}"
    pipenv run pip wheel "${code_dir}/" -r "${code_dir}/${requirements_build_txt}" -w "${wheelhouse}/"
    popd
    conda deactivate
done

# Bundle external shared libraries into the wheels
for whl in $(ls $(echo "${wheelhouse}/numpy_quaternion*.whl")); do
    echo
    delocate-listdeps --depending "$whl"
    delocate-wheel -v "$whl"
    delocate-listdeps --depending "$whl"
    echo
done


# ### NOTE: These lines are specialized for quaternion
# for CONDA_ENV in "${CONDA_ENVS[@]}"; do
#     source activate "${CONDA_ENV}"
#     # Install packages and test ability to import and run simple command
#     pip install --upgrade numpy-quaternion --no-index -f "${wheelhouse}"
#     (cd "$HOME"; python -c 'import numpy as np; import quaternion; print(quaternion.__version__); print("quaternion.z = {0}".format(quaternion.z))')
#     source deactivate
# done

# Just in case we failed to deactivate somehow:
source deactivate

# Upload to pypi
echo "Uploading to pypi"
pip install --quiet --upgrade twine
twine upload "${wheelhouse}"/numpy_quaternion*macosx*.whl
