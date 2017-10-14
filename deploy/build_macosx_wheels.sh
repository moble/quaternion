#! /bin/bash
set -e
export datetime="${1:-$(date +'%Y.%m.%d.%H.%M.%S')}"

wheelhouse="${HOME}/Research/Temp/wheelhouse"

/bin/rm -rf "${wheelhouse}"
mkdir -p "${wheelhouse}"

PYBINS=( "/Users/boyle/.continuum/anaconda3/envs/py27/bin" "/Users/boyle/.continuum/anaconda3/bin" )
LAST_PYBIN="${PYBINS[1]}"

# Compile wheels
for PYBIN in "${PYBINS[@]}"; do
    ### NOTE: The path to the requirements file is specialized for spinsfast
    "${PYBIN}/pip" install -r ./requirements.txt
    "${PYBIN}/pip" wheel ./ -w "${wheelhouse}/"
done

# Bundle external shared libraries into the wheels
for whl in $(ls $(echo "${wheelhouse}/*.whl")); do
    echo
    delocate-listdeps --depending "$whl"
    delocate-wheel -v "$whl"
    delocate-listdeps --depending "$whl"
    echo
done


### NOTE: These lines are specialized for spinsfast
for PYBIN in "${PYBINS[@]}"; do
    # Install packages and test ability to import and run simple command
    "${PYBIN}/pip" install --upgrade spinsfast --no-index -f "${wheelhouse}"
    (cd "$HOME"; "${PYBIN}/python" -c 'import spinsfast; print(spinsfast.__version__); print("N_lm(8) = {0}".format(spinsfast.N_lm(8)))')
done


# Upload to pypi
"${LAST_PYBIN}"/pip install twine
"${LAST_PYBIN}"/twine upload "${wheelhouse}"/*macosx*.whl
