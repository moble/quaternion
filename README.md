# auto_version

Automatically create python package version info based on git commit date and
hash.  This simple little module is intended to be included as a git submodule
by other python packages, and used by `setup.py` on installation.

The `calculate_version` function constructs a version number from the date.
This will be of the form

    2015.02.04.82e4326

where `2015.02.04` represents the date of the commit, and `82e4326` is the
shortened hash of the commit.  Additionally, if the code is not in a clean
state (changes have been made since the last commit), then `.dirty` will be
appended to the version.  This form requires the ability to run a few simple
`git` commands from `python`.  If this is not possible, the system will fall
back to the current date, and `installed_without_git` in place of the commit
hash.

The `build_py_copy_version` class wraps the basic `build_py` class used in
the standard setup function, but adds a step at the end to create a file named
`version.py` that gets copied into the installed module.

To use both of these in the enclosing package, the enclosing `setup.py` file
could contain something like this:

```python
import distutils.core
from auto_version import calculate_version, build_py_copy_version

distutils.core.setup(name='enclosing_package_name',
                     version=calculate_version(),
                     cmdclass={'build_py': build_py_copy_version},
                     ...,
                     )
```

And in the package's `__init__.py` file, you could have something like this:

```python
from .version import __version__
```

Then, in other code you would see the version normally:

```python
import enclosing_package_name
print(enclosing_package_name.__version__)
```
