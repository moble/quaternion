"""Automatically create python package version info based on git date and hash

This simple little module is intended to be included as a git submodule by
other python packages, and used by setup.py on installation.  See the README at
<https://github.com/moble/auto_version> for more details and instructions on
how to use this module.

"""


def calculate_version():
    try:
        import subprocess
        git_revision = subprocess.check_output("git show -s --format='%cI %h' HEAD", shell=True)
        date, short_hash = git_revision.split(' ')
        date = date.split('T')[0]  # remove ISO 8601 time info
        date = date.replace('-', '.')  # make date an acceptable version string
        short_hash = short_hash[:-1]  # remove newline
        dirty = bool(subprocess.call("git diff-files --quiet --", shell=True))
        dirty = dirty or bool(subprocess.call("git diff-index --cached --quiet HEAD --", shell=True))
        version = '{0}.{1}'.format(date, short_hash)
        if dirty:
            version += '.dirty'
    except:
        from datetime import datetime
        date = datetime.now().isoformat().split('T')[0]
        date = date.replace('-', '.')
        version = '0.0.0.' + date
    return version


from distutils.command.build_py import build_py


class build_py_copy_version(build_py):
    def run(self):
        build_py.run(self)  # distutils uses old-style classes, so no super()
        if not self.dry_run:
            import os.path
            for package in self.packages:
                with open(os.path.join(self.build_lib, os.path.join(*package.split('.')), '_version.py'), 'w') as fobj:
                    fobj.write('__version__ = "{0}"'.format(calculate_version()))
