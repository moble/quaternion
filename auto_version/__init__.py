"""Automatically create python package version info based on git date and hash

This simple little module is intended to be included as a git submodule by
other python packages, and used by setup.py on installation.  See the README at
<https://github.com/moble/auto_version> for more details and instructions on
how to use this module.

"""

import traceback

def calculate_version():
    try:
        import subprocess
        git_revision = subprocess.check_output("git show -s --format='%ci %h' HEAD", shell=True)
        print("git_revision:")
        print(git_revision)
        print(":git_revision")
        date, time, utc_offset, short_hash = git_revision.split(' ')
        date = date.replace('-', '.').strip()  # make date an acceptable version string
        short_hash = short_hash.strip()  # remove newline and any other whitespace
        dirty = bool(subprocess.call("git diff-files --quiet --", shell=True))
        dirty = dirty or bool(subprocess.call("git diff-index --cached --quiet HEAD --", shell=True))
        version = '{0}.{1}'.format(date, short_hash)
        if dirty:
            version += '.dirty'
        print('putative __version__ = "{0}"'.format(version))
        exec('putative__version__ = "{0}"'.format(version))  # see if this will raise an error for some reason
    except Exception as e:
        # If any of the above failed for any reason whatsoever, fall back on this dumb version
        print('\nThe `calculate_version` function failed to get the git version; maybe your version of git is too old?')
        print(traceback.format_exc())
        print(e)
        print('This should not be a problem, unless you need an accurate version number.')
        print('Continuing on, in spite of it all...\n')
        from datetime import datetime
        date = datetime.now().isoformat().split('T')[0]
        date = date.replace('-', '.').strip()
        version = '0.0.0.' + date
    return version


from distutils.command.build_py import build_py


class build_py_copy_version(build_py):
    def run(self):
        build_py.run(self)  # distutils uses old-style classes, so no super()
        version = calculate_version()
        print('build_py_copy_version using __version__ = "{0}"'.format(version))
        if not self.dry_run:
            import os.path
            for package in self.packages:
                with open(os.path.join(self.build_lib, os.path.join(*package.split('.')), '_version.py'), 'w') as fobj:
                    fobj.write('__version__ = "{0}"'.format(version))
