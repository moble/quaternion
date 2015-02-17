"""Automatically create python package version info based on git date and hash

This simple little module is intended to be included as a git submodule by
other python packages, and used by setup.py on installation.  See the README at
<https://github.com/moble/auto_version> for more details and instructions on
how to use this module.

"""

import subprocess


if "check_output" not in dir(subprocess):
    """Duck punch python <=2.6 as necessary

    The version of subprocess in python 2.6 doesn't have `check_output`, so we
    have to duck-punch it in, as suggested in this stackoverflow answer:
    <http://stackoverflow.com/a/13160748/1194883>.

    This code is taken from the python 2.7 code, except that
    `CalledProcessError` needs to be given the namespace, and doesn't take a
    the output argument.

    """
    def f(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = subprocess.Popen(stdout=PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise subprocess.CalledProcessError(retcode, cmd)  # , output=output)
        return output
    subprocess.check_output = f


def calculate_version():
    try:
        git_revision = subprocess.check_output("git show -s --format='%ci %h' HEAD", shell=True).decode('ascii')
        date, time, utc_offset, short_hash = git_revision.split(' ')
        date = date.replace('-', '.').strip()  # make date an acceptable version string
        short_hash = short_hash.strip()  # remove newline and any other whitespace
        dirty = bool(subprocess.call("git diff-files --quiet --", shell=True))
        dirty = dirty or bool(subprocess.call("git diff-index --cached --quiet HEAD --", shell=True))
        version = '{0}.{1}'.format(date, short_hash)
        if dirty:
            version += '.dirty'
        exec('putative__version__ = "{0}"'.format(version))  # see if this will raise an error for some reason
    except Exception as e:
        # If any of the above failed for any reason whatsoever, fall back on this dumb version
        print('\nThe `calculate_version` function failed to get the git version.')
        print('Maybe your version of python (<2.7?) is too old.')
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
