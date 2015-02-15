def calculate_version():
    try:
        import subprocess
        git_revision = subprocess.check_output("git show -s --format='%cI %h' HEAD", shell=True)
        date, short_hash = git_revision.split(' ')
        date = date.split('T')[0]  # remove ISO 8601 time info
        short_hash = short_hash[:-1]  # remove newline
        dirty = bool(subprocess.call("git diff-files --quiet --", shell=True))
        dirty = dirty or bool(subprocess.call("git diff-index --cached --quiet HEAD --", shell=True))
    except:
        from datetime import datetime
        date = datetime.now().isoformat().split('T')[0]
        short_hash = 'installed_without_git'
        dirty = False
    date = date.replace('-', '.')
    vers = '{0}.{1}'.format(date, short_hash)
    if dirty:
        vers += '.dirty'
    return vers


from distutils.command.build_py import build_py


class build_py_copy_version(build_py):
    def run(self):
        print(self.__dict__)
        build_py.run(self)  # distutils uses old-style classes, so no super()
        if not self.dry_run:
            import os.path
            with open(os.path.join(self.build_lib, 'waveforms', 'version.py'), 'w') as fobj:
                fobj.write('__version__ = "{0}"'.format(calculate_version()))
