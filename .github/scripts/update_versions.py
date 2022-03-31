import os
import re
import fileinput


def update(version):
    pattern = re.compile('^(__version__|version) *= *".*?"')
    replacement = r'\1 = "' + version + '"'
    with fileinput.input(files=("setup.py", "src/quaternion/__init__.py"), inplace=True) as f:
        for line in f:
            print(pattern.sub(replacement, line))

version = os.environ["new_version"]

update(version)
