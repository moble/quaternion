import os
import re
import fileinput


def update(version):
    if not version:
        raise ValueError("Can't replace version with empty string")
    short_version = ".".join(version.split(".")[:2])

    files = ("setup.py", "src/quaternion/__init__.py")
    pattern = re.compile('^(__version__|version) *= *".*?"')
    replacement = r'\1 = "' + version + '"'
    with fileinput.input(files=files, inplace=True) as f:
        for line in f:
            print(pattern.sub(replacement, line), end="")


version = os.environ["new_version"]

update(version)
