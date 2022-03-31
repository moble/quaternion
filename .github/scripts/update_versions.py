import os
import re
import fileinput


def update(version):
    files = ("pyproject.toml", "setup.py", "src/quaternion/__init__.py")
    pattern = re.compile('^(__version__|version) *= *".*?"')
    replacement = r'\1 = "' + version + '"'
    with fileinput.input(files=files, inplace=True) as f:
        for line in f:
            print(pattern.sub(replacement, line), end="")


version = os.environ["new_version"]

update(version)
