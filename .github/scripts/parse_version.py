import sys


def parse(f):
    import tomli
    with open(f, "rb") as fp:
        pyproject = tomli.load(fp)
    return pyproject["tool"]["poetry"]["version"]


print(parse(sys.argv[1]))
