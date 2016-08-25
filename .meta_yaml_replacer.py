#!/usr/bin/env python

# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

import fileinput
from auto_version import calculate_version

version_string = calculate_version()

with fileinput.FileInput('meta.yaml', inplace=True) as file:
    for line in file:
        print(line.replace("version: '1.0'", "version: '{0}'".format(version_string)), end='')
