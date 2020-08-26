#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import sys
import numpy as np
import quaternion
from numpy import *


eps = np.finfo(float).eps


rot_mat_eps = 200*eps


R1 = np.array([quaternion.quaternion(0, -math.sqrt(0.5), 0, -math.sqrt(0.5))])
print(R1)
print(quaternion.as_rotation_matrix(R1))
R2 = quaternion.from_rotation_matrix(quaternion.as_rotation_matrix(R1))
print(R2)
d = quaternion.rotation_intrinsic_distance(R1[0], R2[0])
print(d)
print()
sys.stdout.flush()
sys.stderr.flush()
assert d < rot_mat_eps, (R1, R2, d)  # Can't use allclose here; we don't care about rotor sign
