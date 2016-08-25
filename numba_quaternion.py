# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import numpy

from numba import utils, float32, float64
from numba.types import Type, number_domain


@utils.total_ordering
class Quaternion(Type):
    def __init__(self, name, underlying_float, **kwargs):
        super(Quaternion, self).__init__(name, **kwargs)
        self.underlying_float = underlying_float
        # Determine bitwidth
        assert self.name.startswith('quaternion')
        bitwidth = int(self.name[10:])
        self.bitwidth = bitwidth

    def cast_python_value(self, value):
        return getattr(numpy, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth


quaternion128 = Quaternion('quaternion128', float32)
quaternion256 = Quaternion('quaternion256', float64)
quaternion_domain = frozenset([quaternion128, quaternion256])
number_domain = number_domain | quaternion_domain

q16 = quaternion128
q32 = quaternion256
