# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##
# @file random.py
#
# @brief File containing randomization functions used for creating RNG generators

import rocal_pybind as b


def coin_flip(*inputs, probability=0.5):
    """!coin flip with a given probability of success.

        @param inputs         list of input arguments.
        @param probability    Probability of getting a "success" outcome.

        @return    An array of coin flip outcomes.
    """
    values = [0, 1]
    frequencies = [1 - probability, probability]
    output_array = b.createIntRand(values, frequencies)
    return output_array


def uniform(*inputs, range=[-1, 1]):
    """!Generates random values uniformly distributed within a specified range.

        @param inputs    list of input arguments.
        @param range     Range for the uniform distribution.

        @return    random values uniformly distributed within the specified range.
    """
    output_param = b.createFloatUniformRand(range[0], range[1])
    return output_param
