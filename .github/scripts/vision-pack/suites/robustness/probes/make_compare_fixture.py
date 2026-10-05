#!/usr/bin/env python3
# Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Create a golden/output pair that must fail rocAL's image_comparison.py: every pixel differs.

    make_compare_fixture.py <dir>    creates <dir>/golden/ and <dir>/rocal/ with Brightness_rgb_hip.png
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
for sub, value in (("golden", 0), ("rocal", 200)):
    (root / sub).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, np.uint8)).save(root / sub / "Brightness_rgb_hip.png")
print(f"fixture in {root}: golden all 0, rocal all 200")
