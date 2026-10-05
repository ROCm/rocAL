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

"""DEV-ONLY stand-in for the few OpenCV calls in rocAL's python_api tests. NOT OpenCV.

Used only on developer hosts without python3-opencv, and only when VP_ROCAL_CV2_SHIM=1;
run.sh refuses it inside CI containers and names it in every affected result message.
It implements just the lossless operations those scripts need (channel swaps, stacking,
PNG/JPEG I/O, rectangles, min-max normalize) with numpy + PIL, so pixel results match
real OpenCV for the golden comparison. Ported from the manual QA session.
"""
import numpy as np
from PIL import Image, ImageDraw

COLOR_RGB2BGR = 4
COLOR_BGR2RGB = 4
COLOR_GRAY2BGR = 8
IMWRITE_PNG_COMPRESSION = 16
IMREAD_COLOR = 1
NORM_MINMAX = 32
CV_32F = 5
__version__ = "vp-rocal-devshim-0"


def _arr(img):
    return img.get() if isinstance(img, UMat) else np.asarray(img)


class UMat:
    def __init__(self, a=None):
        self._a = np.ascontiguousarray(a) if a is not None else None

    def get(self):
        return self._a


def cvtColor(img, code):  # noqa: N802 - OpenCV name
    a = _arr(img)
    if code == COLOR_RGB2BGR:
        return np.ascontiguousarray(a[..., ::-1])
    if code == COLOR_GRAY2BGR:
        g = a[..., 0] if a.ndim == 3 else a
        return np.ascontiguousarray(np.stack([g, g, g], axis=-1))
    raise NotImplementedError(f"cv2 devshim: cvtColor code {code}")


def vconcat(images):
    return np.ascontiguousarray(np.vstack([_arr(i) for i in images]))


def imwrite(path, img, params=None):
    a = _arr(img)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim == 3 and a.shape[2] == 3:
        a = a[..., ::-1]
    Image.fromarray(np.ascontiguousarray(a)).save(path)
    return True


def imread(path, flags=IMREAD_COLOR):
    a = np.asarray(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(a[..., ::-1])


def rectangle(img, pt1, pt2, color, thickness=1):
    a = _arr(img)
    im = Image.fromarray(np.ascontiguousarray(a))
    outline = tuple(int(c) for c in color) if not np.isscalar(color) else int(color)
    ImageDraw.Draw(im).rectangle([tuple(map(int, pt1)), tuple(map(int, pt2))], outline=outline,
                                 width=max(1, int(thickness)))
    out = np.asarray(im)
    if isinstance(img, UMat):
        img._a = out.copy()
        return img
    if isinstance(img, np.ndarray) and img.flags.writeable:
        img[...] = out
        return img
    return out


def normalize(src, dst=None, alpha=0, beta=1, norm_type=NORM_MINMAX, dtype=-1):
    a = _arr(src).astype(np.float64)
    lo, hi = a.min(), a.max()
    out = (a - lo) / (hi - lo) * (beta - alpha) + alpha if hi > lo else np.full_like(a, alpha)
    return out.astype(np.float32 if dtype == CV_32F else _arr(src).dtype)
