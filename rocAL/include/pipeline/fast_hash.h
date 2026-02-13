/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>

struct fast_hash_t {
    uint32_t data[8];

    bool operator==(const fast_hash_t &other) const {
        for (int i = 0; i < 8; i++) {
            if (data[i] != other.data[i])
                return false;
        }
        return true;
    }
    bool operator!=(const fast_hash_t &other) const {
        return !(*this == other);
    }
    bool operator<(const fast_hash_t &other) const {
        for (int i = 0; i < 8; i++) {
            if (data[i] < other.data[i])
                return true;
            if (data[i] > other.data[i])
                return false;
        }
        return false;
    }
};

inline
constexpr uint32_t rotl(uint32_t x, uint8_t r) {
    return (x << r) | (x >> (32 - r));
}

inline void fast_hash(fast_hash_t &hash, const void *data, size_t n) {
    // Loosely based on xxHash 3.
    //
    // This function computes a 256-bit non-cryptographic hash.
    // The entropy is diffused across the whole hash (unlike xxHash) by combining a state entry
    // with another state entry (shifted by 1 word). The bit rotations are different for each
    // entry and a bias is added to avoid lack of entropy in constant 0 inputs.
    const uint8_t *data8 = static_cast<const uint8_t *>(data);
    constexpr uint32_t prime = 2246822519u;  // prime
    constexpr uint32_t bias = 103456789u;    // another prime

    size_t offset = 0;
    for (; offset + 32 <= n; offset += 32) {
        const uint32_t *data32 = reinterpret_cast<const uint32_t *>(data8 + offset);
        fast_hash_t prev = hash;

        // Hash entries are cycled (see index at prev.data) in order to distribute.
        // changes that have a period matching the hash size.
        // Each hash entry is rotated by a different bit count to make hash more sensitive
        // to position of changes.
        hash.data[0] += (rotl(prev.data[7], 13) + data32[0] + bias) * prime;
        hash.data[1] += (rotl(prev.data[0], 16) + data32[1] + bias) * prime;
        hash.data[2] += (rotl(prev.data[1], 15) + data32[2] + bias) * prime;
        hash.data[3] += (rotl(prev.data[2], 17) + data32[3] + bias) * prime;
        hash.data[4] += (rotl(prev.data[3], 14) + data32[4] + bias) * prime;
        hash.data[5] += (rotl(prev.data[4], 18) + data32[5] + bias) * prime;
        hash.data[6] += (rotl(prev.data[5], 12) + data32[6] + bias) * prime;
        hash.data[7] += (rotl(prev.data[6], 19) + data32[7] + bias) * prime;
    }
    // process trailing 32-bit words
    int k = 0;
    for (; offset + 4 <= n; offset += 4, k++) {
        const uint32_t *data32 = reinterpret_cast<const uint32_t *>(data8 + offset);
        hash.data[k] += (rotl(hash.data[(k - 1) & 7], 13) + *data32 + bias) * prime;
    }
    // Process trailing bytes - make sure that the number of trailing constant values (including 0)
    // changes the output.
    uint32_t tail = 0xCCCCCCCCu + offset;
    k &= 7;
    for (; offset < n; offset++) {
        tail = rotl(tail, 17) + data8[offset] * prime;
    }
    hash.data[k] = (rotl(hash.data[(k - 1) & 7], 13) + tail) * prime;

    // Final diffusion, to make hashes visually different even if the data differs near the end
    // and the changes were not cycled through the whole hash yet.
    for (int k = 0; k < 8; k++) {
        hash.data[k] += (hash.data[(k + 1) & 7] + bias) * prime;
    }
    for (int k = 0; k < 8; k++) {
        hash.data[k] += (hash.data[(k - 1) & 7] + bias) * prime;
    }
}

namespace std {

template<>
struct hash<fast_hash_t> {
  size_t operator()(const fast_hash_t &h) const {
    return h.data[0];
  }
};

}  // namespace std
