/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>

struct content_hash_t {
    std::array<std::uint32_t, 8> data{};

    friend bool operator==(const content_hash_t &a, const content_hash_t &b) noexcept {
        return a.data == b.data;
    }

    friend bool operator!=(const content_hash_t &a, const content_hash_t &b) noexcept {
        return !(a == b);
    }

    friend bool operator<(const content_hash_t &a, const content_hash_t &b) noexcept {
        return a.data < b.data;
    }
};

namespace content_hash_detail {

inline constexpr std::uint32_t rotl32(std::uint32_t x, std::uint8_t r) noexcept {
    return (x << r) | (x >> (32U - r));
}

inline std::uint32_t read_u32_unaligned(const std::uint8_t *ptr) noexcept {
    std::uint32_t word = 0;
    std::memcpy(&word, ptr, sizeof(word));
    return word;
}

inline std::uint64_t mix64(std::uint64_t x) noexcept {
    // MurmurHash3 64-bit finalization mix (public domain).
    x ^= x >> 33U;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33U;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33U;
    return x;
}

inline std::uint64_t to_u64(const content_hash_t &h) noexcept {
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 4; ++i) {
        const std::uint64_t lo = static_cast<std::uint64_t>(h.data[i * 2]);
        const std::uint64_t hi = static_cast<std::uint64_t>(h.data[i * 2 + 1]);
        const std::uint64_t chunk = lo | (hi << 32U);
        acc ^= mix64(chunk + (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(i + 1)));
        acc = mix64(acc);
    }
    return acc;
}

}  // namespace content_hash_detail

inline void content_hash(content_hash_t &hash, const void *data, std::size_t n) noexcept {
    // Inspired by xxHash3 (not identical).
    //
    // Produces a 256-bit, non-cryptographic content hash.
    // Entropy is spread across all 8 lanes by mixing each state word with a neighboring state word
    // (shifted by one). Each lane uses a different rotation, and a bias term helps avoid
    // low-entropy results for constant/zero inputs.
    constexpr std::uint32_t kPrime = 2246822519u;
    constexpr std::uint32_t kBias = 103456789u;
    constexpr std::array<std::uint8_t, 8> kBlockRot = {13, 16, 15, 17, 14, 18, 12, 19};

    const auto *data8 = static_cast<const std::uint8_t *>(data);

    std::size_t offset = 0;
    for (; offset + 32 <= n; offset += 32) {
        std::array<std::uint32_t, 8> words{};
        std::memcpy(words.data(), data8 + offset, 32);
        const content_hash_t prev = hash;

        for (int i = 0; i < 8; ++i) {
            const int prev_idx = (i + 7) & 7;
            hash.data[i] += (content_hash_detail::rotl32(prev.data[prev_idx], kBlockRot[i]) + words[i] + kBias) * kPrime;
        }
    }

    // Handle remaining 32-bit words.
    int word_idx = 0;
    for (; offset + 4 <= n; offset += 4, ++word_idx) {
        const std::uint32_t word = content_hash_detail::read_u32_unaligned(data8 + offset);
        hash.data[word_idx] += (content_hash_detail::rotl32(hash.data[(word_idx + 7) & 7], 13) + word + kBias) * kPrime;
    }

    // Mix the final bytes; include the tail length/position so even constant tails (including zeros)
    // affect the output.
    word_idx &= 7;
    std::uint32_t tail = 0xCCCCCCCCu + static_cast<std::uint32_t>(offset);
    for (; offset < n; ++offset) {
        tail = content_hash_detail::rotl32(tail, 17) + static_cast<std::uint32_t>(data8[offset]) * kPrime;
    }
    hash.data[word_idx] = (content_hash_detail::rotl32(hash.data[(word_idx + 7) & 7], 13) + tail) * kPrime;

    // Final avalanche so hashes diverge more, even when differences occur near the end
    // before mixing has propagated through all lanes.
    for (int i = 0; i < 8; ++i) {
        hash.data[i] += (hash.data[(i + 1) & 7] + kBias) * kPrime;
    }
    for (int i = 0; i < 8; ++i) {
        hash.data[i] += (hash.data[(i + 7) & 7] + kBias) * kPrime;
    }
}

namespace std {

template <>
struct hash<content_hash_t> {
    size_t operator()(const content_hash_t &h) const noexcept {
        const std::uint64_t v = content_hash_detail::to_u64(h);
        if constexpr (sizeof(size_t) == sizeof(std::uint64_t)) {
            return static_cast<size_t>(v);
        }
        return static_cast<size_t>(v ^ (v >> 32U));
    }
};

}  // namespace std
