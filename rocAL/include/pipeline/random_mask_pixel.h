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
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "pipeline/tensor.h"

/*! \brief Selects a random pixel from each segmentation mask in a batch.
 *
 * Depending on configuration, selects either a uniformly random pixel or
 * a foreground pixel (matching or exceeding a threshold value). Used by
 * the PixelwiseMask metadata path.
 */
class RandomMaskPixel {
   public:
    RandomMaskPixel(size_t user_batch_size, size_t cpu_num_threads);

    /*! \brief Configure foreground selection behaviour.
     *  \param [in] is_foreground  If true, restrict selection to foreground pixels.
     *  \param [in] value          The label value (or threshold) defining foreground.
     *  \param [in] is_threshold   If true, foreground means pixel > value; otherwise pixel == value.
     */
    void set_config(bool is_foreground, int value, bool is_threshold);

    /*! \brief Run random pixel selection on a batch of masks.
     *  \param [in]  input       Batch of 2D int32 segmentation masks.
     *  \param [out] output_list TensorList to populate with (row, col) per sample.
     *  \return Pointer to output_list after population.
     */
    TensorList *run(rocalTensorList *input, TensorList &output_list);

   private:
    /// Lazy-initialize per-sample RNGs with ParameterFactory seed + "MPIX" salt.
    void ensure_rngs();
    /// Binary-search helper: given run-length encoded foreground spans, find the flat pixel index
    /// corresponding to the val-th foreground pixel. Returns -1 on invalid input.
    int64_t find_pixel(const std::vector<int> &start, const std::vector<int> &foreground_count, int64_t val, int count);

    size_t _user_batch_size;                    ///< Number of samples per batch
    size_t _cpu_num_threads;                    ///< Number of CPU threads for OMP parallelism
    unsigned _rng_seed = 0;                     ///< Cached seed to detect when RNGs need re-seeding
    std::vector<std::mt19937> _rngs;            ///< Per-sample Mersenne Twister RNGs
    std::vector<unsigned> _output_buffer;       ///< Flat buffer holding (row, col) pairs for the batch
    int _pixel_value = 0;                       ///< Label value (or threshold) defining foreground
    bool _is_foreground = false;                ///< If true, restrict pixel selection to foreground
    bool _is_threshold = false;                 ///< If true, foreground = pixel > _pixel_value; else pixel == _pixel_value
};
