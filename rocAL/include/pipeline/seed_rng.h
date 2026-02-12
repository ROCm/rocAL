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

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

template<typename T = std::mt19937, std::size_t state_size = T::state_size>
class SeededRNG {
  /*
  * @param batch_size How many RNGs to store
  * @param state_size How many seed are used to initialize one RNG. Used to lower probablity of
  * collisions between seeds used to initialize RNGs in different operators.
  */
public:
  SeededRNG (int batch_size = 128) {
      std::random_device source;
      _batch_size = batch_size;
      std::size_t _random_data_size = state_size * batch_size ;
      std::vector<std::random_device::result_type> random_data(_random_data_size);
      std::generate(random_data.begin(), random_data.end(), std::ref(source));
      _rngs.reserve(batch_size);
      for (int i=0; i < (int)(_batch_size*state_size); i += state_size) {
        std::seed_seq seeds(std::begin(random_data) + i, std::begin(random_data)+ i +state_size);
        _rngs.emplace_back(T(seeds));
      }
  }

  /**
   * Returns engine corresponding to given sample ID
   */
   T &operator[](int sample) noexcept {
    return _rngs[sample % _batch_size];
  }

private:
    std::vector<T> _rngs;
    int _batch_size;
};
enum RandomObjectBBoxFormat
{
    OUT_BOX = 0,
    OUT_ANCHORSHAPE = 1,
    OUT_STARTEND = 2,
};
