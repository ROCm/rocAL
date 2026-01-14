#include <random>
#include <bits/stdc++.h>


#pragma once
// todo:: move this to common header
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
