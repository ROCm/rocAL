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
#include <dirent.h>
#include <memory>
#include "commons.h"
#include "sndfile_decoder.h"
#include "reader_factory.h"
#include "timing_debug.h"
#include "loader_module.h"
#include "audio_decoder.h"

class AudioReadAndDecode {
public:
    AudioReadAndDecode();
    ~AudioReadAndDecode();
    size_t count();
    void reset();
    void create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size, int device_id = 0);
    //! Loads a decompressed batch of audios into the buffer indicated by buff
    /// \param buff User's buffer provided to be filled with decoded audio samples
    /// \param names User's buffer provided to be filled with name of the audio files
    /// \param max_decoded_samples User's buffer maximum samples per decoded audio. User expects the decoder to downscale the audio if audio's original samples is bigger than max_samples
    /// \param max_decoded_channels user's buffer maximum channels per decoded audio. User expects the decoder to downscale the audio if audio's original channels is bigger than max_channels
    /// \param roi_samples is set by the load() function to the samples of the region that decoded audio is located. It's less than max_samples and is either equal to the original audio samples if original audio samples is smaller than max_samples or downscaled if necessary to fit the max_samples criterion.
    /// \param roi_channels  is set by the load() function to the samples of the region that decoded audio is located.It's less than max_channels and is either equal to the original audio channels if original audio channels is smaller than max_channels or downscaled if necessary to fit the max_channels criterion.
    LoaderModuleStatus load(
            float* buff,
            std::vector<std::string>& names,
            const size_t  max_decoded_samples,
            const size_t max_decoded_channels,
            std::vector<uint32_t> &actual_samples,
            std::vector<uint32_t> &actual_channels,
            std::vector<float> &actual_sample_rates);
    //! returns timing info or other status information
    Timing timing();
private:
    std::vector<std::shared_ptr<AudioDecoder>> _decoder;
    std::shared_ptr<Reader> _reader;
    std::vector<std::string> _audio_names;
    std::vector<std::string> _audio_file_path;
    std::vector<float*> _decompressed_buff_ptrs;
    std::vector<size_t> _actual_decoded_samples;
    std::vector<size_t> _actual_decoded_channels;
    std::vector<size_t> _original_samples;
    std::vector<size_t> _original_channels;
    std::vector<float> _original_sample_rates;
    TimingDBG _file_load_time, _decode_time;
    size_t _batch_size, _num_threads;
    DecoderConfig _decoder_config;
    std::string _input_path;
};
