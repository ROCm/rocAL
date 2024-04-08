/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "audio_decoder.h"
#include "commons.h"
#include "loader_module.h"
#include "reader_factory.h"
#include "generic_audio_decoder.h"
#include "timing_debug.h"

#ifdef ROCAL_AUDIO

// Contains all the meta info for the audio file
struct AudioMetaInfo {
    std::string file_name;  // Name of audio file
    std::string file_path;  // Absolute path to the audio file
    size_t samples;
    size_t channels;
    float sample_rate;
};

class AudioReadAndDecode {
   public:
    AudioReadAndDecode();
    ~AudioReadAndDecode();
    size_t Count();
    void Reset();
    void Create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size, int device_id = 0);
    //! Loads a decompressed batch of audios into the buffer indicated by buff
    /// \param buff User's buffer provided to be filled with decoded audio samples
    /// \param names User's buffer provided to be filled with name of the audio files
    /// \param max_decoded_samples User's buffer maximum samples per decoded audio.
    /// \param max_decoded_channels user's buffer maximum channels per decoded audio.
    /// \param roi_samples is set by the load() function to the samples of the region that decoded audio is located. It's less than max_samples and is either equal to the original audio samples if original audio samples is smaller than max_samples.
    /// \param roi_channels  is set by the load() function to the channels of the region that decoded audio is located. It's less than max_channels and is either equal to the original audio channels if original audio channels is smaller than max_channels.
    /// \param original_sample_rates is set by the load() function to the original sample_rates of the decoded audio samples.
    LoaderModuleStatus Load(
        float *buff,
        std::vector<std::string> &names,
        const size_t max_decoded_samples,
        const size_t max_decoded_channels,
        std::vector<uint32_t> &roi_samples,
        std::vector<uint32_t> &roi_channels,
        std::vector<float> &original_sample_rates);
    //! returns timing info or other status information
    Timing GetTiming();
    size_t last_batch_padded_size();

   private:
    std::vector<std::shared_ptr<AudioDecoder>> _decoder;
    std::shared_ptr<Reader> _reader;
    std::vector<float *> _decompressed_buff_ptrs;
    std::vector<AudioMetaInfo> _audio_meta_info;
    TimingDbg _file_load_time, _decode_time;
    size_t _batch_size, _num_threads;
    DecoderConfig _decoder_config;
};
#endif
