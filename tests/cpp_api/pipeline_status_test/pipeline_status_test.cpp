/*
MIT License

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

// Pipeline status and ring buffer hand-off test.
//
// rocalRun() hands the caller the batch the output thread produced, and its return status has to
// tell three situations apart: a batch was produced, the pipeline ran out of data cleanly, and the
// output thread aborted. The checks below pin that contract down:
//
//   1. Every image is handed out exactly once per epoch. Each sample carries the name of the file
//      it came from, so a batch that was dropped, handed out twice, or left holding the previous
//      batch's contents shows up as a missing or repeated name rather than as a silent pass.
//   2. Running out of data is reported as exhaustion, not as a failure: the terminating rocalRun()
//      returns non-OK with an *empty* rocalGetErrorMessage(). rocalRun flattens every non-OK
//      pipeline status onto ROCAL_RUNTIME_ERROR, so that empty message is the only thing telling a
//      clean end of epoch apart from a graph failure.
//   3. rocalResetLoaders() starts a second epoch that hands out the same images again. This covers
//      the terminal end-of-data state the ring buffer latches when it drains: if that state were
//      not cleared on reset, the second epoch would report empty straight away.
//   4. None of the above deadlocks. The ring buffer hand-off is a two-thread rendezvous, so a
//      watchdog aborts with a message instead of letting a regression hang ctest indefinitely.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rocal_api.h"

namespace {

constexpr unsigned kBatchSize = 4;
constexpr int kWatchdogSeconds = 300;

int g_failures = 0;

void check(bool condition, const std::string &what) {
    std::cout << (condition ? "[  PASSED  ] " : "[  FAILED  ] ") << what << std::endl;
    if (!condition) g_failures++;
}

// Drains one epoch, returning the file name carried by every sample the pipeline handed out.
// Stops at the first non-OK rocalRun(), which is the call that reports the end of the epoch.
std::vector<std::string> drain_epoch(RocalContext handle, unsigned expected_batches,
                                     const std::string &label, RocalStatus *final_status) {
    std::vector<std::string> seen;
    std::vector<int> name_lengths(kBatchSize);
    // Allow a few more iterations than the pipeline should produce, so that a run() handing out
    // batches that were never written is caught by the count check rather than by spinning forever.
    const unsigned max_runs = expected_batches + 4;

    RocalStatus status = ROCAL_OK;
    for (unsigned i = 0; i < max_runs; i++) {
        status = rocalRun(handle);
        if (status != ROCAL_OK) break;

        // A run() that hands out a batch the output thread never produced leaves the metadata for
        // that slot unwritten, and reading it throws rather than returning a status. Report that as
        // a failed check instead of letting the exception take the process down.
        try {
            unsigned names_size = rocalGetImageNameLen(handle, name_lengths.data());
            std::vector<char> names(names_size + 1, '\0');
            rocalGetImageName(handle, names.data());

            int offset = 0;
            for (unsigned s = 0; s < kBatchSize; s++) {
                seen.emplace_back(names.data() + offset, name_lengths[s]);
                offset += name_lengths[s];
            }
        } catch (const std::exception &e) {
            check(false, label + ": rocalRun returned OK but the batch it handed out has no "
                                 "metadata: " + e.what());
            break;
        }
    }
    *final_status = status;
    return seen;
}

// Checks that an epoch handed out every image exactly once. Names are compared as a set, so the
// check holds whatever order the reader walks the folder in, while still failing on an image that
// was dropped, handed out twice, or replaced by a stale batch.
void check_epoch_contents(const std::vector<std::string> &seen, size_t expected_count,
                          const std::string &label) {
    check(seen.size() == expected_count,
          label + ": handed out " + std::to_string(seen.size()) + " samples, expected " +
              std::to_string(expected_count));
    const std::set<std::string> unique(seen.begin(), seen.end());
    check(unique.size() == seen.size(),
          label + ": every sample is a distinct image, none repeated or stale");
}

}  // namespace

int main(int argc, const char **argv) {
    if (argc < 2) {
        std::cout << "Usage: pipeline_status_test <image_dataset_folder - required> "
                     "<processing_device=1/cpu=0> <prefetch_queue_depth>\n";
        return -1;
    }
    const char *folder_path = argv[1];
    bool processing_device = false;
    size_t prefetch_queue_depth = 3;
    if (argc > 2) processing_device = (atoi(argv[2]) != 0);
    // A deeper queue lets the output thread run further ahead, so batches produced before the
    // pipeline stops are already buffered when the caller drains them.
    if (argc > 3) prefetch_queue_depth = static_cast<size_t>(atoi(argv[3]));

    std::cout << ">>> Running on " << (processing_device ? "GPU" : "CPU")
              << " with prefetch queue depth " << prefetch_queue_depth << std::endl;

    // The ring buffer hand-off is a rendezvous between the caller and the output thread, so a
    // regression there hangs rather than fails. Bound the whole run instead of letting ctest wait.
    std::atomic<bool> finished{false};
    std::thread watchdog([&finished]() {
        for (int elapsed = 0; elapsed < kWatchdogSeconds; elapsed++) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (finished.load()) return;
        }
        std::cerr << "[  FAILED  ] pipeline_status_test timed out after " << kWatchdogSeconds
                  << "s: the pipeline is deadlocked" << std::endl;
        std::abort();
    });
    // Leaves the watchdog running on the early returns below; the process exits either way.
    auto fail = [&finished, &watchdog](RocalContext handle, const std::string &message) {
        std::cerr << message << std::endl;
        if (handle) rocalRelease(handle);
        finished = true;
        watchdog.join();
        return -1;
    };

    RocalContext handle = rocalCreate(kBatchSize,
                                      processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU
                                                        : RocalProcessMode::ROCAL_PROCESS_CPU,
                                      0, 1, prefetch_queue_depth, ROCAL_FP32);
    if (handle == nullptr || rocalGetStatus(handle) != ROCAL_OK)
        return fail(nullptr, "Could not create the rocAL context");

    // loop=false, so the pipeline runs out of data at the end of the epoch instead of cycling.
    RocalTensor input = rocalJpegFileSource(handle, folder_path, RocalImageColor::ROCAL_COLOR_RGB24,
                                            1 /*shard count*/, false /*is_output*/, false /*loop*/);
    if (input == nullptr || rocalGetStatus(handle) != ROCAL_OK)
        return fail(handle, std::string("JPEG source could not initialize: ") + rocalGetErrorMessage(handle));

    // The label reader is what makes rocalGetImageName report which file each sample came from.
    rocalCreateLabelReader(handle, folder_path);

    // Any augmentation will do; this one just gives the graph a node to process.
    RocalTensor output = rocalBrightness(handle, input, true /*is_output*/);
    if (output == nullptr || rocalGetStatus(handle) != ROCAL_OK)
        return fail(handle, std::string("Could not add the brightness augmentation: ") + rocalGetErrorMessage(handle));

    if (rocalVerify(handle) != ROCAL_OK)
        return fail(handle, std::string("Could not verify the augmentation graph: ") + rocalGetErrorMessage(handle));

    const size_t image_count = rocalGetRemainingImages(handle);
    std::cout << "Dataset holds " << image_count << " images, batch size " << kBatchSize << std::endl;
    if (image_count == 0 || image_count % kBatchSize != 0)
        return fail(handle, "Dataset size must be a non-zero multiple of the batch size, so that "
                            "the last batch is not padded with repeated images");
    const unsigned expected_batches = static_cast<unsigned>(image_count / kBatchSize);

    // Epoch 1: every image is handed out once.
    RocalStatus epoch1_status = ROCAL_OK;
    const std::vector<std::string> epoch1 = drain_epoch(handle, expected_batches, "epoch 1", &epoch1_status);
    check_epoch_contents(epoch1, image_count, "epoch 1");

    check(epoch1_status != ROCAL_OK, "epoch 1: the run after the last batch reports non-OK");
    const char *epoch1_error = rocalGetErrorMessage(handle);
    const bool epoch1_error_empty = (epoch1_error == nullptr || epoch1_error[0] == '\0');
    if (!epoch1_error_empty)
        std::cout << "          unexpected error message: " << epoch1_error << std::endl;
    check(epoch1_error_empty, "epoch 1: running out of data is not reported as a graph failure");
    check(rocalIsEmpty(handle) != 0, "epoch 1: the pipeline reports itself empty once drained");

    // Epoch 2: resetting the loaders clears the end-of-data state and replays the same images.
    check(rocalResetLoaders(handle) == ROCAL_OK, "rocalResetLoaders succeeds after exhaustion");
    RocalStatus epoch2_status = ROCAL_OK;
    const std::vector<std::string> epoch2 = drain_epoch(handle, expected_batches, "epoch 2", &epoch2_status);
    check_epoch_contents(epoch2, image_count, "epoch 2");
    check(epoch2_status != ROCAL_OK, "epoch 2: the run after the last batch reports non-OK");
    const char *epoch2_error = rocalGetErrorMessage(handle);
    const bool epoch2_error_empty = (epoch2_error == nullptr || epoch2_error[0] == '\0');
    if (!epoch2_error_empty)
        std::cout << "          unexpected error message: " << epoch2_error << std::endl;
    check(epoch2_error_empty, "epoch 2: running out of data is not reported as a graph failure");

    std::set<std::string> epoch1_names(epoch1.begin(), epoch1.end());
    std::set<std::string> epoch2_names(epoch2.begin(), epoch2.end());
    check(epoch1_names == epoch2_names, "the second epoch hands out the same images as the first");

    rocalRelease(handle);
    finished = true;
    watchdog.join();

    if (g_failures != 0) {
        std::cout << "pipeline_status_test: " << g_failures << " check(s) failed" << std::endl;
        return -1;
    }
    std::cout << "pipeline_status_test: all checks passed" << std::endl;
    return 0;
}
