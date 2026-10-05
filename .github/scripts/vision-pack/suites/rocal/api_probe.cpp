/*
Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

// rocAL C API probes, one case per process (run through common.run_child):
//   api_probe nop <cpu|gpu>          rocalNop must return the input unchanged (same as rocalCopy)
//   api_probe copy-size <cpu|gpu>    rocalCopyToOutput with a too-small buffer must fail (M9)
//   api_probe link-only              a program that only calls rocalCreate (used by the H1 link probe)
// Prints one line: RESULT {"status": ..., "message": ...}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rocal_api.h"

static void result(const char *status, const std::string &msg) {
    std::string m;
    for (char c : msg) m += (c == '"' || c == '\\') ? ' ' : c;
    printf("RESULT {\"status\": \"%s\", \"message\": \"%s\"}\n", status, m.c_str());
    fflush(stdout);
}

static RocalContext coco_pipeline(bool gpu, bool nop, int *h, int *w, int *p) {
    const char *env = std::getenv("ROCAL_DATA_PATH");
    std::string data = env ? env : "";
    std::string imgs = data + "/rocal_data/coco/coco_10_img/images/";
    std::string json = data + "/rocal_data/coco/coco_10_img/annotations/coco_data.json";
    auto handle = rocalCreate(2, gpu ? ROCAL_PROCESS_GPU : ROCAL_PROCESS_CPU, 0, 1);
    if (rocalGetStatus(handle) != ROCAL_OK) return nullptr;
    rocalSetSeed(0);
    rocalCreateCOCOReader(handle, json.c_str(), true);
    RocalTensor in = rocalJpegCOCOFileSource(handle, imgs.c_str(), json.c_str(), ROCAL_COLOR_RGB24, 1, false, false,
                                             false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, 416, 416);
    if (nop)
        rocalNop(handle, in, true);
    else
        rocalCopy(handle, in, true);
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK) return nullptr;
    *h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * 2;
    *w = rocalGetOutputWidth(handle);
    int cf = rocalGetOutputColorFormat(handle);
    *p = (cf == 0 || cf == 1 || cf == 3) ? 3 : 1;
    return handle;
}

static int run_nop(bool gpu) {
    int h, w, p;
    std::vector<unsigned char> out[2];
    for (int k = 0; k < 2; k++) {
        auto handle = coco_pipeline(gpu, k == 0, &h, &w, &p);
        if (!handle) { result("fail", "pipeline creation/verify failed"); return 0; }
        if (rocalRun(handle) != 0) { result("fail", "rocalRun failed"); return 0; }
        out[k].assign(static_cast<size_t>(h) * w * p, 0x5A);
        rocalCopyToOutput(handle, out[k].data(), out[k].size());
        rocalRelease(handle);
    }
    size_t diff = 0;
    for (size_t i = 0; i < out[0].size(); i++) diff += out[0][i] != out[1][i];
    char msg[256];
    snprintf(msg, sizeof msg, "rocalNop vs rocalCopy of the same batch: %zu of %zu bytes differ", diff, out[0].size());
    result(diff == 0 ? "pass" : "fail", std::string(msg) + (diff ? " (rocalNop returns uninitialized memory)" : ""));
    return 0;
}

static int run_copy_size(bool gpu) {
    int h, w, p;
    auto handle = coco_pipeline(gpu, false, &h, &w, &p);
    if (!handle) { result("fail", "pipeline creation/verify failed"); return 0; }
    if (rocalRun(handle) != 0) { result("fail", "rocalRun failed"); return 0; }
    size_t need = static_cast<size_t>(h) * w * p;
    std::vector<unsigned char> buf(need, 0xAB);
    RocalStatus st = rocalCopyToOutput(handle, buf.data(), need / 2);
    size_t touched = 0;
    for (size_t i = 0; i < need / 2; i++) touched += buf[i] != 0xAB;
    rocalRelease(handle);
    char msg[256];
    snprintf(msg, sizeof msg, "rocalCopyToOutput(size=%zu, needs %zu) -> status %d, %zu bytes written", need / 2,
             need, static_cast<int>(st), touched);
    if (st != ROCAL_OK)
        result("pass", msg);
    else if (touched == 0)
        result("fail", std::string(msg) + " (M9: returns OK without copying)");
    else
        result("pass", std::string(msg) + " (partial copy)");
    return 0;
}

int main(int argc, char **argv) {
    std::string what = argc > 1 ? argv[1] : "";
    bool gpu = argc > 2 && std::strcmp(argv[2], "gpu") == 0;
    if (what == "nop") return run_nop(gpu);
    if (what == "copy-size") return run_copy_size(gpu);
    if (what == "link-only") {
        auto handle = rocalCreate(1, ROCAL_PROCESS_CPU, 0, 1);
        result(rocalGetStatus(handle) == ROCAL_OK ? "pass" : "fail", "rocalCreate");
        return 0;
    }
    fprintf(stderr, "usage: api_probe nop|copy-size <cpu|gpu> | link-only\n");
    return 2;
}
