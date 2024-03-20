# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import sys
import subprocess  # nosec
import shutil

def run_unit_test(src_path, ref_out_path, test_case, gpu, down_mix, build_folder_path):
    print("\n\n")
    result = subprocess.run([build_folder_path + "/build/rocal_audio_unittests", src_path, ref_out_path, str(test_case), str(gpu), str(down_mix)], stdout=subprocess.PIPE)    # nosec
    decoded_stdout = result.stdout.decode()

    # Count the occurrences of "PASSED" and "FAILED"
    num_passed = decoded_stdout.count("PASSED")
    num_failed = decoded_stdout.count("FAILED")
    print(result.stdout.decode())
    print("Number of PASSED:", num_passed)
    print("Number of FAILED:", num_failed)

    print("------------------------------------------------------------------------------------------")

def main():
    script_path = os.path.dirname(os.path.realpath(__file__))
    rocal_data_path = os.environ.get("ROCAL_DATA_PATH")
    if rocal_data_path is None:
        print("Need to export ROCAL_DATA_PATH")
        sys.exit()

    sys.dont_write_bytecode = True
    input_file_path = rocal_data_path + "/audio/wav"
    ref_out_path = rocal_data_path + "/GoldenOutputsTensor/reference_outputs_audio/audio_decoder_output.bin"
    test_case = 0
    build_folder_path = os.getcwd()

    if len(sys.argv) < 3:
            print("Please pass cpu/gpu(0/1) and down_mix(True/False)")
            exit(0)

    gpu = sys.argv[1]
    down_mix = sys.argv[2]

    # Enable extglob
    if os.path.exists(build_folder_path + "/build"):
        shutil.rmtree(build_folder_path + "/build")
    os.makedirs(build_folder_path + "/build")
    os.chdir(build_folder_path + "/build")

    # Run cmake and make commands
    subprocess.run(["cmake", script_path], cwd=".")   # nosec
    subprocess.run(["make", "-j16"], cwd=".")    # nosec

    run_unit_test(input_file_path, ref_out_path, test_case, gpu, down_mix, build_folder_path)

if __name__ == "__main__":
    main()