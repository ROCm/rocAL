#!/usr/bin/env bash
# Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# rocal suite: rocAL C++ and Python API.
#
#   quick          import smoke, a tiny pipeline per backend, 3 ctests
#   standard       + all 21 ctests (serial, timeouts), a rerun of the VA-API tests without
#                  the VA env, the pybind ctest (6) and its N3 check, audio QA, H1 link probe
#   comprehensive  + C++ golden sweep (upstream rules and strict), Python golden sweep,
#                  readers matrix, API probes, C++ API probes, LMDB probe
#   full           + perf (performance_tests, dataloader_multithread) and the
#                  torch/jax/tensorflow plugins (VP_EXTENDED=1, otherwise blocked)
#
# Suite-local knobs (developer use):
#   VP_ROCAL_ONLY=part,...   run only these parts: smoke ctest pybind audio golden-cpp
#                            golden-py readers probes lmdb perf extended
#   VP_ROCAL_CV2_SHIM=1      bare hosts without python3-opencv: use devshim/cv2.py for the
#                            Python golden sweep (never in a container; noted in every result)
set -uo pipefail

VP_REPO="${VP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${VP_REPO}/build_tools/lib/vp.sh"
vp_init rocal

SUITE="${VP_SUITE_DIR}"
TEST="${ROCM_PATH}/share/rocal/test"
SYSDEPS="${ROCM_PATH}/lib/rocm_sysdeps/lib"
export VP_CTEST_TIMEOUT="${VP_CTEST_TIMEOUT:-600}"

part() { [[ -z "${VP_ROCAL_ONLY:-}" || ",${VP_ROCAL_ONLY}," == *",$1,"* ]]; }
py() { "${VP_PY}" "$@"; }

# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
if [[ ! -f "${ROCM_PATH}/lib/librocal.so" || ! -d "${TEST}" ]]; then
  vp_result env::prefix error "librocal.so or share/rocal/test missing under ${ROCM_PATH}"
  vp_finish
  exit 0
fi

# rocAL's own CI runs with TheRock's VA-API driver and libraries (rocJPEG, rocDecode).
va_libs=("${SYSDEPS}/librocm_sysdeps_va.so.2" "${SYSDEPS}/librocm_sysdeps_va-drm.so.2")
if [[ -f "${va_libs[0]}" && -f "${va_libs[1]}" && -f "${SYSDEPS}/radeonsi_drv_video.so" ]]; then
  vp_result env::va-libs pass "LIBVA_DRIVERS_PATH=${SYSDEPS}; LD_PRELOAD=${va_libs[*]}"
  export LIBVA_DRIVERS_PATH="${SYSDEPS}"
  export LD_PRELOAD="${va_libs[0]}:${va_libs[1]}"
else
  vp_result env::va-libs fail "missing VA libraries/driver in ${SYSDEPS}: ${va_libs[*]} radeonsi_drv_video.so"
fi

py_ver="$(py -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)"
PY_OK=0
if [[ "${py_ver}" == 3.12 ]]; then
  PY_OK=1
  vp_result env::python312 pass "${VP_PY} is Python ${py_ver} (the rocAL modules are cp312-only)"
else
  vp_result env::python312 blocked "${VP_PY} is Python ${py_ver}; the rocAL modules are built for 3.12 only"
fi

CV2_MODE=none
if py -c 'import cv2' >/dev/null 2>&1; then
  CV2_MODE=real
  vp_result env::python-cv2 pass "cv2 $(py -c 'import cv2; print(cv2.__version__)')"
elif [[ "${VP_ROCAL_CV2_SHIM:-0}" == 1 && "${VP_NO_CONTAINER:-0}" == 1 ]]; then
  CV2_MODE=shim
  vp_result env::python-cv2 blocked "python3-opencv missing; DEV-ONLY devshim/cv2.py in use (VP_ROCAL_CV2_SHIM=1)"
else
  vp_result env::python-cv2 blocked "python3-opencv missing: the Python golden sweep is blocked"
fi

HAVE_DATA=0
if [[ -n "${VP_DATA}" && -d "${VP_DATA}/rocal_data" ]]; then
  HAVE_DATA=1
  export VP_ROCAL_LMDB="${VP_WORK}/lmdb"
  vp_copy_lmdb "${VP_ROCAL_LMDB}"
fi
vp_crash_tests_allowed && export VP_ROCAL_CRASH_OK=1

cxx="${ROCM_PATH}/lib/llvm/bin/amdclang++"
[[ -x "${cxx}" ]] || cxx="$(command -v g++ || echo c++)"
rocal_flags=(-std=c++17 -O1 -I"${ROCM_PATH}/include" -I"${ROCM_PATH}/include/rocal" -L"${ROCM_PATH}/lib"
             "-Wl,-rpath,${ROCM_PATH}/lib" -lrocal)

# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------
run_smoke() {
  (cd "${SUITE}" && py probes.py smoke)
}

run_ctest() {
  local b="${VP_WORK}/ctest"
  if ! vp_cmake_build build::ctest "${TEST}" "${b}"; then
    return
  fi
  if vp_tier_ge standard; then
    vp_ctest ctest "${b}"
    # One rerun of the VA-API users without TheRock's VA env: catches documentation gaps.
    (unset LD_PRELOAD LIBVA_DRIVERS_PATH; vp_ctest ctest-novaenv "${b}" -R rocjpeg)
    (unset LD_PRELOAD LIBVA_DRIVERS_PATH; cd "${SUITE}" && py probes.py rocjpeg --group probe-novaenv)
  else
    vp_ctest ctest "${b}" -R '^(basic_test_cpu|basic_test_gpu|basic_test_rocjpeg_rgb)$'
  fi
}

run_pybind() {
  local b="${VP_WORK}/pybind"
  vp_cmake_build build::pybind "${TEST}/pybind" "${b}" || return
  vp_ctest pybind "${b}"
  # N3: the test CMake sets ENVIRONMENT "PYTHONPATH=<rocm>/lib:$PYTHONPATH"; CMake keeps
  # $PYTHONPATH literal, so the caller's PYTHONPATH is replaced, not extended.
  local envs
  envs="$(cd "${b}" && ctest --show-only=json-v1 2>/dev/null \
    | jq -r '.tests[].properties[]? | select(.name == "ENVIRONMENT") | .value[]' 2>/dev/null)"
  if [[ -z "${envs}" ]]; then
    vp_result env::pybind-pythonpath error "could not read the pybind tests' ENVIRONMENT (ctest --show-only=json-v1)"
  elif grep -qF '$PYTHONPATH' <<<"${envs}"; then
    vp_result env::pybind-pythonpath fail \
      "N3: pybind tests set a literal '$(head -1 <<<"${envs}")', which replaces the caller's PYTHONPATH"
  else
    vp_result env::pybind-pythonpath pass "ENVIRONMENT: $(head -1 <<<"${envs}")"
  fi
}

run_audio() {
  if [[ "${HAVE_DATA}" != 1 ]]; then
    vp_blocked audio::dataset "rocal_data not available (VP_DATA=${VP_DATA:-unset})"
    return
  fi
  (cd "${SUITE}" && py audio.py python)
  local b="${VP_WORK}/audio-build"
  vp_cmake_build build::audio_tests "${TEST}/audio_tests" "${b}"
  (cd "${SUITE}" && py audio.py cpp --bin "${b}/audio_tests")
  (cd "${SUITE}" && py audio.py nontorch)
}

run_h1() {
  local d="${VP_WORK}/h1"
  mkdir -p "${d}"
  # H1: a customer C++ program that links only librocal must link and run.
  vp_run probe::h1-link-without-libpython --timeout 300 -- \
    bash -c "$(printf '%q ' "${cxx}" "${SUITE}/api_probe.cpp" -o "${d}/link_only" "${rocal_flags[@]}") && ${d}/link_only link-only"
}

build_api_probe() {
  local d="${VP_WORK}/api-probe" libdir
  mkdir -p "${d}"
  libdir="$(py -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
  vp_run build::api_probe --timeout 600 -- "${cxx}" "${SUITE}/api_probe.cpp" -o "${d}/api_probe" \
    "${rocal_flags[@]}" -L"${libdir}" "-lpython${py_ver}"
}

# Replays testAllScripts.sh with a recording ./unit_tests; see ut_sweep.py.
run_golden_cpp() {
  local w="${VP_WORK}/golden-cpp" src="${TEST}/unit_tests" dev d reason=""
  local golden="${VP_DATA}/rocal_data/GoldenOutputsTensor/"
  mkdir -p "${w}/compare"
  if ! py "${SUITE}/ut_sweep.py" sanitize --kind cpp --src "${src}/testAllScripts.sh" --out "${w}/driver.sh" \
      --out-dir "${w}/out" >"${VP_OUT}/logs/golden-cpp.sanitize.log" 2>&1; then
    vp_result golden-cpp::driver error "could not sanitize testAllScripts.sh: $(tail -1 "${VP_OUT}/logs/golden-cpp.sanitize.log")"
    return
  fi
  vp_result golden-cpp::driver pass "$(tail -1 "${VP_OUT}/logs/golden-cpp.sanitize.log")"
  for dev in host hip; do
    d=0; [[ "${dev}" == hip ]] && d=1
    py "${SUITE}/ut_sweep.py" wrapper "${w}/run-${dev}" "${w}/run-${dev}/dry"
    (cd "${w}/run-${dev}/dry" && VP_UT_DIR="${PWD}" VP_UT_DRY=1 VP_UT_LMDB="${VP_ROCAL_LMDB:-/nonexistent}" \
      ROCAL_DATA_PATH="${VP_DATA:-/nonexistent}" bash "${w}/driver.sh" "${d}" 2 >dry.log 2>&1)
  done
  [[ "${HAVE_DATA}" == 1 ]] || reason="rocal_data not available (VP_DATA=${VP_DATA:-unset})"
  if [[ -z "${reason}" ]] && ! vp_cmake_build build::unit_tests "${src}" "${w}/build"; then
    reason="unit_tests did not build"
  fi
  if [[ -z "${reason}" ]]; then
    if grep -q -- '-DENABLE_OPENCV=1' "${w}/build/build.ninja" 2>/dev/null; then
      vp_result golden-cpp::unit_tests-opencv pass "unit_tests built with ENABLE_OPENCV=1 (it writes PNGs)"
    else
      vp_result golden-cpp::unit_tests-opencv error \
        "unit_tests built without OpenCV: it writes no PNGs, so the golden check would pass vacuously (install libopencv-dev)"
      reason="unit_tests built without OpenCV"
    fi
  fi
  if [[ -n "${reason}" ]]; then
    py "${SUITE}/ut_sweep.py" blocked --kind cpp --strict --runs "${w}/run-host" "${w}/run-hip" --reason "${reason}"
    return
  fi
  for dev in host hip; do
    d=0; [[ "${dev}" == hip ]] && d=1
    (cd "${w}/run-${dev}" && VP_UT_DIR="${PWD}" VP_UT_BIN="${w}/build/unit_tests" VP_UT_LMDB="${VP_ROCAL_LMDB}" \
      VP_UT_TIMEOUT=300 timeout -k 30 "$(vp__scale_timeout 3600)" bash "${w}/driver.sh" "${d}" 2 >driver.log 2>&1)
    cp "${w}/run-${dev}/driver.log" "${VP_OUT}/logs/golden-cpp.${dev}.driver.log"
  done
  local crc=0
  (cd "${w}/compare" && timeout -k 30 1200 "${VP_PY}" "${src}/pixel_comparison/image_comparison.py" "${golden}" \
    "${w}/out/" >comparator.log 2>&1) || crc=$?
  cp "${w}/compare/comparator.log" "${VP_OUT}/logs/golden-cpp.comparator.log"
  py "${SUITE}/ut_sweep.py" classify-cpp --runs "${w}/run-host" "${w}/run-hip" --golden "${golden}" \
    --comparator-log "${VP_OUT}/logs/golden-cpp.comparator.log" --comparator-rc "${crc}"
}

# Replays python_api/unit_tests.sh through a recording python3.12 function.
run_golden_py() {
  local w="${VP_WORK}/golden-py" src="${TEST}/python_api" dev d reason="" shim=""
  local golden="${VP_DATA}/rocal_data/GoldenOutputsTensor/"
  mkdir -p "${w}/compare"
  if ! py "${SUITE}/ut_sweep.py" sanitize --kind py --src "${src}/unit_tests.sh" --out "${w}/driver.sh" \
      --out-dir "${w}/out" >"${VP_OUT}/logs/golden-py.sanitize.log" 2>&1; then
    vp_result golden-py::driver error "could not sanitize unit_tests.sh: $(tail -1 "${VP_OUT}/logs/golden-py.sanitize.log")"
    return
  fi
  vp_result golden-py::driver pass "$(tail -1 "${VP_OUT}/logs/golden-py.sanitize.log")"
  for dev in host hip; do
    d=0; [[ "${dev}" == hip ]] && d=1
    py "${SUITE}/ut_sweep.py" wrapper "${w}/run-${dev}" "${w}/run-${dev}/dry"
    (cd "${w}/run-${dev}/dry" && VP_UT_DIR="${PWD}" VP_UT_DRY=1 VP_UT_LMDB="${VP_ROCAL_LMDB:-/nonexistent}" \
      ROCAL_DATA_PATH="${VP_DATA:-/nonexistent}" bash "${w}/driver.sh" "${d}" 2 >dry.log 2>&1)
  done
  [[ "${HAVE_DATA}" == 1 ]] || reason="rocal_data not available (VP_DATA=${VP_DATA:-unset})"
  [[ "${PY_OK}" == 1 ]] || reason="${reason:-${VP_PY} is not Python 3.12}"
  case "${CV2_MODE}" in
    real) ;;
    shim) shim="${SUITE}/devshim"; export VP_UT_NOTE="dev-only cv2 shim (VP_ROCAL_CV2_SHIM=1), not OpenCV" ;;
    *) reason="${reason:-python3-opencv (cv2) is not installed; unit_test.py writes its PNGs with cv2}" ;;
  esac
  if [[ -n "${reason}" ]]; then
    py "${SUITE}/ut_sweep.py" blocked --kind py --runs "${w}/run-host" "${w}/run-hip" --reason "${reason}"
    return
  fi
  for dev in host hip; do
    d=0; [[ "${dev}" == hip ]] && d=1
    # unit_test.py writes output_folder/ into its CWD: run it from a fresh copy.
    cp -r "${src}" "${w}/api-${dev}"
    chmod -R u+w "${w}/api-${dev}"
    (cd "${w}/api-${dev}" && VP_UT_DIR="${w}/run-${dev}" VP_UT_PY="${VP_PY}" VP_UT_LMDB="${VP_ROCAL_LMDB}" \
      VP_UT_TIMEOUT=300 PYTHONPATH="${shim:+${shim}:}${PYTHONPATH}" \
      timeout -k 30 "$(vp__scale_timeout 3600)" bash "${w}/driver.sh" "${d}" 2 >"${w}/run-${dev}/driver.log" 2>&1)
    cp "${w}/run-${dev}/driver.log" "${VP_OUT}/logs/golden-py.${dev}.driver.log"
  done
  local crc=0
  (cd "${w}/compare" && timeout -k 30 1200 "${VP_PY}" "${src}/image_comparison.py" "${golden}" "${w}/out/" \
    >comparator.log 2>&1) || crc=$?
  cp "${w}/compare/comparator.log" "${VP_OUT}/logs/golden-py.comparator.log"
  py "${SUITE}/ut_sweep.py" classify-py --runs "${w}/run-host" "${w}/run-hip" --golden "${golden}" \
    --comparator-log "${VP_OUT}/logs/golden-py.comparator.log" --comparator-rc "${crc}"
  unset VP_UT_NOTE
}

run_readers() {
  if [[ "${HAVE_DATA}" != 1 ]]; then
    vp_blocked readers::dataset "rocal_data not available (VP_DATA=${VP_DATA:-unset})"
    return
  fi
  (cd "${SUITE}" && py readers.py all)
}

run_probes() {
  if [[ "${HAVE_DATA}" != 1 ]]; then
    vp_blocked probe::dataset "rocal_data not available (VP_DATA=${VP_DATA:-unset})"
    return
  fi
  (cd "${SUITE}" && py probes.py rocjpeg --group probe && py probes.py all)
  build_api_probe
  (cd "${SUITE}" && py probes.py cpp --bin "${VP_WORK}/api-probe/api_probe")
  if [[ -x "${VP_WORK}/golden-cpp/build/unit_tests" ]]; then
    (cd "${SUITE}" && py probes.py box-encoder --bin "${VP_WORK}/golden-cpp/build/unit_tests")
  else
    vp_blocked probe::box-encoder-release "unit_tests was not built (the C++ golden sweep did not run)"
  fi
}

run_lmdb() {
  if [[ "${HAVE_DATA}" != 1 ]]; then
    vp_blocked lmdb::dataset "rocal_data not available (VP_DATA=${VP_DATA:-unset})"
    return
  fi
  (cd "${SUITE}" && py lmdb_probe.py all --conv "${VP_WORK}/lmdb-converted")
}

run_perf() {
  local b="${VP_WORK}/perf" i f
  vp_cmake_build build::performance_tests "${TEST}/performance_tests" "${b}/pt" || return
  vp_cmake_build build::dataloader_multithread "${TEST}/dataloader_multithread" "${b}/dm" || return
  # 20 real copies of AMD-tinyDataSet (5,120 JPEGs); symlinks would trip M10.
  mkdir -p "${b}/data/0"
  for i in $(seq -w 1 20); do
    for f in "${TEST}/data/images/AMD-tinyDataSet/"*.JPEG; do
      cp "${f}" "${b}/data/0/r${i}_$(basename "${f}")"
    done
  done
  (cd "${SUITE}" && py perf.py --pt "${b}/pt/performance_tests" --dm "${b}/dm/dataloader_multithread" \
    --data "${b}/data" --out "${b}/perf.json")
  vp_perf rocal-throughput "${b}/perf.json"
}

# ---------------------------------------------------------------------------
# tiers
# ---------------------------------------------------------------------------
if [[ "${PY_OK}" == 1 ]]; then
  part smoke && run_smoke
else
  part smoke && vp_blocked smoke::python "the rocAL Python modules need Python 3.12"
fi
part ctest && run_ctest

if vp_tier_ge standard; then
  part pybind && run_pybind
  part audio && run_audio
  part probes && run_h1
fi

if vp_tier_ge comprehensive; then
  part golden-cpp && run_golden_cpp
  part golden-py && run_golden_py
  if [[ "${PY_OK}" == 1 ]]; then
    part readers && run_readers
    part probes && run_probes
    part lmdb && run_lmdb
  fi
fi

if vp_tier_ge full; then
  part perf && run_perf
  part extended && (cd "${SUITE}" && py probes.py extended)
fi

vp_finish
exit 0
