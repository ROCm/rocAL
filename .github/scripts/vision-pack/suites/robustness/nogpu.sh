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

# robustness-nogpu (gpu_access: none), rocAL checks: CPU paths with no GPU at all.
# baselines/known_issues.yaml matches these result IDs: keep them stable.
#
# In CI the job container has no /dev/kfd and no render node. On a GPU host,
# ROCR_VISIBLE_DEVICES=-1 (set by run_suite.sh --gpu-access none) emulates that:
# it leaves ROCr with zero GPU agents but keeps /dev/kfd and the render nodes
# openable; env::no-gpu-agents records which situation this run is in.
set -uo pipefail
source "${VP_REPO}/build_tools/lib/vp.sh"
vp_init robustness-nogpu

if ! vp_tier_ge comprehensive; then
  vp_skip "tier::below-comprehensive" "robustness-nogpu runs in the comprehensive tier and above"
  vp_finish
  exit 0
fi

CR="${VP_REPO}/suites/robustness/checked_run.py"
PROBES="${VP_REPO}/suites/robustness/probes"
TINY="${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet"
export TMPDIR="${VP_WORK}/tmp"
mkdir -p "${TMPDIR}"

have_py() { "${VP_PY}" -c "import $1" >/dev/null 2>&1; }

fresh_dir() {
  local d="${VP_WORK}/cwd/$1"
  rm -rf "${d}"
  mkdir -p "${d}"
  printf '%s' "${d}"
}

cr() {
  local id="$1"; shift
  "${VP_PY}" "${CR}" --id "${id}" --timeout 300 --error-rc 70 "$@"
}

# The premise: nothing may see a GPU. If this fails, every result below is suspect.
cr "env::no-gpu-agents" --timeout 120 -- "${VP_PY}" "${PROBES}/gpu_agents.py" --expect 0

# Imports of every shipped rocAL Python module (the torch plugin needs torch).
for m in rocal_pybind amd.rocal amd.rocal.fn amd.rocal.pipeline amd.rocal.types amd.rocal.readers \
         amd.rocal.decoders amd.rocal.plugin.generic; do
  vp_run "import::${m}" --timeout 120 -- "${VP_PY}" -c "import ${m}"
done
if have_py torch; then
  vp_run "import::amd.rocal.plugin.pytorch" --timeout 120 -- "${VP_PY}" -c "import amd.rocal.plugin.pytorch"
else
  vp_blocked "import::amd.rocal.plugin.pytorch" "torch is not installed"
fi

# rocAL: CPU pipelines (M13: CPU mode still needs a GPU); a GPU pipeline must fail cleanly.
cr "rocal::cpu-decode-only" --backend CPU --cwd "$(fresh_dir rocal_cpu_decode)" \
  -- "${VP_PY}" "${PROBES}/rocal_pipeline.py" --backend cpu --mode correct --data "${TINY}"
cr "rocal::cpu-decode-resize" --backend CPU --cwd "$(fresh_dir rocal_cpu_resize)" \
  -- "${VP_PY}" "${PROBES}/rocal_pipeline.py" --backend cpu --mode correct --data "${TINY}" --resize 224 224
cr "rocal::gpu-request-clean-error" --backend GPU --cwd "$(fresh_dir rocal_gpu)" \
  -- "${VP_PY}" "${PROBES}/rocal_pipeline.py" --backend gpu --mode error --data "${TINY}" --resize 224 224

vp_finish
exit 0
