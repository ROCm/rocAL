# syntax=docker/dockerfile:1

# Docker pulls this public image automatically when the image is not already
# present locally. A separate `docker pull` inside the image is neither needed
# nor possible without nesting a Docker daemon.
ARG BASE_IMAGE=rocm/pytorch:rocm7.14_ubuntu26.04_py3.14_pytorch_release_2.12.0
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG THEROCK_URL=https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-gfx94X-dcgpu-7.14.0a20260612.tar.gz
ARG RPP_REF=develop
ARG ROCAL_REF=develop
ARG ROCM_SYSTEMS_REF=develop
ARG BUILD_JOBS=16
ARG ROCM_ARCH=gfx942

ENV DEBIAN_FRONTEND=noninteractive \
    ROCM_PATH=/workspace/install \
    ROCAL_INSTALL=/workspace/install \
    ROCJPEG_INSTALL=/workspace/install \
    ROCM_HOME=/workspace/install \
    HIP_PATH=/workspace/install \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
    PATH=/workspace/install/bin:/workspace/install/lib/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/workspace/install/lib:/workspace/install/lib/rocm_sysdeps/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
    CMAKE_PREFIX_PATH=/workspace/install:/workspace/install/lib/cmake:/workspace/install/lib64/cmake \
    PYTHONPATH=/workspace/install/lib:${PYTHONPATH:-}

RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf automake build-essential ca-certificates clang cmake ffmpeg \
        gcc g++ git hwloc libavcodec-dev libavformat-dev libavutil-dev \
        libbz2-dev libdc1394-dev libdlpack-dev libdrm-dev libgflags-dev \
        libgoogle-glog-dev libgtk2.0-dev libjpeg-dev libjsoncpp-dev \
        liblmdb-dev libomp-dev libpng-dev libsndfile1-dev libssl-dev \
        libswscale-dev libtbb-dev libtbbmalloc2 libtiff-dev libtool libva-dev \
        libva-drm2 make mesa-va-drivers nasm numactl perl pkg-config python3-dev \
        python3-pip unzip vainfo vim wget yasm zip \
    && rm -rf /var/lib/apt/lists/*

# Install the matching TheRock distribution over the public image's userspace
# ROCm stack. Keeping everything under one prefix prevents mixed /opt/rocm and
# /workspace/install link resolution.
RUN mkdir -p "${ROCM_PATH}" \
    && wget --progress=dot:giga -O /tmp/therock.tar.gz "${THEROCK_URL}" \
    && tar -xf /tmp/therock.tar.gz -C "${ROCM_PATH}" \
    && rm -f /tmp/therock.tar.gz

RUN python3 -m pip install --upgrade pip --break-system-packages \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        Cython matplotlib opencv-python pybind11 pytest==7.3.1

WORKDIR /tmp/rocal-deps

RUN git clone --depth 1 --branch 3.0.2 https://github.com/libjpeg-turbo/libjpeg-turbo.git \
    && cmake -S libjpeg-turbo -B libjpeg-turbo/build \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_STATIC=OFF \
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib \
        -DWITH_JPEG8=ON \
    && cmake --build libjpeg-turbo/build -j"${BUILD_JOBS}" \
    && cmake --install libjpeg-turbo/build \
    && ldconfig

RUN git clone --depth 1 --branch v3.21.9 https://github.com/protocolbuffers/protobuf.git \
    && cd protobuf \
    && git submodule update --init --recursive --depth 1 \
    && ./autogen.sh \
    && ./configure \
    && make -j"${BUILD_JOBS}" \
    && make install \
    && ldconfig

RUN git clone --depth 1 https://github.com/Tencent/rapidjson.git \
    && cmake -S rapidjson -B rapidjson/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && cmake --build rapidjson/build -j"${BUILD_JOBS}" \
    && cmake --install rapidjson/build

RUN git clone --depth 1 --branch "${RPP_REF}" https://github.com/ROCm/rpp.git \
    && cmake -S rpp -B rpp/build \
        -DBACKEND=HIP \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
    && cmake --build rpp/build -j"${BUILD_JOBS}" \
    && cmake --install rpp/build \
    && ldconfig

RUN git clone --depth 1 https://github.com/ROCm/MIVisionX.git \
    && cmake -S MIVisionX -B MIVisionX/build \
        -DBACKEND=HIP \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
    && cmake --build MIVisionX/build -j"${BUILD_JOBS}" \
    && cmake --install MIVisionX/build \
    && ldconfig

RUN git clone --depth 1 --branch "${ROCM_SYSTEMS_REF}" \
        https://github.com/ROCm/rocm-systems.git /workspace/rocm-systems \
    && cmake -S /workspace/rocm-systems/projects/rocjpeg \
        -B /workspace/rocm-systems/projects/rocjpeg/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
    && cmake --build /workspace/rocm-systems/projects/rocjpeg/build -j"${BUILD_JOBS}" \
    && cmake --install /workspace/rocm-systems/projects/rocjpeg/build \
    && cmake -S /workspace/rocm-systems/projects/rocdecode \
        -B /workspace/rocm-systems/projects/rocdecode/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
    && cmake --build /workspace/rocm-systems/projects/rocdecode/build -j"${BUILD_JOBS}" \
    && cmake --install /workspace/rocm-systems/projects/rocdecode/build \
    && ldconfig

RUN git clone --depth 1 --branch "${ROCAL_REF}" https://github.com/ROCm/rocAL.git /workspace/rocAL \
    && cd /workspace/rocAL \
    && sed -i 's/VERSION_GREATER "3.13"/VERSION_GREATER "3.14"/' CMakeLists.txt \
    && sed -i 's/"3.13" "3.12"/"3.14" "3.13" "3.12"/' rocAL_pybind/CMakeLists.txt \
    && if ! grep -q 'find_package(JPEG QUIET)' rocAL/CMakeLists.txt; then \
         sed -i '/find_package(TurboJpeg QUIET)/a find_package(JPEG QUIET)' rocAL/CMakeLists.txt; \
       fi \
    && if ! grep -q 'JPEG_LIBRARIES' rocAL/CMakeLists.txt; then \
         perl -0pi -e 's|(set\(LINK_LIBRARY_LIST \$\{LINK_LIBRARY_LIST\} \$\{TurboJpeg_LIBRARIES\}\))|$1\n    if(JPEG_FOUND)\n        include_directories(\$\{JPEG_INCLUDE_DIRS\})\n        set(LINK_LIBRARY_LIST \$\{LINK_LIBRARY_LIST\} \$\{JPEG_LIBRARIES\})\n    endif()|' rocAL/CMakeLists.txt; \
       fi \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ROCAL_INSTALL}" \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
        -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
        -DBUILD_PYPACKAGE=ON \
        -DPYTHON_VERSION_SUGGESTED="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" \
    && cmake --build build -j"${BUILD_JOBS}" \
    && cmake --install build \
    && ldconfig

# Build optional diagnostic binaries while all source trees are available.
RUN cmake -S /workspace/rocm-systems/projects/rocjpeg/samples/jpegDecodePerf \
        -B /workspace/rocm-systems/projects/rocjpeg/samples/jpegDecodePerf/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
    && cmake --build /workspace/rocm-systems/projects/rocjpeg/samples/jpegDecodePerf/build -j"${BUILD_JOBS}" \
    && CC="${ROCM_PATH}/lib/llvm/bin/amdclang" \
       CXX="${ROCM_PATH}/lib/llvm/bin/amdclang++" \
       cmake -S /workspace/rocAL/tests/cpp_api/dataloader_multithread \
        -B /workspace/rocAL/build-dataloader-multithread \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
    && cmake --build /workspace/rocAL/build-dataloader-multithread -j"${BUILD_JOBS}"

# Preserve the convenient environment script used by the manual procedure.
RUN printf '%s\n' \
      'export ROCM_PATH=/workspace/install' \
      'export ROCAL_INSTALL=/workspace/install' \
      'export ROCJPEG_INSTALL=/workspace/install' \
      'export ROCM_HOME=/workspace/install' \
      'export HIP_PATH=/workspace/install' \
      'export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}' \
      'export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"' \
      'export LD_LIBRARY_PATH="$ROCAL_INSTALL/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"' \
      'export CMAKE_PREFIX_PATH="$ROCM_PATH:$ROCM_PATH/lib/cmake:$ROCM_PATH/lib64/cmake:${CMAKE_PREFIX_PATH:-}"' \
      'export PYTHONPATH="$ROCAL_INSTALL/lib:${PYTHONPATH:-}"' \
      > /workspace/rocal_test_env.sh

RUN cat > /workspace/rocal_versions.sh <<'SCRIPT'
#!/usr/bin/env bash

# Report the software stack without stopping when an optional component does
# not expose version information.
set -u

if [[ -f /workspace/rocal_test_env.sh ]]; then
    source /workspace/rocal_test_env.sh
fi

echo "===== Operating system ====="
sed -n -E 's/^(PRETTY_NAME|VERSION_ID)=/\1=/p' /etc/os-release 2>/dev/null || true
uname -a

echo
echo "===== Python stack ====="
python3 - <<'PY'
import importlib.metadata as metadata
import sys

print("Python:", sys.version.replace("\n", " "))
for package in (
    "torch",
    "pybind11",
    "numpy",
    "opencv-python",
):
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "not installed or no package metadata"
    print(f"{package}: {version}")

try:
    import torch
    print("torch.version.hip:", torch.version.hip)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}:", torch.cuda.get_device_name(index))
except Exception as error:
    print("PyTorch runtime query failed:", repr(error))

for module_name in ("pybind11", "rocal_pybind"):
    try:
        module = __import__(module_name)
        print(f"{module_name} module:", getattr(module, "__file__", "unknown"))
        if module_name == "pybind11":
            print("pybind11.__version__:", getattr(module, "__version__", "unknown"))
    except Exception as error:
        print(f"{module_name} import failed:", repr(error))
PY

echo
echo "===== ROCm and build tools ====="
echo "ROCM_PATH=${ROCM_PATH:-unset}"
echo "ROCAL_INSTALL=${ROCAL_INSTALL:-unset}"

for version_file in \
    "${ROCM_PATH:-/workspace/install}/.info/version" \
    "${ROCM_PATH:-/workspace/install}/.info/version-dev" \
    /opt/rocm/.info/version; do
    if [[ -f "$version_file" ]]; then
        echo "$version_file: $(<"$version_file")"
    fi
done

command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -m1 -E 'Runtime Version|ROCm' || true
command -v hipcc >/dev/null 2>&1 && hipcc --version | sed -n '1,5p' || true
command -v amdclang++ >/dev/null 2>&1 && amdclang++ --version | sed -n '1,2p' || true
command -v cmake >/dev/null 2>&1 && cmake --version | sed -n '1p' || true
command -v gcc >/dev/null 2>&1 && gcc --version | sed -n '1p' || true

echo
echo "===== Built source revisions ====="
for repository in /workspace/rocAL /workspace/rocm-systems; do
    if git -C "$repository" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf '%s: ' "$repository"
        git -C "$repository" log -1 --format='%H %s'
    fi
done

echo
echo "===== rocAL linkage ====="
rocal_library="${ROCAL_INSTALL:-/workspace/install}/lib/librocal.so"
if [[ -e "$rocal_library" ]]; then
    ldd "$rocal_library" 2>/dev/null | \
        grep -E 'rocjpeg|rpp|turbojpeg|jpeg|amdhip|hsa|openvx|protobuf' || true
else
    echo "Not found: $rocal_library"
fi
SCRIPT
RUN chmod 0755 /workspace/rocal_versions.sh

# Validate the framework and rocAL installation without depending on an
# application repository. Application source will be mounted at runtime.
WORKDIR /workspace

RUN python3 - <<'PY'
import torch
import amd.rocal.pipeline
import amd.rocal.fn
import amd.rocal.types
import rocal_pybind

print("torch:", torch.__version__, "HIP:", torch.version.hip)
print("rocAL Python import: OK")
PY

# GPU-dependent checks (torch.cuda, rocAL ctest, jpegdecodeperf) must run after
# `docker run` with /dev/kfd and /dev/dri passed through; they cannot be validly
# executed during `docker build`.
CMD ["bash"]
