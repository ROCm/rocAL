# syntax=docker/dockerfile:1

# Docker pulls this public image automatically when the image is not already
# present locally. A separate `docker pull` inside the image is neither needed
# nor possible without nesting a Docker daemon.
ARG BASE_IMAGE=rocm/pytorch:rocm7.14_ubuntu26.04_py3.14_pytorch_release_2.12.0
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_LIBRARIES_REF=develop
ARG ROCAL_REF=develop
ARG BUILD_JOBS=16
ARG ROCM_WHEEL_INDEX=https://rocm.nightlies.amd.com/whl-multi-arch/
ARG ROCM_DEVEL_VERSION=7.14.0a20260612

ENV DEBIAN_FRONTEND=noninteractive \
    ROCM_PATH=/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel \
    ROCAL_INSTALL=/workspace/install \
    ROCM_HOME=/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel \
    HIP_PATH=/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel \
    MIVisionX_PATH=/workspace/install \
    HALF_DIR=/workspace/install \
    PATH=/workspace/install/bin:/opt/venv/bin:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel/bin:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/workspace/install/lib:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib:/usr/lib/x86_64-linux-gnu \
    CMAKE_PREFIX_PATH=/workspace/install:/opt/venv:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib/cmake:/workspace/install/lib/cmake \
    PYTHONPATH=/workspace/install/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf automake build-essential ca-certificates clang cmake ffmpeg \
        gcc g++ git hwloc libavcodec-dev libavformat-dev libavutil-dev \
        libbz2-dev libdc1394-dev libdlpack-dev libdrm-dev libgflags-dev \
        libgoogle-glog-dev libgtk2.0-dev libhalf-dev libjpeg-dev libjsoncpp-dev \
        liblmdb-dev libomp-dev libpng-dev libprotobuf-dev libsndfile1-dev libssl-dev \
        libswscale-dev libtbb-dev libtbbmalloc2 libtiff-dev libtool libva-dev \
        libturbojpeg0-dev libva-drm2 make nasm numactl perl pkg-config protobuf-compiler python3-dev \
        python3-pip unzip vainfo vim yasm zip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p "${HALF_DIR}/include/half" \
    && cp /usr/include/half.hpp "${HALF_DIR}/include/half/half.hpp" \
    && python3 -m pip install --no-cache-dir --index-url "${ROCM_WHEEL_INDEX}" \
        "rocm-sdk-devel==${ROCM_DEVEL_VERSION}" \
    && rocm-sdk init

RUN python3 -m pip install --upgrade pip --break-system-packages \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        Cython matplotlib opencv-python pybind11 pytest==7.3.1

WORKDIR /tmp/rocal-deps

RUN git clone --depth 1 https://github.com/Tencent/rapidjson.git \
    && cmake -S rapidjson -B rapidjson/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && cmake --build rapidjson/build -j"${BUILD_JOBS}" \
    && cmake --install rapidjson/build

RUN git clone --depth 1 --branch "${ROCM_LIBRARIES_REF}" \
        https://github.com/ROCm/rocm-libraries.git /workspace/rocm-libraries \
    && cmake -S /workspace/rocm-libraries/projects/rpp \
        -B /workspace/rocm-libraries/projects/rpp/build \
        -DBACKEND=HIP \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
    && cmake --build /workspace/rocm-libraries/projects/rpp/build -j"${BUILD_JOBS}" \
    && cmake --install /workspace/rocm-libraries/projects/rpp/build \
    && ldconfig

RUN git clone --depth 1 https://github.com/ROCm/MIVisionX.git \
    && cmake -S MIVisionX -B MIVisionX/build \
        -DBACKEND=HIP \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
    && cmake --build MIVisionX/build -j"${BUILD_JOBS}" \
    && cmake --install MIVisionX/build \
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
        -DCMAKE_DISABLE_FIND_PACKAGE_rocdecode=TRUE \
        -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
        -DBUILD_PYPACKAGE=ON \
        -DPYTHON_VERSION_SUGGESTED="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" \
    && cmake --build build -j"${BUILD_JOBS}" \
    && cmake --install build \
    && ldconfig

WORKDIR /workspace
CMD ["bash"]
