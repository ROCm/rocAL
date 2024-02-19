FROM ubuntu:20.04

ARG ROCM_INSTALLER_REPO=https://repo.radeon.com/amdgpu-install/5.4.1/ubuntu/focal/amdgpu-install_5.4.50401-1_all.deb 
ARG ROCM_INSTALLER_PACKAGE=amdgpu-install_5.4.50401-1_all.deb

ENV ROCAL_DEPS_ROOT=/rocAL-deps
WORKDIR $ROCAL_DEPS_ROOT

RUN apt-get update -y

# install rocAL base dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git

# install ROCm for rocAL OpenCL/HIP dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration &&  \
        sudo apt-get -y clean && dpkg --add-architecture i386 && \
        wget ${ROCM_INSTALLER_REPO} && \
        sudo apt-get install -y ./${ROCM_INSTALLER_PACKAGE} && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=graphics,rocm

# install OpenCV
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip && \
        mkdir OpenCV && cd OpenCV && wget https://github.com/opencv/opencv/archive/4.6.0.zip && unzip 4.6.0.zip && \
        mkdir build && cd build && cmake -DWITH_GTK=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_OPENCL=OFF ../opencv-4.6.0 && make -j8 && sudo make install && sudo ldconfig && cd

# install FFMPEG
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/"
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev \
        libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo wget zlib1g-dev \
        nasm yasm libx264-dev libx265-dev libnuma-dev libfdk-aac-dev && \
        wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.4.2.zip && unzip n4.4.2.zip && cd FFmpeg-n4.4.2/ && sudo ldconfig && \
        ./configure --enable-shared --disable-static --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libass --enable-gpl --enable-nonfree && \
        make -j8 && sudo make install && cd

# install rocAL neural net dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install rocblas rocblas-dev miopen-hip miopen-hip-dev migraphx && \
        mkdir neuralNet && cd neuralNet && wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && sudo mkdir -p /usr/local/include/half && sudo cp half-files/include/half.hpp /usr/local/include/half && cd

# install rocAL dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install wget libbz2-dev libssl-dev python-dev python3-dev libgflags-dev libgoogle-glog-dev liblmdb-dev nasm yasm libjsoncpp-dev clang && \
        git clone -b 3.0.1 https://github.com/libjpeg-turbo/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib -DWITH_JPEG8=TRUE ../ && \
        git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && cd rpp && mkdir build && cd build && \
        cmake -DBACKEND=HIP ../ && make -j4 && sudo make install && cd ../../ && \
        git clone -b v3.12.4 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && sudo make install && sudo ldconfig && cd
ENV CUPY_INSTALL_USE_HIP=1
ENV ROCM_HOME=/opt/rocm
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3 python3-pip git g++ hipblas hipsparse rocrand hipfft rocfft rocthrust-dev hipcub-dev python3-dev && \
        git clone https://github.com/Tencent/rapidjson.git && cd rapidjson && mkdir build && cd build && \
        cmake ../ && make -j4 && sudo make install && cd ../../ && \
        pip install pytest==3.1 && git clone -b v2.10.4 https://github.com/pybind/pybind11 && cd pybind11 && mkdir build && cd build && \
        cmake -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON ../ && make -j4 && sudo make install && cd ../../ && \
        git clone https://github.com/ROCmSoftwarePlatform/cupy.git && cd cupy && git submodule update --init && \
        pip install -e . --no-cache-dir -vvvv && pip install numpy==1.21

ENV ROCAL_WORKSPACE=/workspace
WORKDIR $ROCAL_WORKSPACE

# install MIVisionX 
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && \
        mkdir build && cd build && cmake -DBACKEND=HIP -DROCAL=OFF ../MIVisionX && make -j8 && make install
