ARG PYTORCH_VERSION=latest
FROM rocm/pytorch:${PYTORCH_VERSION}

ENV ROCAL_DEPS_ROOT=/rocAL-deps
WORKDIR $ROCAL_DEPS_ROOT

RUN apt-get update -y

# install rocAL base dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git

# install ROCm dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration 

# install OpenCV
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip && \
        mkdir OpenCV && cd OpenCV && wget https://github.com/opencv/opencv/archive/4.6.0.zip && unzip 4.6.0.zip && \
        mkdir build && cd build && cmake -DWITH_GTK=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_OPENCL=OFF ../opencv-4.6.0 && make -j8 && sudo make install && sudo ldconfig && cd

# install FFMPEG
RUN DEBIAN_FRONTEND=noninteractive apt-get -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev

# install rocAL neural net dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install half rocblas-dev miopen-hip-dev migraphx-dev

# install rocAL dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install rpp-dev wget libbz2-dev libssl-dev python-dev python3-dev libgflags-dev libgoogle-glog-dev liblmdb-dev nasm yasm libjsoncpp-dev clang && \
        git clone -b 2.0.6.2 https://github.com/rrawther/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ../ && make -j4 && sudo make install && cd ../../ && \
        git clone -b v3.12.4 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && sudo make install && sudo ldconfig && cd
ENV CUPY_INSTALL_USE_HIP=1
ENV ROCM_HOME=/opt/rocm
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3 python3-pip git g++ hipblas hipsparse rocrand hipfft rocfft rocthrust-dev hipcub-dev python3-dev && \
        git clone https://github.com/Tencent/rapidjson.git && cd rapidjson && mkdir build && cd build && \
        cmake ../ && make -j4 && sudo make install && cd ../../ && \
        pip install pytest==3.1 && git clone -b v2.10.4 https://github.com/pybind/pybind11 && cd pybind11 && mkdir build && cd build && \
        cmake -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON ../ && make -j4 && sudo make install && cd ../../ && \
        pip install numpy==1.24.2 scipy==1.9.3 cython==0.29.* git+https://github.com/ROCm/hipify_torch.git && \
        env CC=$MPI_HOME/bin/mpicc python -m pip install mpi4py && \
        git clone https://github.com/ROCm/cupy.git && cd cupy && git submodule update --init && \
        pip install -e . --no-cache-dir -vvvv

# install MIVisionX 
RUN git clone https://github.com/ROCm/MIVisionX.git && cd MIVisionX && \
        mkdir build && cd build && cmake -DBACKEND=HIP ../ && make -j8 && make install

ENV ROCAL_WORKSPACE=/workspace
WORKDIR $ROCAL_WORKSPACE

# Install rocAL
RUN git clone -b develop https://github.com/ROCm/rocAL && \
        mkdir build && cd build && cmake ../rocAL && make -j8 && cmake --build . --target PyPackageInstall && make install