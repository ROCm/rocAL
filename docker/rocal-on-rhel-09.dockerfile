FROM compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-rel-5.4:67-rhel-9.x-stg1

ENV ROCAL_DEPS_ROOT=/rocAL-deps
WORKDIR $ROCAL_DEPS_ROOT

RUN sudo yum update -y

# install rocAL base dependencies
RUN sudo yum -y install gcc g++ cmake pkg-config git kernel-devel

# install OpenCV
RUN sudo yum install opencv opencv-devel

# install rocAL neural net dependencies
RUN sudo yum -y install rocblas rocblas-devel miopen-hip miopen-hip-devel migraphx migraphx-devel


# install rocAL dependencies
RUN apt-get -y install curl make g++ unzip libomp-dev libpthread-stubs0-dev wget clang
RUN mkdir rocAL_deps && cd rocAL_deps && wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && sudo mkdir -p /usr/local/include/half && sudo cp half-files/include/half.hpp /usr/local/include/half && cd
RUN apt-get update -y && apt-get -y install autoconf automake libbz2-dev libssl-dev python3-dev libgflags-dev libgoogle-glog-dev liblmdb-dev nasm yasm libjsoncpp-dev && \
        git clone -b 2.0.6.2 https://github.com/rrawther/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ../ && make -j4 && sudo make install && cd
RUN apt-get -y install sqlite3 libsqlite3-dev libtool build-essential
RUN git clone -b v3.21.9 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && sudo make install && sudo ldconfig && cd
RUN git clone -b 0.99  https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && cd rpp && mkdir build && cd build && \
        cmake -DBACKEND=HIP ../ && make -j4 && sudo make install && cd

ENV ROCAL_WORKSPACE=/workspace
WORKDIR $ROCAL_WORKSPACE

# Install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && \
        mkdir build && cd build && cmake -DBACKEND=HIP -DROCAL=OFF ../MIVisionX && make -j8 && make install