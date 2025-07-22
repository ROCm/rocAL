# Copyright (c) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
import argparse
import platform
import traceback
if sys.version_info[0] < 3:
    import commands
else:
    import subprocess

libraryName = "rocAL"

__copyright__ = f"Copyright(c) 2018 - 2025, AMD ROCm {libraryName}"
__version__ = "4.1.0"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"

# ANSI Escape codes for info messages
TEXT_WARNING = "\033[93m\033[1m"
TEXT_ERROR = "\033[91m\033[1m"
TEXT_INFO = "\033[1m"
TEXT_DEFAULT = "\033[0m"

def info(msg):
    print(f"{TEXT_INFO}INFO:{TEXT_DEFAULT} {msg}")

def warn(msg):
    print(f"{TEXT_WARNING}WARNING:{TEXT_DEFAULT} {msg}")

def error(msg):
    print(f"{TEXT_ERROR}ERROR:{TEXT_DEFAULT} {msg}")

# error check for calls
def ERROR_CHECK(waitval):
    if(waitval != 0): # return code and signal flags
        error('ERROR_CHECK failed with status:'+str(waitval))
        traceback.print_stack()
        status = ((waitval >> 8) | waitval) & 255 # combine exit code and wait flags into single non-zero byte
        exit(status)

def install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, package_list):
    cmd_str = 'sudo ' + linuxFlag + ' ' + linuxSystemInstall + \
        ' ' + linuxSystemInstall_check+' install '
    for i in range(len(package_list)):
        cmd_str += package_list[i] + " "
    ERROR_CHECK(os.system(cmd_str))

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--directory', 	type=str, default='~/rocal-deps',
                    help='Setup home directory - optional (default:~/)')
parser.add_argument('--rocm_path', 	type=str, default='/opt/rocm',
                    help='ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required')
parser.add_argument('--backend', 	type=str, default='HIP',
                    help='rocAL Dependency Backend - optional (default:HIP) [options:CPU/OCL/HIP]')
parser.add_argument('--reinstall', 	type=str, default='OFF',
                    help='Remove previous setup and reinstall - optional (default:OFF) [options:ON/OFF]')
args = parser.parse_args()

setupDir = args.directory
ROCM_PATH = args.rocm_path
backend = args.backend.upper()
reinstall = args.reinstall.upper()


if reinstall not in ('OFF', 'ON'):
    error(
        "ERROR: Re-Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit(-1)
if backend not in ('OCL', 'HIP', 'CPU'):
    error(
        "ERROR: Backend Option Not Supported - [Supported Options: CPU or OCL or HIP]\n")
    parser.print_help()
    exit(-1)

# override default path if env path set 
if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
info("ROCm PATH set to -- "+ROCM_PATH+"\n")


# check ROCm installation
if os.path.exists(ROCM_PATH) and backend != 'CPU':
    info("ROCm Installation Found -- "+ROCM_PATH+"\n")
    os.system('echo ROCm Info -- && '+ROCM_PATH+'/bin/rocminfo')
else:
    if backend != 'CPU':
        warn("\nWARNING: ROCm Not Found at -- "+ROCM_PATH+"\n")
        warn(
            "WARNING: If ROCm installed, set ROCm Path with \"--rocm_path\" option for full installation [Default:/opt/rocm]\n")
        warn("WARNING: Limited dependencies will be installed\n")
        backend = 'CPU'
    else:
        info("STATUS: CPU Backend Install\n")
    neuralNetInstall = 'OFF'
    inferenceInstall = 'OFF'

# Setup Directory for Deps
if setupDir == '~/rocal-deps':
    setupDir_deps = setupDir
else:
    setupDir_deps = setupDir+'/rocal-deps'

# setup directory path
deps_dir = os.path.expanduser(setupDir_deps)
deps_dir = os.path.abspath(deps_dir)

# get platform info
platformInfo = platform.platform()

# sudo requirement check
sudoLocation = ''
userName = ''
if sys.version_info[0] < 3:
    status, sudoLocation = commands.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = commands.getstatusoutput("whoami")
else:
    status, sudoLocation = subprocess.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = subprocess.getstatusoutput("whoami")

# check os version
os_info_data = 'NOT Supported'
if os.path.exists('/etc/os-release'):
    with open('/etc/os-release', 'r') as os_file:
        os_info_data = os_file.read().replace('\n', ' ')
        os_info_data = os_info_data.replace('"', '')

# setup for Linux
linuxSystemInstall = ''
linuxCMake = 'cmake'
linuxSystemInstall_check = ''
linuxFlag = ''
sudoValidate = 'sudo -v'
osUpdate = ''
if "centos" in os_info_data or "redhat" in os_info_data or "Oracle" in os_info_data:
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    osUpdate = 'makecache'
    if "VERSION_ID=8" in os_info_data:
        platformInfo = platformInfo+'-centos-8-based'
    elif "VERSION_ID=9" in os_info_data:
        platformInfo = platformInfo+'-centos-9-based'
    else:
        platformInfo = platformInfo+'-centos-undefined-version'
elif "Ubuntu" in os_info_data:
    linuxSystemInstall = 'apt-get -y'
    linuxSystemInstall_check = '--allow-unauthenticated'
    osUpdate = 'update'
    linuxFlag = '-S'
    if "VERSION_ID=22" in os_info_data:
        platformInfo = platformInfo+'-ubuntu-22'
    elif "VERSION_ID=24" in os_info_data:
        platformInfo = platformInfo+'-ubuntu-24'
    else:
        platformInfo = platformInfo+'-ubuntu-undefined-version'
elif "SLES" in os_info_data:
    linuxSystemInstall = 'zypper -n'
    linuxSystemInstall_check = '--no-gpg-checks'
    osUpdate = 'refresh'
    platformInfo = platformInfo+'-sles'
elif "Mariner" in os_info_data:
    linuxSystemInstall = 'tdnf -y'
    linuxSystemInstall_check = '--nogpgcheck'
    platformInfo = platformInfo+'-mariner'
    osUpdate = 'makecache'
else:
    error("rocAL Setup on "+platformInfo+" is unsupported\n")
    error("rocAL Setup Supported on: Ubuntu 22/24, RedHat 8/9, & SLES 15\n")
    exit(-1)

# rocAL Setup
info(f"{libraryName} Setup on: "+platformInfo)
info(f"{libraryName} Dependencies Installation with rocAL-setup.py V-"+__version__)

if userName == 'root':
    ERROR_CHECK(os.system(linuxSystemInstall+' '+osUpdate))
    ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

# Delete previous install
if reinstall == 'ON':
    ERROR_CHECK(os.system(sudoValidate))
    if os.path.exists(deps_dir):
        ERROR_CHECK(os.system('sudo rm -rf '+deps_dir))
        info("rocAL Setup: Removing Previous Install -- "+deps_dir+"\n")

# common packages
coreCommonPackages = [
    'cmake',
    'wget',
    'unzip',
    'pkg-config',
    'inxi'
]

# rocm pacakges
rocmDebianPackages = [
    'half',
    'mivisionx-dev',
    'rocjpeg-dev',
    'rocdecode-dev'
]

rocjpegPackage = "rocjpeg-devel"
rocdecodePackage = "rocdecode-devel"
if "mariner" in platformInfo:
    rocjpegPackage = "mivisionx-devel" # TBD - rocJPEG unsupported on Mariner
    rocdecodePackage = "mivisionx-devel" # TBD - rocDecode unsupported on Mariner
rocmRPMPackages = [
    'half',
    'mivisionx-devel',
    str(rocjpegPackage),
    str(rocdecodePackage)
]

# core package
coreDebianPackages = [
    'nasm',
    'yasm',
    'liblmdb-dev',
    #'rapidjson-dev',
    'libsndfile1-dev', # for audio features
    'python3-dev',
    'python3-pip',
    'python3-protobuf',
    'libprotobuf-dev',
    'libprotoc-dev',
    'protobuf-compiler',
    'libturbojpeg0-dev'
]

libsndFile = "libsndfile-devel"
libPythonProto = "python3-protobuf"
libProtoCompiler = "protobuf-compiler"
libTurboJPEG = "turbojpeg-devel"
if "sles" in platformInfo:
    libProtoCompiler = "libprotobuf-c-devel"
    libsndFile = "cmake" # TBD - libsndfile-devel  fails to install in SLES
    libTurboJPEG = "cmake" # TBD libturbojpeg0 dev/devel package unavailable in SLES
coreRPMPackages = [
    'nasm',
    'yasm',
    'lmdb-devel',
    'jsoncpp-devel',
    #'rapidjson-devel',
    str(libsndFile), # for audio features
    'python3-devel',
    'python3-pip',
    str(libPythonProto),
    'protobuf-devel',
    str(libProtoCompiler),
    str(libTurboJPEG)
]

pip3Packages = [
    'pytest~=7.0.0',
    'wheel~=0.37.0'
]

openclDebianPackages = [
    'ocl-icd-opencl-dev'
]

openclRPMPackages = [
    'ocl-icd-devel'
]

opencvDebianPackages = [
    'libopencv-dev'
]

opencvRPMPackages = [
    'gtk2-devel',
    'libjpeg-devel',
    'libpng-devel',
    'libtiff-devel',
    'libavc1394'
]

# update
ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +' '+linuxSystemInstall_check+' '+osUpdate))

ERROR_CHECK(os.system(sudoValidate))
# common packages
install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, coreCommonPackages)
# HIP Backend support
if backend == 'HIP':
    if "ubuntu" in platformInfo:
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, rocmDebianPackages)
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, coreDebianPackages)
    else:
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, rocmRPMPackages)
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, coreRPMPackages)

# Install OpenCL ICD Loader
if backend == 'OCL':
    if "ubuntu" in platformInfo:
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, openclDebianPackages)
    else:
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, openclRPMPackages)

# OpenCV
if "ubuntu" in platformInfo:
    install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, opencvDebianPackages)

#pip3 packages
for i in range(len(pip3Packages)):
    ERROR_CHECK(os.system('pip3 install '+ pip3Packages[i]))
        
if os.path.exists(deps_dir):
    info("rocAL Setup: Re-Installed Libraries\n")
# Clean Install
else:
    info("rocAL Dependencies Clean Installation with rocAL-setup.py V-"+__version__+"\n")
    ERROR_CHECK(os.system(sudoValidate))
    # Create deps & build folder
    ERROR_CHECK(os.system('mkdir '+deps_dir))
    ERROR_CHECK(os.system('(cd '+deps_dir+'; mkdir build )'))

    # turbo-JPEG - https://github.com/libjpeg-turbo/libjpeg-turbo.git -- 3.0.2
    if "SLES" in platformInfo:
        turboJpegVersion = '3.0.2'
        ERROR_CHECK(os.system(
                    '(cd '+deps_dir+'; git clone -b '+turboJpegVersion+' https://github.com/libjpeg-turbo/libjpeg-turbo.git )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/libjpeg-turbo; mkdir build; cd build; '+linuxCMake +
                    ' -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib -DWITH_JPEG8=TRUE ..; make -j$(nproc); sudo make install )'))

    # PyBind11 - https://github.com/pybind/pybind11 -- v2.11.1
    pybind11Version = 'v2.11.1'
    ERROR_CHECK(os.system('(cd '+deps_dir+'; git clone -b '+pybind11Version+' https://github.com/pybind/pybind11; cd pybind11; mkdir build; cd build; ' +
            linuxCMake+' -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON ../; make -j$(nproc); sudo make install)'))
    
    # dlpack - https://github.com/dmlc/dlpack
    if "ubuntu" in platformInfo:
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install libdlpack-dev'))
    elif "sles" in platformInfo:
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install dlpack-devel'))
    else:
        ERROR_CHECK(os.system('(cd '+deps_dir+'; git clone -b v1.0 https://github.com/dmlc/dlpack.git)'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/dlpack; mkdir -p build && cd build; '+linuxCMake+' ..; make -j$(nproc); sudo make install)'))

    # RapidJSON - Source TBD: Package install of RapidJSON has compile issues - https://github.com/Tencent/rapidjson.git -- master
    ERROR_CHECK(os.system('(cd '+deps_dir+'; git clone https://github.com/Tencent/rapidjson.git; cd rapidjson; mkdir build; cd build; ' +	
            linuxCMake+' ../; make -j$(nproc); sudo make install)'))
    
    # libtar - https://repo.or.cz/libtar.git ; version - v1.2.20
    libtar_version = 'v1.2.20'
    ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install autoconf libtool'))
    ERROR_CHECK(os.system(
        '(cd '+deps_dir+'; git clone -b '+ libtar_version+' https://repo.or.cz/libtar.git )'))
    ERROR_CHECK(os.system('(cd '+deps_dir+'/libtar; '+
            ' autoreconf --force --install; CFLAGS="-fPIC" ./configure; make -j$(nproc); sudo make install )'))
    
    # Install OpenCV -- TBD cleanup
    ERROR_CHECK(os.system('(cd '+deps_dir+'/build; mkdir OpenCV )'))
    # Install
    if "ubuntu" in platformInfo:
        info("STATUS: rocAL Setup: OpenCV Package installed for Ubuntu\n")
    else:
        if "centos" in platformInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' groupinstall \'Development Tools\''))
        elif "sles" in platformInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install -t pattern devel_basis'))

        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, opencvRPMPackages)
        # OpenCV 4.6.0
        # Get Source and install
        opencvVersion = '4.6.0'
        ERROR_CHECK(os.system(
            '(cd '+deps_dir+'; wget https://github.com/opencv/opencv/archive/'+opencvVersion+'.zip )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'; unzip '+opencvVersion+'.zip )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; '+linuxCMake +
                        ' -D WITH_EIGEN=OFF \
                        -D WITH_GTK=ON \
                        -D WITH_JPEG=ON \
                        -D BUILD_JPEG=ON \
                        -D WITH_OPENCL=OFF \
                        -D WITH_OPENCLAMDFFT=OFF \
                        -D WITH_OPENCLAMDBLAS=OFF \
                        -D WITH_VA_INTEL=OFF \
                        -D WITH_OPENCL_SVM=OFF  \
                        -D CMAKE_INSTALL_PREFIX=/usr/local \
                        -D BUILD_LIST=core,features2d,highgui,imgcodecs,imgproc,photo,video,videoio  \
                        -D CMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
                        ../../opencv-'+opencvVersion+' )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; make -j$(nproc))'))
        ERROR_CHECK(os.system(sudoValidate))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; sudo make install)'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; sudo ldconfig)'))

info(f"{libraryName} Dependencies Installed with rocAL-setup.py V-"+__version__+" on "+platformInfo+"\n")
