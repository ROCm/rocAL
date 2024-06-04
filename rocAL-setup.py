# Copyright (c) 2022 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

__copyright__ = "Copyright 2022 - 2024, AMD ROCm Augmentation Library"
__license__ = "MIT"
__version__ = "2.1.0"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"

# error check calls
def ERROR_CHECK(call):
    status = call
    if(status != 0):
        print('ERROR_CHECK failed with status:'+str(status))
        traceback.print_stack()
        exit(status)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--directory', 	type=str, default='~/rocal-deps',
                    help='Setup home directory - optional (default:~/)')
parser.add_argument('--opencv',    	type=str, default='4.6.0',
                    help='OpenCV Version - optional (default:4.6.0)')
parser.add_argument('--pybind11',   type=str, default='v2.11.1',
                    help='PyBind11 Version - optional (default:v2.11.1)')
parser.add_argument('--backend', 	type=str, default='HIP',
                    help='rocAL Dependency Backend - optional (default:HIP) [options:CPU/OCL/HIP]')
parser.add_argument('--rocm_path', 	type=str, default='/opt/rocm',
                    help='ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required')
parser.add_argument('--reinstall', 	type=str, default='OFF',
                    help='Remove previous setup and reinstall - optional (default:OFF) [options:ON/OFF]')
args = parser.parse_args()

setupDir = args.directory
opencvVersion = args.opencv
pybind11Version = args.pybind11
ROCM_PATH = args.rocm_path
backend = args.backend.upper()
reinstall = args.reinstall.upper()

# override default path if env path set 
if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
print("\nROCm PATH set to -- "+ROCM_PATH+"\n")

# check developer inputs
if reinstall not in ('OFF', 'ON'):
    print(
        "ERROR: Re-Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit()
if backend not in ('OCL', 'HIP', 'CPU'):
    print(
        "ERROR: Backend Option Not Supported - [Supported Options: CPU or OCL or HIP]\n")
    parser.print_help()
    exit()

# check ROCm installation
if os.path.exists(ROCM_PATH) and backend != 'CPU':
    print("\nROCm Installation Found -- "+ROCM_PATH+"\n")
    os.system('echo ROCm Info -- && '+ROCM_PATH+'/bin/rocminfo')
else:
    if backend != 'CPU':
        print("\nWARNING: ROCm Not Found at -- "+ROCM_PATH+"\n")
        print(
            "WARNING: If ROCm installed, set ROCm Path with \"--rocm_path\" option for full installation [Default:/opt/rocm]\n")
        print("WARNING: Limited dependencies will be installed\n")
        backend = 'CPU'
    else:
        print("\nSTATUS: CPU Backend Install\n")

# get platfrom info
platfromInfo = platform.platform()

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

# Setup Directory for Deps
if setupDir == '~/rocal-deps':
    setupDir_deps = setupDir
else:
    setupDir_deps = setupDir+'/rocal-deps'

# setup directory path
deps_dir = os.path.expanduser(setupDir_deps)
deps_dir = os.path.abspath(deps_dir)

# setup for Linux
linuxSystemInstall = ''
linuxCMake = 'cmake'
linuxSystemInstall_check = ''
linuxFlag = ''
sudoValidateOption= '-v'
if "centos" in platfromInfo or "redhat" in platfromInfo or os.path.exists('/usr/bin/yum'):
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    if "centos-7" in platfromInfo or "redhat-7" in platfromInfo:
        linuxCMake = 'cmake3'
        ERROR_CHECK(os.system(linuxSystemInstall+' install cmake3'))
        sudoValidateOption = ''
    if not "centos" in platfromInfo or not "redhat" in platfromInfo:
        if "8" in platform.version():
            platfromInfo = platfromInfo+'-redhat-8'
        if "9" in platform.version():
            platfromInfo = platfromInfo+'-redhat-9'
elif "Ubuntu" in platfromInfo or os.path.exists('/usr/bin/apt-get'):
    linuxSystemInstall = 'apt-get -y'
    linuxSystemInstall_check = '--allow-unauthenticated'
    linuxFlag = '-S'
    if not "Ubuntu" in platfromInfo:
        platfromInfo = platfromInfo+'-Ubuntu'
elif os.path.exists('/usr/bin/zypper'):
    linuxSystemInstall = 'zypper -n'
    linuxSystemInstall_check = '--no-gpg-checks'
    platfromInfo = platfromInfo+'-SLES'
else:
    print("\nrocAL Setup on "+platfromInfo+" is unsupported\n")
    print("\nrocAL Setup Supported on: Ubuntu 20/22, CentOS 7, RedHat 8/9, & SLES 15-SP4\n")
    exit()

# rocAL Setup
print("\nrocAL Setup on: "+platfromInfo+"\n")

if userName == 'root':
    ERROR_CHECK(os.system(linuxSystemInstall+' update'))
    ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

# Delete previous install
if os.path.exists(deps_dir) and reinstall == 'ON':
    ERROR_CHECK(os.system('sudo '+sudoValidateOption))
    ERROR_CHECK(os.system('sudo rm -rf '+deps_dir))
    print("\nrocAL Setup: Removing Previous Install -- "+deps_dir+"\n")

# source install - package dependencies
commonPackages = [
    'gcc',
    'cmake',
    'git',
    'wget',
    'unzip',
    'pkg-config',
    'inxi'
]

rocmDebianPackages = [
    'half',
    'rpp',
    'rpp-dev',
    'mivisionx',
    'mivisionx-dev',
    'rocdecode',
    'rocdecode-dev'
]

rocmRPMPackages = [
    'half',
    'rpp',
    'rpp-devel',
    'mivisionx',
    'mivisionx-devel',
    'rocdecode',
    'rocdecode-devel'
]

opencvDebianPackages = [
    'build-essential',
    'pkg-config',
    'libgtk2.0-dev',
    'libavcodec-dev',
    'libavformat-dev',
    'libswscale-dev',
    'libtbb2',
    'libtbb-dev',
    'libjpeg-dev',
    'libpng-dev',
    'libtiff-dev',
    'libdc1394-dev',
    'unzip'
]

opencvRPMPackages = [
    'gtk2-devel',
    'libjpeg-devel',
    'libpng-devel',
    'libtiff-devel',
    'libavc1394',
    'unzip'
]

ffmpegDebianPackages = [
    'ffmpeg',
    'libavcodec-dev',
    'libavformat-dev',
    'libavutil-dev',
    'libswscale-dev'
]

coreDebianPackages = [
    'nasm',
    'yasm',
    'liblmdb-dev',
    #'rapidjson-dev',
    'python3-dev',
    'python3-pip',
    'python3-protobuf',
    'libprotobuf-dev',
    'libprotoc-dev',
    'protobuf-compiler'
]

coreRPMPackages = [
    'nasm',
    'yasm',
    'lmdb-devel',
    'jsoncpp-devel',
    #'rapidjson-devel',
    'python3-devel',
    'python3-pip',
    'python3-protobuf',
    'protobuf-devel',
    'protobuf-compiler'
]

# Install
ERROR_CHECK(os.system('sudo '+sudoValidateOption))
if os.path.exists(deps_dir):
    print("\nrocAL Setup: install found -- "+deps_dir)
    print("\nrocAL Setup: use option --reinstall ON to reinstall all dependencies")
    print("\nrocAL Dependencies Previously Installed with rocAL-setup.py")
    exit(0)
# Clean Install
else:
    print("\nrocAL Dependencies Installation with rocAL-setup.py V-"+__version__+"\n")
    ERROR_CHECK(os.system('mkdir '+deps_dir))
    # Create Build folder
    ERROR_CHECK(os.system('(cd '+deps_dir+'; mkdir build )'))
    # install common pre-reqs
    ERROR_CHECK(os.system('sudo '+sudoValidateOption))
    # common packages
    for i in range(len(commonPackages)):
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ commonPackages[i]))

    # ROCm Packages
    if "Ubuntu" in platfromInfo:
        for i in range(len(rocmDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rocmDebianPackages[i]))
    else:
        for i in range(len(rocmRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ rocmRPMPackages[i]))

    ERROR_CHECK(os.system('sudo '+sudoValidateOption))
    # rocAL Core Packages
    if "Ubuntu" in platfromInfo:
        for i in range(len(coreDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ coreDebianPackages[i]))
    else:
        for i in range(len(coreRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ coreRPMPackages[i]))

    # turbo-JPEG - https://github.com/libjpeg-turbo/libjpeg-turbo.git -- 3.0.2
    ERROR_CHECK(os.system(
        '(cd '+deps_dir+'; git clone -b 3.0.2 https://github.com/libjpeg-turbo/libjpeg-turbo.git )'))
    ERROR_CHECK(os.system('(cd '+deps_dir+'/libjpeg-turbo; mkdir build; cd build; '+linuxCMake +
            ' -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib -DWITH_JPEG8=TRUE ..; make -j$(nproc); sudo make install )'))

    # PyBind11
    ERROR_CHECK(os.system('pip install pytest==7.3.1'))
    ERROR_CHECK(os.system('(cd '+deps_dir+'; git clone -b '+pybind11Version+' https://github.com/pybind/pybind11; cd pybind11; mkdir build; cd build; ' +
            linuxCMake+' -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON ../; make -j$(nproc); sudo make install)'))

    # RapidJSON - Source TBD: Package install of RapidJSON has compile issues
    os.system('(cd '+deps_dir+'; git clone https://github.com/Tencent/rapidjson.git; cd rapidjson; mkdir build; cd build; ' +	
            linuxCMake+' ../; make -j$(nproc); sudo make install)')

    # Python Wheel
    ERROR_CHECK(os.system('pip install wheel'))

# Optional Deps
    # Install OpenCV -- TBD cleanup
    ERROR_CHECK(os.system('(cd '+deps_dir+'/build; mkdir OpenCV )'))
    # Install pre-reqs
    ERROR_CHECK(os.system('sudo '+sudoValidateOption))
    if "Ubuntu" in platfromInfo:
        for i in range(len(opencvDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ opencvDebianPackages[i]))
    else:
        if "centos" in platfromInfo or "redhat" in platfromInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' groupinstall \'Development Tools\''))
        elif "SLES" in platfromInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install -t pattern devel_basis'))
        for i in range(len(opencvRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                        ' '+linuxSystemInstall_check+' install -y '+ opencvRPMPackages[i]))
    # OpenCV 4.6.0
    # Get Installation Source
    ERROR_CHECK(os.system(
        '(cd '+deps_dir+'; wget https://github.com/opencv/opencv/archive/'+opencvVersion+'.zip )'))
    ERROR_CHECK(os.system('(cd '+deps_dir+'; unzip '+opencvVersion+'.zip )'))
    ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; '+linuxCMake +
            ' -D WITH_GTK=ON -D WITH_JPEG=ON -D BUILD_JPEG=ON -D WITH_OPENCL=OFF -D WITH_OPENCLAMDFFT=OFF -D WITH_OPENCLAMDBLAS=OFF -D WITH_VA_INTEL=OFF -D WITH_OPENCL_SVM=OFF  -D CMAKE_INSTALL_PREFIX=/usr/local ../../opencv-'+opencvVersion+' )'))
    ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; make -j$(nproc))'))
    ERROR_CHECK(os.system('sudo '+sudoValidateOption))
    ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; sudo make install)'))
    ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; sudo ldconfig)'))

    # Install ffmpeg
    if "Ubuntu" in platfromInfo:
        for i in range(len(ffmpegDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                                ' '+linuxSystemInstall_check+' install -y '+ ffmpegDebianPackages[i]))

    elif "centos-7" in platfromInfo or "redhat-7" in platfromInfo:
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install epel-release'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' localinstall --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install ffmpeg ffmpeg-devel'))
    elif "centos-8" in platfromInfo or "redhat-8" in platfromInfo:
        # el8 x86_64 packages
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/SDL2-2.0.10-2.el8.x86_64.rpm'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                        ' install ffmpeg ffmpeg-devel'))
    elif "centos-9" in platfromInfo or "redhat-9" in platfromInfo:
        # el8 x86_64 packages
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install https://dl.fedoraproject.org/pub/epel/epel-next-release-latest-9.noarch.rpm'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-$(rpm -E %rhel).noarch.rpm'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install ffmpeg ffmpeg-free-devel'))
    elif "SLES" in platfromInfo:
        # FFMPEG-4 packages
        ERROR_CHECK(os.system(
                    'sudo zypper ar -cfp 90 \'https://ftp.gwdg.de/pub/linux/misc/packman/suse/openSUSE_Leap_$releasever/Essentials\' packman-essentials'))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                    ' install ffmpeg-4'))

print("\nrocAL Dependencies Installed with rocAL-setup.py V-"+__version__+"\n")
