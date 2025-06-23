// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String backend = 'HIP'
    String enableSCL = 'echo rocAL Build'
    String libLocation = ''

    if (platform.jenkinsLabel.contains('centos')) {
        backend = 'CPU'
        if (platform.jenkinsLabel.contains('centos7')) {
            enableSCL = 'source scl_source enable llvm-toolset-7'
        }
    }
    else if (platform.jenkinsLabel.contains('rhel')) {
        libLocation = ':/usr/local/lib'
    }
    else if (platform.jenkinsLabel.contains('ubuntu20')) {
        backend = 'OCL'
    }

    def command = """#!/usr/bin/env bash
                set -x
                ${enableSCL}
                echo Build rocAL - ${buildTypeDir}
                cd ${project.paths.project_build_prefix}
                sudo python rocAL-setup.py --backend ${backend}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                cmake -DBACKEND=${backend} -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" ../..
                make -j\$(nproc)
                sudo cmake --build . --target PyPackageInstall
                sudo make install
                ldd -v /opt/rocm/lib/librocal.so
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project) {

    String libLocation = ''
    String packageManager = 'apt -y'
    String toolsPackage = 'llvm-amdgpu-dev'
    String llvmLocation = '/opt/amdgpu/lib/x86_64-linux-gnu/llvm-20.1/bin'
    
    if (platform.jenkinsLabel.contains('rhel')) {
        libLocation = ':/usr/local/lib:/usr/local/lib/x86_64-linux-gnu'
        packageManager = 'yum -y'
        toolsPackage = 'llvm-amdgpu-devel'
        llvmLocation = '/opt/amdgpu/lib64/llvm-20.1/bin'
    }
    else if (platform.jenkinsLabel.contains('sles')) {
        libLocation = ':/usr/local/lib:/usr/local/lib/x86_64-linux-gnu'
        packageManager = 'zypper -n'
        toolsPackage = 'llvm-amdgpu-devel'
        llvmLocation = '/opt/amdgpu/lib64/llvm-20.1/bin'
    }

    def videoTestArgs = [
        inputFile: "/opt/rocm/share/rocal/test/data/videos/AMD_driving_virtual_20.mp4",
        reader_case: 1,
        processing_device: 0,
        hardware_decode_mode: 0,
        batch_size: 1,
        sequence_length: 3,
        frame_step: 3,
        frame_stride: 1,
        gray_scale_rgb: 1,
        display_on_off: 0,
        shuffle: 1,
        resize_width: 1280,
        resize_height: 720,
        filelist_framenum: 0,
        enable_meta_data: 0,
        enable_framenumber: 0,
        enable_timestamps: 0,
        enable_sequence_rearrange: 0
    ]

    def videoTestArgs1 = videoTestArgs + [
        enable_meta_data: 1,
        enable_framenumber: 1,
        enable_timestamps: 1,
        enable_sequence_rearrange: 1
    ]

    def videoTestArgsFrameNum = videoTestArgs + [
        inputFile: "../../MIVisionX-data-main/rocal_data/video_and_sequence_samples/test_frame/test_frame_num.txt",
        filelist_framenum: 1,
        enable_framenumber: 1,
    ]

    def videoTestArgsTimestamps = videoTestArgs + [
        inputFile: "../../MIVisionX-data-main/rocal_data/video_and_sequence_samples/test_timestamps/test_timestamps.txt",
        enable_timestamps: 1,
    ]

    def videoTestArgsLabelled = videoTestArgs + [
        inputFile: "../../MIVisionX-data-main/rocal_data/video_and_sequence_samples/labelled_videos/",
        resize_width: 640,
        resize_height: 480,
    ]

    def videoTestArgsSequence = videoTestArgs + [
        inputFile: "../../MIVisionX-data-main/rocal_data/video_and_sequence_samples/sequence/",
        reader_case: 3,
    ]

    def videoResizeTestArgs = videoTestArgs + [
        reader_case: 2,
        processing_device: 1,
        hardware_decode_mode: 1,
        shuffle: 0,
        resize_width: 400,
        resize_height: 400,
    ]

    def videoHardwareTestArgs = videoTestArgs + [
        reader_case: 1,
        processing_device: 1,
        hardware_decode_mode: 1,
    ]

    String commitSha
    String repoUrl
    (commitSha, repoUrl) = util.getGitHubCommitInformation(project.paths.project_src_prefix)

    withCredentials([string(credentialsId: "mathlibs-codecov-token-rocal", variable: 'CODECOV_TOKEN')])
    {
        def command = """#!/usr/bin/env bash
                    export HOME=/home/jenkins
                    set -x
                    cd ${project.paths.project_build_prefix}/build
                    export LLVM_PROFILE_FILE=\"\$(pwd)/rawdata/rocal-%p.profraw\"
                    echo \$LLVM_PROFILE_FILE
                    cd release
                    mkdir -p test && cd test
                    cmake /opt/rocm/share/rocal/test/
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ctest -VV --rerun-failed --output-on-failure
                    cd ../
                    wget http://math-ci.amd.com/userContent/computer-vision/MIVisionX-data/MIVisionX-data-main.zip
                    unzip MIVisionX-data-main.zip
                    export ROCAL_DATA_PATH=\$(pwd)/MIVisionX-data-main/
                    mkdir -p rocal-unit-tests && cd rocal-unit-tests
                    python3 -m pip install Pillow
                    cp -r /opt/rocm/share/rocal/test/unit_tests/ .
                    cd unit_tests/
                    chmod +x -R testAllScripts.sh
                    ./testAllScripts.sh
                    cd ../../ && mkdir -p external-source-reader-test && cd external-source-reader-test
                    cmake /opt/rocm/share/rocal/test/external_source/
                    make -j
                    ./external_source ../MIVisionX-data-main/rocal_data/coco/coco_10_img/images/
                    ./external_source ../MIVisionX-data-main/rocal_data/coco/coco_10_img/images/ 1
                    cd ../ && mkdir -p audio-tests && cd audio-tests
                    python3 /opt/rocm/share/rocal/test/audio_tests/audio_tests.py
                    cd ../ && mkdir -p cifar10-dataloader-test && cd cifar10-dataloader-test
                    cmake /opt/rocm/share/rocal/test/dataloader/
                    make -j
                    wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
                    tar xvf cifar-10-binary.tar.gz
                    ./dataloader ./cifar-10-batches-bin/ 0 64 64 1 1
                    ./dataloader ./cifar-10-batches-bin/ 1 64 64 1 1
                    cd ../ && mkdir -p video-tests && cd video-tests
                    cp -r /opt/rocm/share/rocal/test/video_tests/* .
                    mkdir -p build && cd build
                    cmake ..
                    make -j
                    ./video_tests \
                        ${videoTestArgs1.inputFile} ${videoTestArgs1.reader_case} ${videoTestArgs1.processing_device} \
                        ${videoTestArgs1.hardware_decode_mode} ${videoTestArgs1.batch_size} \
                        ${videoTestArgs1.sequence_length} ${videoTestArgs1.frame_step} ${videoTestArgs1.frame_stride} \
                        ${videoTestArgs1.gray_scale_rgb} ${videoTestArgs1.display_on_off} ${videoTestArgs1.shuffle} \
                        ${videoTestArgs1.resize_width} ${videoTestArgs1.resize_height} \
                        ${videoTestArgs1.filelist_framenum} ${videoTestArgs1.enable_meta_data} ${videoTestArgs1.enable_framenumber} \
                        ${videoTestArgs1.enable_timestamps} ${videoTestArgs1.enable_sequence_rearrange}

                    ./video_tests \
                        ${videoTestArgsFrameNum.inputFile} ${videoTestArgsFrameNum.reader_case} ${videoTestArgsFrameNum.processing_device} \
                        ${videoTestArgsFrameNum.hardware_decode_mode} ${videoTestArgsFrameNum.batch_size} \
                        ${videoTestArgsFrameNum.sequence_length} ${videoTestArgsFrameNum.frame_step} ${videoTestArgsFrameNum.frame_stride} \
                        ${videoTestArgsFrameNum.gray_scale_rgb} ${videoTestArgsFrameNum.display_on_off} ${videoTestArgsFrameNum.shuffle} \
                        ${videoTestArgsFrameNum.resize_width} ${videoTestArgsFrameNum.resize_height} \
                        ${videoTestArgsFrameNum.filelist_framenum} ${videoTestArgsFrameNum.enable_meta_data} ${videoTestArgsFrameNum.enable_framenumber} \
                        ${videoTestArgsFrameNum.enable_timestamps} ${videoTestArgsFrameNum.enable_sequence_rearrange}

                    ./video_tests \
                        ${videoTestArgsTimestamps.inputFile} ${videoTestArgsTimestamps.reader_case} ${videoTestArgsTimestamps.processing_device} \
                        ${videoTestArgsTimestamps.hardware_decode_mode} ${videoTestArgsTimestamps.batch_size} \
                        ${videoTestArgsTimestamps.sequence_length} ${videoTestArgsTimestamps.frame_step} ${videoTestArgsTimestamps.frame_stride} \
                        ${videoTestArgsTimestamps.gray_scale_rgb} ${videoTestArgsTimestamps.display_on_off} ${videoTestArgsTimestamps.shuffle} \
                        ${videoTestArgsTimestamps.resize_width} ${videoTestArgsTimestamps.resize_height} \
                        ${videoTestArgsTimestamps.filelist_framenum} ${videoTestArgsTimestamps.enable_meta_data} ${videoTestArgsTimestamps.enable_framenumber} \
                        ${videoTestArgsTimestamps.enable_timestamps} ${videoTestArgsTimestamps.enable_sequence_rearrange}

                    ./video_tests \
                        ${videoTestArgsLabelled.inputFile} ${videoTestArgsLabelled.reader_case} ${videoTestArgsLabelled.processing_device} \
                        ${videoTestArgsLabelled.hardware_decode_mode} ${videoTestArgsLabelled.batch_size} \
                        ${videoTestArgsLabelled.sequence_length} ${videoTestArgsLabelled.frame_step} ${videoTestArgsLabelled.frame_stride} \
                        ${videoTestArgsLabelled.gray_scale_rgb} ${videoTestArgsLabelled.display_on_off} ${videoTestArgsLabelled.shuffle} \
                        ${videoTestArgsLabelled.resize_width} ${videoTestArgsLabelled.resize_height} \
                        ${videoTestArgsLabelled.filelist_framenum} ${videoTestArgsLabelled.enable_meta_data} ${videoTestArgsLabelled.enable_framenumber} \
                        ${videoTestArgsLabelled.enable_timestamps} ${videoTestArgsLabelled.enable_sequence_rearrange}

                    ./video_tests \
                        ${videoTestArgsSequence.inputFile} ${videoTestArgsSequence.reader_case} ${videoTestArgsSequence.processing_device} \
                        ${videoTestArgsSequence.hardware_decode_mode} ${videoTestArgsSequence.batch_size} \
                        ${videoTestArgsSequence.sequence_length} ${videoTestArgsSequence.frame_step} ${videoTestArgsSequence.frame_stride} \
                        ${videoTestArgsSequence.gray_scale_rgb} ${videoTestArgsSequence.display_on_off} ${videoTestArgsSequence.shuffle} \
                        ${videoTestArgsSequence.resize_width} ${videoTestArgsSequence.resize_height} \
                        ${videoTestArgsSequence.filelist_framenum} ${videoTestArgsSequence.enable_meta_data} ${videoTestArgsSequence.enable_framenumber} \
                        ${videoTestArgsSequence.enable_timestamps} ${videoTestArgsSequence.enable_sequence_rearrange}                    

                    ./video_tests \
                        ${videoResizeTestArgs.inputFile} ${videoResizeTestArgs.reader_case} ${videoResizeTestArgs.processing_device} \
                        ${videoResizeTestArgs.hardware_decode_mode} ${videoResizeTestArgs.batch_size} \
                        ${videoResizeTestArgs.sequence_length} ${videoResizeTestArgs.frame_step} ${videoResizeTestArgs.frame_stride} \
                        ${videoResizeTestArgs.gray_scale_rgb} ${videoResizeTestArgs.display_on_off} ${videoResizeTestArgs.shuffle} \
                        ${videoResizeTestArgs.resize_width} ${videoResizeTestArgs.resize_height}                  

                    ./video_tests \
                        ${videoHardwareTestArgs.inputFile} ${videoHardwareTestArgs.reader_case} ${videoHardwareTestArgs.processing_device} ${videoHardwareTestArgs.hardware_decode_mode}
                    cd ..
                    chmod a+x ./testScript.sh
                    ./testScript.sh ../MIVisionX-data-main/rocal_data/video_and_sequence_samples/labelled_videos/ 2
                    ./testScript.sh ../MIVisionX-data-main/rocal_data/video_and_sequence_samples/sequence/ 3
                    cd ../ && mkdir -p image-augmentation-app && cd image-augmentation-app
                    cmake /opt/rocm/share/rocal/test/image_augmentation/
                    make -j
                    ./image_augmentation /opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet/ 0 416 416 0 1 1 1 0 1
                    cd ../ && mkdir -p multiple-dataloaders-test && cd multiple-dataloaders-test
                    cmake /opt/rocm/share/rocal/test/multiple_dataloaders_test/
                    make -j
                    ./multiple_dataloaders_test ../MIVisionX-data-main/rocal_data/numpy/ dataloader_op 
                    cd ../ && mkdir -p python-api-tests && cd python-api-tests
                    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib/
                    export PATH=\$PATH:/opt/rocm/bin
                    export PYTHONPATH=/opt/rocm/lib:\$PYTHONPATH
                    python3 -m pip install opencv-python
                    python3 /opt/rocm/share/rocal/test/python_api/prefetch_queue_depth/prefetch_queue_depth.py /opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet cpu 128 8
                    python3 /opt/rocm/share/rocal/test/python_api/prefetch_queue_depth/prefetch_queue_depth.py /opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet gpu 128 8
                    python3 /opt/rocm/share/rocal/test/python_api/external_source_reader.py cpu 128
                    python3 /opt/rocm/share/rocal/test/python_api/external_source_reader.py gpu 128
                    python3 /opt/rocm/share/rocal/test/python_api/numpy_reader.py --image-dataset-path ../MIVisionX-data-main/rocal_data/numpy/ --no-rocal-gpu
                    python3 /opt/rocm/share/rocal/test/python_api/numpy_reader.py --image-dataset-path ../MIVisionX-data-main/rocal_data/numpy/ --rocal-gpu
                    cd ../../
                    sudo ${packageManager} install lcov ${toolsPackage}
                    ${llvmLocation}/llvm-profdata merge -sparse rawdata/*.profraw -o rocal.profdata
                    ${llvmLocation}/llvm-cov export -object release/lib/librocal.so --instr-profile=rocal.profdata --format=lcov > coverage.info
                    lcov --remove coverage.info '/opt/*' --output-file coverage.info
                    lcov --list coverage.info
                    lcov --summary  coverage.info
                    curl -Os https://uploader.codecov.io/latest/linux/codecov
                    chmod +x codecov
                    ./codecov -v -U \$http_proxy -t ${CODECOV_TOKEN} --file coverage.info --name rocAL --sha ${commitSha}
                    """

        platform.runCommand(this, command)
    }
// Unit tests - TBD
}

def runPackageCommand(platform, project) {

    def packageHelper = platform.makePackage(platform.jenkinsLabel, "${project.paths.project_build_prefix}/build/release")
    
    String packageType = ''
    String packageInfo = ''
    String packageDetail = ''
    String osType = ''
    String packageRunTime = ''

    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('rhel') || platform.jenkinsLabel.contains('sles')) {
        packageType = 'rpm'
        packageInfo = 'rpm -qlp'
        packageDetail = 'rpm -qi'
        packageRunTime = 'rocal-*'

        if (platform.jenkinsLabel.contains('sles')) {
            osType = 'sles'
        }
        else if (platform.jenkinsLabel.contains('centos7')) {
            osType = 'centos7'
        }
        else if (platform.jenkinsLabel.contains('rhel8')) {
            osType = 'rhel8'
        }
        else if (platform.jenkinsLabel.contains('rhel9')) {
            osType = 'rhel9'
        }
    }
    else
    {
        packageType = 'deb'
        packageInfo = 'dpkg -c'
        packageDetail = 'dpkg -I'
        packageRunTime = 'rocal_*'

        if (platform.jenkinsLabel.contains('ubuntu20')) {
            osType = 'ubuntu20'
        }
        else if (platform.jenkinsLabel.contains('ubuntu22')) {
            osType = 'ubuntu22'
        }
    }

    def command = """#!/usr/bin/env bash
                set -x
                export HOME=/home/jenkins
                echo Make rocal Package
                cd ${project.paths.project_build_prefix}/build/release
                sudo make package
                mkdir -p package
                mv rocal-test*.${packageType} package/${osType}-rocal-test.${packageType}
                mv rocal-dev*.${packageType} package/${osType}-rocal-dev.${packageType}
                mv ${packageRunTime}.${packageType} package/${osType}-rocal.${packageType}
                mv Testing/Temporary/LastTest.log ${osType}-LastTest.log
                mv Testing/Temporary/LastTestsFailed.log ${osType}-LastTestsFailed.log
                ${packageDetail} package/${osType}-rocal-test.${packageType}
                ${packageDetail} package/${osType}-rocal-dev.${packageType}
                ${packageDetail} package/${osType}-rocal.${packageType}
                ${packageInfo} package/${osType}-rocal-test.${packageType}
                ${packageInfo} package/${osType}-rocal-dev.${packageType}
                ${packageInfo} package/${osType}-rocal.${packageType}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
