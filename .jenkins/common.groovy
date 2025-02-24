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
                cmake -DBACKEND=${backend} ${buildTypeArg} ../..
                make -j\$(nproc)
                sudo cmake --build . --target PyPackageInstall
                sudo make install
                ldd -v /opt/rocm/lib/librocal.so
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project) {

    String libLocation = ''
    if (platform.jenkinsLabel.contains('rhel') || platform.jenkinsLabel.contains('sles')) {
        libLocation = ':/usr/local/lib'
    }

    def command = """#!/usr/bin/env bash
                set -x
                export HOME=/home/jenkins
                echo Make Test
                cd ${project.paths.project_build_prefix}/build/release
                LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} make test ARGS="-VV --rerun-failed --output-on-failure"
                ldd -v /opt/rocm/lib/librocal.so
                """

    platform.runCommand(this, command)
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
