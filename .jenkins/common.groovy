// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib ${project.paths.build_command}
            """

    platform.runCommand(this,command)
}

def runNPKITCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    def command = """#!/usr/bin/env bash
                set -x
                mdkir ${project.paths.project_build_prefix}/build
                cd ${project.paths.project_build_prefix}/build
                CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ .. -DNPKIT_FLAGS="-DENABLE_NPKIT -DENABLE_NPKIT_EVENT_TIME_SYNC_CPU -DENABLE_NPKIT_EVENT_TIME_SYNC_GPU -DENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY -DENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT -DENABLE_NPKIT_EVENT_NET_SEND_ENTRY -DENABLE_NPKIT_EVENT_NET_SEND_EXIT -DENABLE_NPKIT_EVENT_NET_RECV_ENTRY -DENABLE_NPKIT_EVENT_NET_RECV_EXIT"
                make -j
                rm -rf *}
            """

    platform.runCommand(this,command)
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release/test
                ${sudo} ulimit -l unlimited
                ulimit -a
                ${sudo} RCCL_ENABLE_SIGNALHANDLER=1 NCCL_DEBUG=INFO HSA_FORCE_FINE_GRAIN_PCIE=1 ./rccl-UnitTests --gtest_filter=${gfilter} --gtest_output=xml --gtest_color=yes
            """

   platform.runCommand(this, command)
   junit "${project.paths.project_build_prefix}/build/release/test/*.xml"
}

def runPackageCommand(platform, project, jobName)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")

    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
