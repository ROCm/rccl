#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName->

    def prj  = new rocProject('rccl', 'Performance')

    prj.timeout.test = 1440
    prj.paths.build_command = './install.sh -t '

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        //rccl build
        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName)

        //rrcl-tests build

        def command = """#!/usr/bin/env bash
                         set -ex

                         cd ${project.paths.project_build_prefix}
                         export _HIP_HOME=/opt/rocm/hip
                         export _RCCL_HOME=`pwd`/build
                         
                         git clone -b create_scripts_PTS https://github.com/ROCmSoftwarePlatform/rccl-tests.git
                         cd rccl-tests
                         make HIP_HOME=\$_HIP_HOME NCCL_HOME=\$_RCCL_HOME
                      """
        platform.runCommand(this,command)
    }

    def testCommand =
    {
        platform, project->

        def command = """#!/usr/bin/env bash
                         set -ex

                         sudo dmidecode | grep "Product Name"
                         rocm-smi --showtopo

                         cd ${project.paths.project_build_prefix}
                         ls
                         ls rccl-tests

                         ngpus=8
                         cd rccl-tests/build
                         echo "Allreduce Test"
                         ./all_reduce_perf -b 8 -e 1G -f 2 -g \$ngpus
                         echo "Broadcast Test"
                         ./broadcast_perf -b 8 -e 1G -f 2 -g \$ngpus
                         echo "Reduce Test"
                         ./reduce_perf -b 8 -e 1G -f 2 -g \$ngpus
                         echo "Reduce_scatter Test"
                         ./reduce_scatter_perf -b 8 -e 1G -f 2 -g \$ngpus
                         echo "Allgather Test"
                         ./all_gather_perf -b 8 -e 1G -f 2 -g \$ngpus
                         echo "Send_Recv Test"
                         ./sendrecv_perf -b 8 -e 1G -f 2 -g \$ngpus
                         echo "Scatter Test"
                         ./scatter_perf -b 8 -e 1G -f 2 -g \$ngpus
                         echo "Gather Test"
                         ./gather_perf -b 8 -e 1G -f 2 -g \$ngpus
                         echo "Alltoall Test"
                         ./alltoall_perf -b 8 -e 1G -f 2 -g \$ngpus
                         echo "Alltoallv Test"
                         ./alltoallv_perf -b 8 -e 1G -f 2 -g \$ngpus

                      """
        platform.runCommand(this,command)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])]]

    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([sles15sp1:['4gfx906'],centos8:['8gfx908'],centos7:['8gfx906'],ubuntu18:['4gfx906', '4gfx908']])]

    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(urlJobName) {
            runCI([ubuntu18:['8gfx906']], urlJobName)
        }
    }
}
