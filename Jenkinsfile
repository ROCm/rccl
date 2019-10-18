#!/usr/bin/env groovy
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rccl
@Library('rocJenkins@rccl') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])


////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rcclCI:
{

    def rccl = new rocProject('rccl')
    // customize for project
    rccl.paths.build_command = './install.sh'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['rccl && ubuntu'], rccl)

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        def command = """#!/usr/bin/env bash
                  set -x
                  cd ${project.paths.project_build_prefix}
                  LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX= ${project.paths.build_command} -t
                """

        platform.runCommand(this,command)
    }

    def testCommand =
    {
        platform, project->

        def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release/test/
                HSA_FORCE_FINE_GRAIN_PCIE=1 ./UnitTests --gtest_output=xml --gtest_color=yes
            """

        platform.runCommand(this,command)
        //junit "${project.paths.project_build_prefix}/build/release/*.xml"
    }

    def packageCommand =
    {
        platform, project->

        def command = """
                      set -x
                      cd ${project.paths.project_build_prefix}/build/release
                      make package
                      rm -rf package && mkdir -p package
                      mv *.deb package/
                      """
	platform.runCommand(this,command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
    }

    buildProject(rccl, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}
