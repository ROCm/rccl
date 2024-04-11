#!/usr/bin/env groovy
// Copyright (c) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
// This shared library is available at https://github.com/ROCm/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()
}

def runCI =
{
    nodeDetails, jobName->

    def prj  = new rocProject('rccl-internal', 'StaticAnalysis')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false
    boolean staticAnalysis = true

    def compileCommand =
    {
        platform, project->

        runCompileCommand(platform, project, jobName, false)
    }

    buildProject(prj , formatCheck, nodes.dockerArray, compileCommand, null, null, staticAnalysis)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 6')])]))
    stage(urlJobName) {
        runCI([ubuntu20:['cpu']], urlJobName)
    }
}
