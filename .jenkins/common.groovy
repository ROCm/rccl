// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    def build_command = ${project.paths.build_command}
    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        pullRequest.labels.each
        {
            if (it == "debugCI")
            {
                build_command = ${project.paths.build_command_debug}
            }
        }
    }

    def command = """#!/usr/bin/env bash
                      set -x
                      cd ${project.paths.project_build_prefix}
                      LD_LIBRARY_PATH=/opt/rocm/hcc/lib ${build_command}
                  """

    platform.runCommand(this,command)
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def unit_test_path = "${project.paths.project_build_prefix}/build/release/test"
    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        pullRequest.labels.each
        {
            if (it == "debugCI")
            {
                unit_test_path = "${project.paths.project_build_prefix}/build/debug/test"
            }
        }
    }

    def command = """#!/usr/bin/env bash
                     set -x
                     cd ${unit_test_path}
                     ${sudo} ulimit -l unlimited
                     ulimit -a
                     ${sudo} NCCL_DEBUG=INFO HSA_FORCE_FINE_GRAIN_PCIE=1 ./UnitTests --gtest_filter=${gfilter} --gtest_output=xml --gtest_color=yes
                  """

   platform.runCommand(this, command)
   junit "${unit_test_path}/*.xml"
}

def runPackageCommand(platform, project, jobName)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")

    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
