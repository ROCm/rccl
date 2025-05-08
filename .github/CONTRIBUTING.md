<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to RCCL">
  <meta name="keywords" content="ROCm, contributing, RCCL">
</head>

# Contributing to RCCL #

We welcome contributions to RCCL.  Please follow these details to help ensure your contributions will be successfully accepted.

## Issue Discussion ##

Please use the GitHub Issues tab to notify us of issues.

* Use your best judgement for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When filing an issue, be sure to provide as much information as possible, including script output so
    we can collect information about your configuration. This helps reduce the time required to
    reproduce your issue.
  * Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.
* You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

<!-- [PART1] Instructions:
Above is fairly standard.  Please add any special instructions (if any) your team may have regarding issues -->

## Acceptance Criteria ##

<!-- [PART2] Instructions:
Include details of your library such as:
   1. The overarching goals of a library
   2. The sample features that fulfils those goals
   3. Any performance criteria needed for those features.

The specification of performance criteria shouldn't include exact numeric values i.e x% perf gains.
However, it should include some language suggesting the need of notable performance gain to avoid
code duplication and code maintenance if no gain is found. Contributors should be guided about
any performance benchmarking mechanism including baseline comparison.

Additionally, to encourage contributors, we should point out that the GitHub "Issues" tab could be
leveraged to discuss features they would like to bring in. Or use it to discuss changes, along with the
perf gains it would bring, before raising pull requests.

MIOpen Example:
The goal of MIOpen library is to provide the optimized implementation of ops found
in machine learning models. The library offers the implementation of either fwd or
bwd or both pass. It also supports the fusion of multiple ops.

Contributors wanting to submit new implementations of ops, should follow
the below-mentioned guidelines.

* New kernels - not present in MIOpen/
   + Level-1 bar: The critical ops found in machine learning ops, found in cuDNN, cuTLASS
     but missing in MIOpen.
   + If it meets this Level-1 bar,
        - If new op is present in frameworks such as rocm/pytorch, rocm/tf, deepspeed or
            fastertransformer either as a hipified version or a standalone version,
            it would be accepted
            + Level-2 perf bar:
                - If notable performance gain is demonstrated on average (geomean)
                  against the baseline at per-op level.
                - If notable performance gain is demonstrated at end-to-end model level
                  where minor gains accumulated by frequent op-level invocations could lead to
                  noticeable model level improvement.
        - If new op is not present in any frameworks or ROCm libraries
            + No performance gating factor
   + If it doesn't meet this Level-1 bar, **exceptions** are allowed which could be discussed
      via Issue tab with the community.
* Overlapping kernels - present in MIOpen
   + AMD would accept op/kernel, based on the Level-2 perf bar defined above.

-->

### Exceptions ###

<!-- [PART3] Instructions (optional):
It may be necessary to state the exceptions permitted within the perimeter of
acceptance criteria and specify a particular set of guidelines developers could
follow to meet those exceptions.
-->

## Code Structure ##

<!-- [PART4] Instructions (optional):
In this section, one should explain the organization of code - its directory structure.
It might not be necessary to cover each and every directory but very important to
specify the external API directory; key source, test, performance and doc directories;
third-party and output directories.
-->

## Coding Style ##

<!-- [PART5] Instructions (optional):
Each library has its own coding style.  If there are particular style details you would like to highlight,
this is the section to document it.  Try to be concise and just highlight big do/don't.
If coding style is too long, the contributor won't bother with submitting a PR.
Remember, your team will be able to review PRs so that's your chance to correct any coding style changes later.
-->

## Pull Request Guidelines ##

By creating a pull request, you agree to the statements made in the [code license](#code-license) section. Your pull request should target the default branch. Our current default branch is the **develop** branch, which serves as our integration branch.


<!-- [PART6] Instructions:
This section should offer guidelines to follow when raising a pull request.
If you're one of the few repos that don't use the develop branch as default, you should update above text to your default branch.
You can also base this section on the generic steps provided in https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md, but append it with more details about your library's methods.

MIOpen Example:
MIOpen accepts the implementation of kernels in either OpenCL or HIP. It is recommended to
leverage HIP for modularity and scalability purpose e.g. AMD's composable kernel library is
built in HIP.
-->

### Deliverables ###

<!-- [PART7] Instructions:
   * Kinds of tests - unit, integration & perf, a feature should accompany
   * External API documentation updates
   * Provide a snapshot of license header or a link to license header for files
   * Commit message format or its specification

MIOpen Example:
1. Here are the key components required in kernel's pull request.
+ Each kernel needs to support these datatypes
     - Until ROCm5.6,  fp16, bfloat16, fp32
     - Starting ROCm5.7, f8
+ Introduce the host-side API, if required
E.g. For LayerNorm, https://github.com/ROCm/MIOpen/blob/develop/include/miopen/miopen.h
+ Add unit tests in gtest format
Dir: https://github.com/ROCm/MIOpen/tree/5c8118ee4339c2aae55f20241440af207e59d2a7/test/gtest
+ Include the relevant code in driver to measure per-kernel performance
https://github.com/ROCm/MIOpen/tree/master/driver
+ Post the performance # against the baseline kernel.
+ For new kernels in MIOpen, e.g. LayerNorm 
   - obtain the baseline from ROCm/Pytorch build: https://github.com/ROCm/pytorch/blob/main/aten/src
/ATen/native/cuda/layer_norm_kernel.cu
   - obtain the baseline from ROCm/Composable kernel library.
+ For overlapping kernels in MIOpen,
   - obtain the baseline by running MIOpen/driver code

1. For each new file in repository, 
Please include the licensing header
```
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 20xx Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
```
-->

### Process ###

<!-- [PART8] Instructions:
   * Outline the PR review process, most importantly, ensure list of reviewers are updated in CODEOWNERS file
   * List out code formatting steps (if script is used to check formatting) or refer contributor to coding-style section if you've documented it there already.
   * Detail how to get and run you test suite
   * Explain the external CI/CD process (if one exists at the time)
       - different hardware targets under test
       - access mechanism to CI logs
       - typical failures and resolution suggestion

MIOpen Example:
* Please follow the generic MIOpen PR review process, as defined here.
* Reviewers are listed in the CODEOWNERS file
* For each commit message, please follow this recommend guidelines [here](https://github.com/ROCm/MIOpen/wiki/How-to-assign-labels-to-Commit-Messages-and-PRs%3F).
* Code format guidelines

MIOpen uses the clang-format tool for formatting code in source files.
The tool is installed as part of dependency installation of MIOpen 
The formatting style is captured in .clang-format which is located at
the root of MIOpen. These are different options to follow:
   1. Run the command -  clang-format-12 -style=file -i <path-to-the-source-file>
   1. For multiple files in your stashed commit, run the composed bash command
                described here 
   1. Automate running this command, upon git commit, use ./.githooks/install

* Static Code Analysis

MIOpen uses cppcheck & tidy and it is recommend to run before committing the code.
The tool is installed as part of dependency installation of MIOpen 
   1. Run 'make tidy' & 'make cppcheck' separately inside the "build" directory
   1. Run 'make analyze' inside the "build" directory

-->

## Code License ##

All code contributed to this project will be licensed under the license identified in the [License.txt](../LICENSE.txt). Your contribution will be accepted under the same license.

## Release Cadence ##

<!-- [PART9] Instructions (optional):
This section should explain the release process of a library and how it intersects
with ROCm releases.

It should inform the developers about the release process including the following elements,

1. Which is a chosen branch used to create a release branch?
1. Inform about the release cadence.
1. Any notification sent to the community about the upcoming release and its goals.
1. Explain the bug resolution mechanism on the release branch.
-->

## References ##

<!-- [PART10] Instructions (optional):
This section captures the links to GitHub or ROCm documentation where the developers
could obtain the detailed information.

MIOpen Example:
1. [MIOpen Wiki](https://github.com/ROCm/MIOpen/wiki)
1. [MIOpen Readme](https://github.com/ROCm/MIOpen#readme)
-->
