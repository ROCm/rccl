/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                   \
    do {                                                                \
        hipError_t error = (cmd);                                       \
        if (error != hipSuccess)                                        \
        {                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                   \
        }                                                               \
    } while (0)

#define MAX_NAME_LEN 64
#define BLOCKSIZE 256
#define COPY_UNROLL 4

// Each link is defined between a source GPU and destination GPU
struct Link
{
    int srcGpu;         // Source GPU      (global memory source)
    int dstGpu;         // Destination GPU (fine-grained memory destination)
    int numBlocksToUse; // Number of threadblocks to use for this link
};

// Each threadblock copies N floats from src to dst
struct BlockParam
{
    int N;
    float* src;
    float* dst;
};

// GPU copy kernel
__global__ void __launch_bounds__(BLOCKSIZE)
CopyKernel(BlockParam* blockParams)
{
    // Collect the arguments for this block
    int N = blockParams[blockIdx.x].N;
    const float* __restrict__ src = (float* )blockParams[blockIdx.x].src;
    float* __restrict__ dst = (float* )blockParams[blockIdx.x].dst;

    Copy<COPY_UNROLL, BLOCKSIZE>(dst, src, N);
}

// Helper function to parse a link of link definitions
void ParseLinks(char const* line, std::vector<Link>& links)
{
    links.clear();
    int numLinks = 0;

    std::istringstream iss;
    iss.clear();
    iss.str(line);
    iss >> numLinks;
    links.resize(numLinks);
    if (iss.fail()) return;


    for (int i = 0; i < numLinks; i++)
        iss >> links[i].srcGpu >> links[i].dstGpu >> links[i].numBlocksToUse;
}

// Helper function to either fill a device pointer with pseudo-random data, or to check to see if it matches
void CheckOrFill(int N, float* devPtr, bool doCheck)
{
    float* refBuffer = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
        refBuffer[i] = i % 383 + 31;

    if (doCheck)
    {
        float* hostBuffer = (float*) malloc(N * sizeof(float));
        HIP_CALL(hipMemcpy(hostBuffer, devPtr, N * sizeof(float), hipMemcpyDeviceToHost));
        for (int i = 0; i < N; i++)
        {
            if (refBuffer[i] != hostBuffer[i])
            {
                printf("[ERROR] Mismatch at element %d Ref: %f Actual: %f\n", i, refBuffer[i], hostBuffer[i]);
                exit(1);
            }
        }
    }
    else
    {
        HIP_CALL(hipMemcpy(devPtr, refBuffer, N * sizeof(float), hipMemcpyHostToDevice));
    }
    free(refBuffer);
}
