/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_AllReduceAbort.hpp"
#include "../include/comm.h"

#define NUM_ITER 8
#define FAKE_OP_COUNT NUM_ITER+1

namespace CorrectnessTests
{
    #define HIPCHECK(cmd)                                                          \
    do {                                                                           \
      hipError_t error = (cmd);                                                    \
      if (error != hipSuccess) {                                                   \
        std::cerr << "Encountered HIP error (" << error << ") at line "            \
                  << __LINE__ << " in file " << __FILE__ << "\n";                  \
        exit(-1);                                                                  \
      }                                                                            \
    } while (0)

    #define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
    #define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

    TEST_P(AllReduceAbortTest, Correctness) {
        if (numDevices > numDevicesAvailable) return;

        // Prepare input / output / expected results
        Dataset dataset;
        dataset.Initialize(numDevices, numElements, dataType, inPlace);
        FillDatasetWithPattern(dataset);

        int gpu = 0; // GPU number to trigger abort
        ncclComm_t comm = comms[gpu];

        HIPCHECK(hipSetDevice(gpu));
        hipStream_t stream;
        HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        struct ncclChannel* channel = comm->channels;
        uint64_t **p_dev_head = (uint64_t **)((uint8_t*)(channel->devPeers + channel->ring.next) + offsetof(struct ncclPeer, send.conn.head));
        uint64_t *real_head, *fake_head, *fake_h;

        // get original head
        HIPCHECK(hipMemcpy(&real_head, p_dev_head, sizeof(uint64_t*), hipMemcpyDefault));
        // allocate and install fakes
        HIPCHECK(hipHostMalloc(&fake_head, sizeof(uint64_t*), hipHostMallocMapped));
        HIPCHECK(hipMemcpy(p_dev_head, &fake_head, sizeof(uint64_t*), hipMemcpyDefault));
        *fake_head = 0;
        // read back fakes to confirm
        HIPCHECK(hipMemcpy(&fake_h, p_dev_head, sizeof(uint64_t*), hipMemcpyDefault));
        //std::cerr << "[          ] replaced gpu " << gpu << " real_opCount = " << real_opCount << " to fake_opCount = " << fake_o << std::endl;
        //std::cerr << "[          ] replaced gpu " << gpu << " real_head = " << real_head << " to fake_head = " << fake_h << std::endl;

        // Perform a number of iterations and introduce abort
        for (int j = 0; j < NUM_ITER; j++) {
            //std::cerr << "[          ] iter = " << j << std::endl;
            // Start a group call
            ncclGroupStart();
            for (int i = 0; i < numDevices; i++) {
                ncclAllReduce(dataset.inputs[i], dataset.outputs[i],
                              numElements, dataType, op, comms[i], streams[i]);
            }
            // Signal end of group call
            ncclGroupEnd();
        }

        // Wait for reduction to complete
        auto start = std::chrono::high_resolution_clock::now();
        hipError_t hipErr;
        int remaining = numDevices;
        int* done = (int*)malloc(sizeof(int)*numDevices);
        memset(done, 0, sizeof(int)*numDevices);
        bool timeout = false, abort_called = false;
        while (remaining) {
            int idle = 1;
            for (int i=0; i<numDevices; i++) {
                if (done[i]) continue;

                hipErr = hipStreamQuery(streams[i]);
                if (hipErr == hipSuccess) {
                    done[i] = 1;
                    remaining--;
                    idle = 0;
                    continue;
                }

 #if NCCL_MAJOR >= 2
 #if NCCL_VERSION_CODE >= NCCL_VERSION(2,4,0)
                auto delta = std::chrono::high_resolution_clock::now() - start;
                double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
                if (deltaSec > 10.0 && !timeout) {
                    std::cerr << "[          ] timeout condition, calling ncclCommAbort ... " << std::endl;
                    timeout = true;
                }
                ncclResult_t ncclAsyncErr;
                ncclCommGetAsyncError(comms[i], &ncclAsyncErr);
                if ((ncclAsyncErr != ncclSuccess || timeout) && !abort_called) {
                    // An asynchronous error happened. Stop the operation and destroy
                    // the communicator
                    std::cerr << "[          ] ncclAsyncErr = " << ncclAsyncErr << std::endl;
                    for (int i=0; i<numDevices; i++)
                      ncclCommAbort(comms[i]);
                    // Abort the perf test
                    abort_called = true;
                    break;
                }
#endif
#endif
            }
            // We might want to let other threads (including NCCL threads) use the CPU.
            if (idle) pthread_yield();
        }

        free(done);
        HIPCHECK(hipHostFree(fake_head));
        HIPCHECK(hipStreamDestroy(stream));
        dataset.Release();
    }

    INSTANTIATE_TEST_SUITE_P(AllReduceAbortSweep,
                            AllReduceAbortTest,
                            testing::Combine(
                                // Reduction operator
                                testing::Values(ncclSum),
                                // Data types
                                testing::Values(ncclFloat32),
                                // Number of elements
                                testing::Values(1024, 1048576),
                                // Number of devices
                                testing::Values(2, 4),
                                // In-place or not
                                testing::Values(false),
                                testing::Values("")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
