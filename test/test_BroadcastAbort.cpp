/*************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_BroadcastAbort.hpp"
#include "../include/core.h"
#include <omp.h>

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

    TEST_P(BroadcastAbortTest, Correctness) {
        if (numDevices > numDevicesAvailable) return;

        // Prepare input / output / expected results
        Dataset dataset;
        dataset.Initialize(numDevices, numElements, dataType, inPlace);
        FillDatasetWithPattern(dataset);

        int root = 0;
        int gpu = 0; // GPU number to trigger abort
        ncclComm_t comm = comms[gpu];

        HIPCHECK(hipSetDevice(gpu));
        hipStream_t stream;
        HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        struct ncclChannel* channel = comm->channels;
        struct ncclRing *ring = &channel->ring;
        struct ncclConnector* send = &channel->peers[ring->next].send;
        size_t op_offset = &(send->conn.opCountRem) - (uint64_t **)channel->peers;
        size_t head_offset = &(send->conn.head) - (uint64_t **)channel->peers;
        uint64_t **p_dev_opCount = (uint64_t **)(channel->devPeers) + op_offset;
        uint64_t **p_dev_head = (uint64_t **)(channel->devPeers) + head_offset;
        uint64_t *real_opCount, *fake_opCount, *fake_o;
        uint64_t *real_head, *fake_head, *fake_h;

        // get original opCount and head
        HIPCHECK(hipMemcpyAsync(&real_opCount, p_dev_opCount, sizeof(uint64_t*), hipMemcpyDeviceToHost, stream));
        HIPCHECK(hipMemcpyAsync(&real_head, p_dev_head, sizeof(uint64_t*), hipMemcpyDeviceToHost, stream));
        HIPCHECK(hipStreamSynchronize(stream));
        // allocate and install fakes
        HIPCHECK(hipHostMalloc(&fake_opCount, sizeof(uint64_t*), hipHostMallocMapped));
        HIPCHECK(hipMemcpyAsync(p_dev_opCount, &fake_opCount, sizeof(uint64_t*), hipMemcpyHostToDevice, stream));
        *fake_opCount = FAKE_OP_COUNT;
        HIPCHECK(hipHostMalloc(&fake_head, sizeof(uint64_t*), hipHostMallocMapped));
        HIPCHECK(hipMemcpyAsync(p_dev_head, &fake_head, sizeof(uint64_t*), hipMemcpyHostToDevice, stream));
        *fake_head = 0;
        HIPCHECK(hipStreamSynchronize(stream));
        // read back fakes to confirm
        HIPCHECK(hipMemcpyAsync(&fake_o, p_dev_opCount, sizeof(uint64_t*), hipMemcpyDeviceToHost, stream));
        HIPCHECK(hipMemcpyAsync(&fake_h, p_dev_head, sizeof(uint64_t*), hipMemcpyDeviceToHost, stream));
        HIPCHECK(hipStreamSynchronize(stream));
        //std::cerr << "[          ] replaced gpu " << gpu << " real_opCount = " << real_opCount << " to fake_opCount = " << fake_o << std::endl;
        //std::cerr << "[          ] replaced gpu " << gpu << " real_head = " << real_head << " to fake_head = " << fake_h << std::endl;

        // Perform a number of iterations and introduce abort
        for (int j = 0; j < NUM_ITER; j++) {
            //std::cerr << "[          ] iter = " << j << std::endl;
            // Start a group call
            ncclGroupStart();
            for (int i = 0; i < numDevices; i++) {
                ncclBroadcast(dataset.inputs[i],
                              dataset.outputs[i],
                              numElements, dataType,
                              root, comms[i], streams[i]);
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

        HIPCHECK(hipHostFree(fake_opCount));
        HIPCHECK(hipStreamDestroy(stream));
        dataset.Release();
    }

    INSTANTIATE_TEST_CASE_P(BroadcastAbortSweep,
                            BroadcastAbortTest,
                            testing::Combine(
                                // Reduction operator
                                testing::Values(ncclSum),
                                // Data types
                                testing::Values(ncclFloat32),
                                // Number of elements
                                testing::Values(1048576),
                                // Number of devices
                                testing::Values(2, 4),
                                // In-place or not
                                testing::Values(false)));
} // namespace
