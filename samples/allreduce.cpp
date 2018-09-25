/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file allreduce.cpp
 * @brief Sample contains how to implement rcclAllReduce
 *
 * @author Aditya Atluri
 */

//! This sample shows how rcclAllReduce can be used in a single process, single
//! thread environment.
//!
//! The sample operates on upto 4 gpus, where it creates n number of source
//! buffers on host and device and n number of destination buffers on host and
//! device.
//!
//! The source buffer on host of length 4 is populated with integer value 1.
//!
//! The destination buffer on host of length 4 is populated with integer value
//! 0.
//!
//! The source buffer on device of length 4 is created and source host buffer is
//! copied to it
//!
//! The destination buffer on device of length 4 is created and destination
//! device buffer is copied to it.
//!
//! HIP streams, rccl communicators are created and multiple rccl calls are
//! launched at the same time.
//!
//! Streams are synchronized on host across all the gpus.
//!
//! Output from rccl ops are copied back to cpu and printed out

#include <hip/hip_runtime_api.h>
#include <rccl/rccl.h>
#include <iostream>
#include <vector>

//! Number of elements involved in rcclAllReduce
const int len = 4;
//! Size of the buffer
const size_t size = len * sizeof(int);

int main() {
    int ndev = 0;
    //! Find number of devices system contains
    hipGetDeviceCount(&ndev);

    //! Force to use no more than 4 gpus
    ndev = std::min(ndev, 4);
    //! Store device indices
    std::vector<int> device_list(ndev);
    for (int i = 0; i < ndev; i++) {
        device_list[i] = i;
    }

    //! multi-gpu pointers for source, destination and host, device
    std::vector<int*> source_host_buffers(ndev), dest_host_buffers(ndev),
        source_device_buffers(ndev), dest_device_buffers(ndev);
    //! multi-gpu streams
    std::vector<hipStream_t> streams(ndev);

    //! prepare data
    for (int i = 0; i < ndev; i++) {
        //! switch to device i
        hipSetDevice(i);

        //! create stream on device i
        hipStreamCreate(&streams[i]);

        //! allocate source host memory corresponding to current gpu
        source_host_buffers[i] = new int[len];

        //! fill source host buffer with integer value 1
        std::fill(source_host_buffers[i], source_host_buffers[i] + len, 1);

        //! allocate destination host memory corresponding to current gpu
        dest_host_buffers[i] = new int[len];

        //! fill destination host buffer with integer value 0
        std::fill(dest_host_buffers[i], dest_host_buffers[i] + len, 0);

        //! allocate source and destination buffers on current gpu
        hipMalloc(&source_device_buffers[i], size);
        hipMalloc(&dest_device_buffers[i], size);

        //! copy host data for source and destination to device buffers
        hipMemcpy(source_device_buffers[i], source_host_buffers[i], size,
                  hipMemcpyHostToDevice);
        hipMemcpy(dest_device_buffers[i], dest_host_buffers[i], size,
                  hipMemcpyHostToDevice);
    }

    //! initialize comms
    std::vector<rcclComm_t> comms(ndev);
    rcclCommInitAll(comms.data(), ndev, device_list.data());

    //! launch allreduce ops on all gpus
    for (int i = 0; i < ndev; i++) {
        hipSetDevice(i);
        rcclAllReduce(source_device_buffers[i], dest_device_buffers[i], len,
                      rcclInt, rcclSum, comms[i], streams[i]);
    }

    //! sync the ops on host and copy output data back to cpu from all gpus
    for (int i = 0; i < ndev; i++) {
        hipSetDevice(i);
        hipStreamSynchronize(streams[i]);
        hipMemcpy(dest_host_buffers[i], dest_device_buffers[i], size,
                  hipMemcpyDeviceToHost);
    }

    //! print output
    for (int i = 0; i < len; i++) {
        std::cout << "[";
        for (int j = 0; j < ndev - 1; j++) {
            std::cout << source_host_buffers[j][i] << " + ";
        }
        std::cout << source_host_buffers[ndev - 1][i]
                  << " ] = " << dest_host_buffers[0][i] << std::endl;
    }

    //! free memory on host and device before exiting
    for (int i = 0; i < ndev; i++) {
        delete dest_host_buffers[i];
        delete source_host_buffers[i];
        hipFree(dest_device_buffers[i]);
        hipFree(source_device_buffers[i]);
        hipStreamDestroy(streams[i]);
    }
}
