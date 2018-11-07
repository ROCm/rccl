/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include <algorithm>
#include <iostream>
#include <list>
#include <typeinfo>
#include <vector>
#include "common.h"
#include "performance/performance.h"
#include "rccl/rccl.h"

void CallReduce(signed char* psrc_buff, signed char* pdst_buff, size_t buff_len,
                rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(psrc_buff, pdst_buff, buff_len, rcclChar, op, root,
                         comm, stream));
}

void CallReduce(signed int* psrc_buff, signed int* pdst_buff, size_t buff_len,
                rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(psrc_buff, pdst_buff, buff_len, rcclInt, op, root,
                         comm, stream));
}

void CallReduce(signed long* psrc_buff, signed long* pdst_buff, size_t buff_len,
                rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(psrc_buff, pdst_buff, buff_len, rcclInt64, op, root,
                         comm, stream));
}

void CallReduce(unsigned long* psrc_buff, unsigned long* pdst_buff,
                size_t buff_len, rcclRedOp_t op, int root, rcclComm_t comm,
                hipStream_t stream) {
    RCCLCHECK(rcclReduce(psrc_buff, pdst_buff, buff_len, rcclUint64, op, root,
                         comm, stream));
}

void CallReduce(__fp16* psrc_buff, __fp16* pdst_buff, size_t buff_len,
                rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(psrc_buff, pdst_buff, buff_len, rcclHalf, op, root,
                         comm, stream));
}

void CallReduce(float* psrc_buff, float* pdst_buff, size_t buff_len,
                rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(psrc_buff, pdst_buff, buff_len, rcclFloat, op, root,
                         comm, stream));
}

void CallReduce(double* psrc_buff, double* pdst_buff, size_t buff_len,
                rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(psrc_buff, pdst_buff, buff_len, rcclDouble, op, root,
                         comm, stream));
}

template <typename T>
void DoReduce(std::vector<int>& device_list,
              std::vector<hipStream_t>& device_streams,
              std::vector<rcclComm_t>& rccl_comms,
              std::vector<void*>& host_buffers,
              std::vector<void*>& device_buffers, void*& dst_host_buffer,
              void*& dst_device_buffer, size_t buff_size, int root) {
    size_t buff_len = buff_size / sizeof(T);
    size_t num_gpus = device_list.size();
    for (size_t i = 0; i < buff_len; i++) {
        reinterpret_cast<T*>(dst_host_buffer)[i] = static_cast<T>(0);
    }
    for (int i = 0; i < device_list.size(); i++) {
        for (size_t j = 0; j < buff_len; j++) {
            reinterpret_cast<T*>(host_buffers[i])[j] =
                static_cast<T>(kbuffer_values[device_list[i]]);
        }
        HIPCHECK(hipSetDevice(device_list[i]));
        HIPCHECK(hipMemcpy(device_buffers[i], host_buffers[i], buff_size,
                           hipMemcpyHostToDevice));
    }
    for (auto p_ops = umap_rccl_op.begin(); p_ops != umap_rccl_op.end();
         p_ops++) {
        perf_marker mark;
        for (size_t iter_id = 0; iter_id < knum_iter; iter_id++) {
            for (size_t i = 0; i < num_gpus; i++) {
                HIPCHECK(hipSetDevice(device_list[i]));
                CallReduce(reinterpret_cast<T*>(device_buffers[i]),
                           reinterpret_cast<T*>(dst_device_buffer), buff_len,
                           p_ops->second, root, rccl_comms[i],
                           device_streams[i]);
            }
        }
        for (size_t i = 0; i < num_gpus; i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamSynchronize(device_streams[i]));
        }
        mark.done();
        mark.bw(buff_size * knum_iter);
    }
}

void RandomReduceTest(std::vector<int>& device_list, int num_tests, int root) {
    size_t num_gpus = device_list.size();
    EnableDevicePeerAccess(device_list);

    std::vector<rcclComm_t> rccl_comms(num_gpus);
    RCCLCHECK(rcclCommInitAll(rccl_comms.data(), num_gpus, device_list.data()));

    std::list<size_t> buffer_lengths;  // in bytes
    size_t max_allocated_memory = 0;
    RandomSizeGen_t rsg(num_tests, 0, kmax_buffer_size);
    for (int i = 0; i < num_tests; i++) {
        size_t val = rsg.GetSize();
        max_allocated_memory = std::max(max_allocated_memory, val);
        buffer_lengths.push_back(val);
    }

    std::vector<void*> device_buffers(num_gpus);
    std::vector<void*> host_buffers(num_gpus);
    void *dst_device_buffer, *dst_host_buffer;

    std::vector<hipStream_t> device_streams(num_gpus);

    {
        dst_host_buffer =
            reinterpret_cast<void*>(new signed char[max_allocated_memory]);
        CurrDeviceGuard_t g;
        HIPCHECK(hipSetDevice(root));
        HIPCHECK(hipMalloc(&dst_device_buffer, max_allocated_memory));
    }

    for (int i = 0; i < device_list.size(); i++) {
        host_buffers[i] =
            reinterpret_cast<void*>(new signed char[max_allocated_memory]);
    }

    {  // used new scope to force current-device guard to destruct after
       // changing active device
        CurrDeviceGuard_t g;
        for (int i = 0; i < device_list.size(); i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamCreate(&device_streams[i]));
            HIPCHECK(hipMalloc(&(device_buffers[i]), max_allocated_memory));
        }
    }

    for (auto pbuff_len = buffer_lengths.begin();
         pbuff_len != buffer_lengths.end(); pbuff_len++) {
        DoReduce<signed char>(device_list, device_streams, rccl_comms,
                              host_buffers, device_buffers, dst_host_buffer,
                              dst_device_buffer, *pbuff_len, root);
        DoReduce<signed int>(device_list, device_streams, rccl_comms,
                             host_buffers, device_buffers, dst_host_buffer,
                             dst_device_buffer, *pbuff_len, root);
        DoReduce<signed long>(device_list, device_streams, rccl_comms,
                              host_buffers, device_buffers, dst_host_buffer,
                              dst_device_buffer, *pbuff_len, root);
        DoReduce<unsigned long>(device_list, device_streams, rccl_comms,
                                host_buffers, device_buffers, dst_host_buffer,
                                dst_device_buffer, *pbuff_len, root);
        DoReduce<float>(device_list, device_streams, rccl_comms, host_buffers,
                        device_buffers, dst_host_buffer, dst_device_buffer,
                        *pbuff_len, root);
        DoReduce<double>(device_list, device_streams, rccl_comms, host_buffers,
                         device_buffers, dst_host_buffer, dst_device_buffer,
                         *pbuff_len, root);
        DoReduce<__fp16>(device_list, device_streams, rccl_comms, host_buffers,
                         device_buffers, dst_host_buffer, dst_device_buffer,
                         *pbuff_len, root);
    }

    // free allocted buffers on both host and device
    HIPCHECK(hipFree(dst_device_buffer));
    delete reinterpret_cast<signed char*>(dst_host_buffer);

    for (auto iter = device_buffers.begin(); iter != device_buffers.end();
         iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for (auto iter = host_buffers.begin(); iter != host_buffers.end(); iter++) {
        delete reinterpret_cast<signed char*>(*iter);
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        if (strcmp(argv[1], "-r") == 0) {
            if (argc > 5) {
                int num_tests = atoi(argv[2]);
                int num_gpus = atoi(argv[4]);
                int root_gpu = atoi(argv[3]);
                std::vector<int> device_list(num_gpus);
                if (argc == num_gpus + 5) {
                    for (int i = 0; i < num_gpus; i++) {
                        device_list[i] = atoi(argv[i + 5]);
                    }
                    RandomReduceTest(device_list, num_tests, root_gpu);
                } else {
                    print_out(
                        "The size of gpus in list is less than specified "
                        "length");
                }
            }
            return 0;
        }
    }
    if (argc != 3 || argc != 6) {
        std::cout << "Usage: ./a.out -r <num tests> <root gpu> <num gpus> "
                     "<list of gpus>"
                  << std::endl;
        std::cout << "./a.out -r 99 1 3 1 2 3" << std::endl;
        return 0;
    }
}
