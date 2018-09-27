/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl/rccl.h"
#include "performance/performance.h"
#include <iostream>
#include <vector>
#include "common.h"
#include <typeinfo>
#include <list>
#include <algorithm>

void CallAllReduce(signed char* psrc_buff, signed char* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclChar, op, comm, stream));
}

void CallAllReduce(unsigned char* psrc_buff, unsigned char* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclUchar, op, comm, stream));
}

void CallAllReduce(signed short* psrc_buff, signed short* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclShort, op, comm, stream));
}

void CallAllReduce(unsigned short* psrc_buff, unsigned short* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclUshort, op, comm, stream));
}

void CallAllReduce(signed int* psrc_buff, signed int* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclInt, op, comm, stream));
}

void CallAllReduce(unsigned int* psrc_buff, unsigned int* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclUint, op, comm, stream));
}

void CallAllReduce(signed long* psrc_buff, signed long* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclLong, op, comm, stream));
}

void CallAllReduce(unsigned long* psrc_buff, unsigned long* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclUlong, op, comm, stream));
}

void CallAllReduce(__fp16* psrc_buff, __fp16* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclHalf, op, comm, stream));
}

void CallAllReduce(float* psrc_buff, float* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclFloat, op, comm, stream));
}

void CallAllReduce(double* psrc_buff, double* pdst_buff, size_t buff_len, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclDouble, op, comm, stream));
}

template<typename T>
void DoAllReduce(std::vector<int>& device_list, std::vector<hipStream_t>& device_streams,
    std::vector<rcclComm_t>& rccl_comms, std::vector<void*>& src_host_buffers,
    std::vector<void*>& src_device_buffers, std::vector<void*>& dst_host_buffers,
    std::vector<void*>& dst_device_buffers, size_t buff_size) {

    size_t buff_len = buff_size / sizeof(T);
    size_t num_gpus = device_list.size();

    for(int i = 0; i < device_list.size(); i++) {
        for(size_t j = 0; j < buff_len; j++) {
            reinterpret_cast<T*>(src_host_buffers[i])[j] = static_cast<T>(kbuffer_values[device_list[i]]);
            reinterpret_cast<T*>(dst_host_buffers[i])[j] = static_cast<T>(kbuffer_values[device_list[i]]);
        }
        HIPCHECK(hipSetDevice(device_list[i]));
        HIPCHECK(hipMemcpy(src_device_buffers[i], src_host_buffers[i], buff_size, hipMemcpyHostToDevice));
    }
    for(auto p_ops = umap_rccl_op.begin(); p_ops != umap_rccl_op.end(); p_ops++) {
        perf_marker mark;
        for(size_t iter_id = 0; iter_id < knum_iter; iter_id++) {
        for(size_t i = 0; i < num_gpus; i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            CallAllReduce(reinterpret_cast<T*>(src_device_buffers[i]), reinterpret_cast<T*>(dst_device_buffers[i]), buff_len, p_ops->second, rccl_comms[i], device_streams[i]);
        }
        }
        for(size_t i = 0; i < num_gpus; i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamSynchronize(device_streams[i]));
        }
        mark.done();
        mark.bw(buff_size*knum_iter*num_gpus);
    }
}

void RandomReduceTest(std::vector<int>& device_list, int num_tests) {
    size_t num_gpus = device_list.size();
    EnableDevicePeerAccess(device_list);

    std::vector<rcclComm_t> rccl_comms(num_gpus);
    RCCLCHECK(rcclCommInitAll(rccl_comms.data(), num_gpus, device_list.data()));

    std::list<size_t> buffer_lengths; // in bytes
    size_t max_allocated_memory = 0;
    RandomSizeGen_t rsg(num_tests, 0, kmax_buffer_size);
    for(int i = 0; i < num_tests; i++) {
        size_t val = rsg.GetSize();
        max_allocated_memory = std::max(max_allocated_memory, val);
        buffer_lengths.push_back(val);
    }

    std::vector<void*> src_device_buffers(num_gpus);
    std::vector<void*> src_host_buffers(num_gpus);
    std::vector<void*> dst_device_buffers(num_gpus);
    std::vector<void*> dst_host_buffers(num_gpus);

    std::vector<hipStream_t> device_streams(num_gpus);

    for(int i = 0; i < device_list.size(); i++) {
        src_host_buffers[i] = reinterpret_cast<void*>(new signed char[max_allocated_memory]);
        dst_host_buffers[i] = reinterpret_cast<void*>(new signed char[max_allocated_memory]);
    }

    { // used new scope to force current-device guard to destruct after changing active device
        CurrDeviceGuard_t g;
        for(int i = 0; i < device_list.size(); i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamCreate(&device_streams[i]));
            HIPCHECK(hipMalloc(&(src_device_buffers[i]), max_allocated_memory));
            HIPCHECK(hipMalloc(&(dst_device_buffers[i]), max_allocated_memory));
        }
    }

     for(auto pbuff_len = buffer_lengths.begin(); pbuff_len != buffer_lengths.end(); pbuff_len++) {
        DoAllReduce<signed char>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<unsigned char>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<signed short>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<unsigned short>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<signed int>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<unsigned int>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<signed long>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<unsigned long>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<float>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<double>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<__fp16>(device_list, device_streams, rccl_comms, src_host_buffers, src_device_buffers, dst_host_buffers, dst_device_buffers, *pbuff_len);
    }

// free allocted buffers on both host and device

    for(auto iter = src_device_buffers.begin(); iter != src_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for(auto iter = src_host_buffers.begin(); iter != src_host_buffers.end(); iter++) {
        delete reinterpret_cast<signed char*>(*iter);
    }

    for(auto iter = dst_device_buffers.begin(); iter != dst_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for(auto iter = dst_host_buffers.begin(); iter != dst_host_buffers.end(); iter++) {
        delete reinterpret_cast<signed char*>(*iter);
    }

}

int main(int argc, char* argv[]) {
    if(argc > 1) {
    if(strcmp(argv[1], "-r") == 0) {
        if(argc > 4) {
            int num_tests = atoi(argv[2]);
            int num_gpus = atoi(argv[3]);
            std::vector<int> device_list(num_gpus);
            if(argc == num_gpus + 4) {
                for(int i = 0; i < num_gpus; i++) {
                    device_list[i] = atoi(argv[i+4]);
                }
                RandomReduceTest(device_list, num_tests);
            } else {
                print_out("The size of gpus in list is less than specified length");
            }
        }
        return 0;
    }
    }
    if(argc != 3 || argc != 6) {
        std::cout<<"Usage: ./a.out -r <num tests> <root gpu> <num gpus> <list of gpus>"<<std::endl;
        std::cout<<"./a.out -r 99 1 3 1 2 3"<<std::endl;
        std::cout<<"[-b] enables validation across random generated data sizes, uses <num tests> as seed"<<std::endl;
        std::cout<<"./a.out -r <num gpus> <number of elements> <op> <datatype>"<<std::endl;
        std::cout<<"Example: ./a.out -s 4 1024 rcclSum rcclInt"<<std::endl;
        std::cout<<"[-r] enabled validation for certain number of gpus, elements, ops and datatypes"<<std::endl;
        return 0;
    }
}
