/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl/rccl.h"
//#include "rcclCheck.h"
#include "validation/validate.h"
#include <iostream>
#include <vector>
#include "common.h"
#include <typeinfo>
#include <list>
#include <algorithm>

void CallBcast(signed char* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclChar, root, comm, stream));
}

void CallBcast(unsigned char* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclUchar, root, comm, stream));
}

void CallBcast(signed short* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclShort, root, comm, stream));
}

void CallBcast(unsigned short* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclUshort, root, comm, stream));
}

void CallBcast(signed int* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclInt, root, comm, stream));
}

void CallBcast(unsigned int* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclUint, root, comm, stream));
}

void CallBcast(signed long* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclLong, root, comm, stream));
}

void CallBcast(unsigned long* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclUlong, root, comm, stream));
}

void CallBcast(__fp16* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclHalf, root, comm, stream));
}

void CallBcast(float* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclFloat, root, comm, stream));
}

void CallBcast(double* psrc_buff, size_t buff_len, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclBcast(psrc_buff, buff_len, rcclDouble, root, comm, stream));
}

template<typename T>
void DoBcast(std::vector<int>& device_list, std::vector<hipStream_t>& device_streams,
    std::vector<rcclComm_t>& rccl_comms, std::vector<void*>& src_device_buffers,
    std::vector<void*>& src_host_buffers, std::vector<void*>& dst_host_buffers,
    size_t buff_size, int root) {

    size_t buff_len = buff_size / sizeof(T);
    size_t num_gpus = device_list.size();

    for(int i = 0; i < device_list.size(); i++) {
        if(root == device_list[i]) {
            for(size_t j = 0; j < buff_len; j++) {
                reinterpret_cast<T*>(src_host_buffers[i])[j] = static_cast<T>(kbuffer_values[device_list[i]]);
            }
        } else {
            for(size_t j = 0; j < buff_len; j++) {
                reinterpret_cast<T*>(src_host_buffers[i])[j] = static_cast<T>(0);
            }
        }
        HIPCHECK(hipSetDevice(device_list[i]));
        HIPCHECK(hipMemcpy(src_device_buffers[i], src_host_buffers[i], buff_size, hipMemcpyHostToDevice));
    }
    for(size_t i = 0; i < num_gpus; i++) {
        HIPCHECK(hipSetDevice(device_list[i]));
        CallBcast(reinterpret_cast<T*>(src_device_buffers[i]), buff_len, root, rccl_comms[i], device_streams[i]);
    }
    for(size_t i = 0; i < num_gpus; i++) {
        HIPCHECK(hipSetDevice(device_list[i]));
        HIPCHECK(hipStreamSynchronize(device_streams[i]));
    }
    for(size_t i = 0; i < num_gpus; i++) {
        HIPCHECK(hipSetDevice(root));
        HIPCHECK(hipMemcpy(dst_host_buffers[i], src_device_buffers[i], buff_size, hipMemcpyDeviceToHost));
    }
    for(size_t i = 0; i < num_gpus; i++) {
        validate(reinterpret_cast<T*>(dst_host_buffers[i]), static_cast<T>(kbuffer_values[root]), buff_len, 1, 0);
    }
}

void RandomReduceTest(std::vector<int>& device_list, int num_tests, int root) {
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
        }
    }

     for(auto pbuff_len = buffer_lengths.begin(); pbuff_len != buffer_lengths.end(); pbuff_len++) {
        DoBcast<signed char>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);
        DoBcast<unsigned char>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);
        DoBcast<signed short>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);
        DoBcast<unsigned short>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);
        DoBcast<signed int>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);
        DoBcast<unsigned int>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);
        DoBcast<signed long>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);
        DoBcast<unsigned long>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);
        DoBcast<float>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);
        DoBcast<double>(device_list, device_streams, rccl_comms, src_device_buffers, src_host_buffers, dst_host_buffers, *pbuff_len, root);

//        DoBcast<__fp16>(device_list, device_streams, rccl_comms, host_buffers, device_buffers, dst_host_buffer, dst_device_buffer, *pbuff_len, root);

    }

// free allocted buffers on both host and device

    for(auto iter = src_device_buffers.begin(); iter != src_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for(auto iter = src_host_buffers.begin(); iter != src_host_buffers.end(); iter++) {
        delete reinterpret_cast<signed char*>(*iter);
    }
    for(auto iter = dst_host_buffers.begin(); iter != dst_host_buffers.end(); iter++) {
        delete reinterpret_cast<signed char*>(*iter);
    }

}

int main(int argc, char* argv[]) {
    if(argc > 1) {
    if(strcmp(argv[1], "-r") == 0) {
        if(argc > 5) {
            int num_tests = atoi(argv[2]);
            int num_gpus = atoi(argv[4]);
            int root_gpu = atoi(argv[3]);
            std::vector<int> device_list(num_gpus);
            if(argc == num_gpus + 5) {
                for(int i = 0; i < num_gpus; i++) {
                    device_list[i] = atoi(argv[i+5]);
                }
                RandomReduceTest(device_list, num_tests, root_gpu);
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
