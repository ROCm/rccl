/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl/rccl.h"
//#include "rcclCheck.h"
#include <algorithm>
#include <iostream>
#include <list>
#include <typeinfo>
#include <vector>
#include "common.h"
#include "validation/validate.h"

void CallAllGather(signed char* psrc_buff, signed char* pdst_buff,
                   size_t buff_len, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(
        rcclAllGather(psrc_buff, buff_len, rcclChar, pdst_buff, comm, stream));
}

void CallAllGather(unsigned char* psrc_buff, unsigned char* pdst_buff,
                   size_t buff_len, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(
        rcclAllGather(psrc_buff, buff_len, rcclUchar, pdst_buff, comm, stream));
}

void CallAllGather(signed short* psrc_buff, signed short* pdst_buff,
                   size_t buff_len, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(
        rcclAllGather(psrc_buff, buff_len, rcclShort, pdst_buff, comm, stream));
}

void CallAllGather(unsigned short* psrc_buff, unsigned short* pdst_buff,
                   size_t buff_len, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllGather(psrc_buff, buff_len, rcclUshort, pdst_buff, comm,
                            stream));
}

void CallAllGather(signed int* psrc_buff, signed int* pdst_buff,
                   size_t buff_len, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(
        rcclAllGather(psrc_buff, buff_len, rcclInt, pdst_buff, comm, stream));
}

void CallAllGather(unsigned int* psrc_buff, unsigned int* pdst_buff,
                   size_t buff_len, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(
        rcclAllGather(psrc_buff, buff_len, rcclUint, pdst_buff, comm, stream));
}

void CallAllGather(signed long* psrc_buff, signed long* pdst_buff,
                   size_t buff_len, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(
        rcclAllGather(psrc_buff, buff_len, rcclLong, pdst_buff, comm, stream));
}

void CallAllGather(unsigned long* psrc_buff, unsigned long* pdst_buff,
                   size_t buff_len, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(
        rcclAllGather(psrc_buff, buff_len, rcclUlong, pdst_buff, comm, stream));
}

void CallAllGather(__fp16* psrc_buff, __fp16* pdst_buff, size_t buff_len,
                   rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(
        rcclAllGather(psrc_buff, buff_len, rcclHalf, pdst_buff, comm, stream));
}

void CallAllGather(float* psrc_buff, float* pdst_buff, size_t buff_len,
                   rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(
        rcclAllGather(psrc_buff, buff_len, rcclFloat, pdst_buff, comm, stream));
}

void CallAllGather(double* psrc_buff, double* pdst_buff, size_t buff_len,
                   rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllGather(psrc_buff, buff_len, rcclDouble, pdst_buff, comm,
                            stream));
}

template <typename T>
void DoAllGather(std::vector<int>& device_list,
                 std::vector<hipStream_t>& device_streams,
                 std::vector<rcclComm_t>& rccl_comms,
                 std::vector<void*>& src_host_buffers,
                 std::vector<void*>& src_device_buffers,
                 std::vector<void*>& dst_host_buffers,
                 std::vector<void*>& dst_device_buffers, size_t buff_size) {
    size_t buff_len = buff_size / sizeof(T);
    size_t num_gpus = device_list.size();

    for (int i = 0; i < device_list.size(); i++) {
        for (size_t j = 0; j < buff_len; j++) {
            reinterpret_cast<T*>(src_host_buffers[i])[j] =
                static_cast<T>(kbuffer_values[device_list[i]]);
        }
        for (size_t j = 0; j < buff_len * num_gpus; j++) {
            reinterpret_cast<T*>(dst_host_buffers[i])[j] = static_cast<T>(0);
        }
        HIPCHECK(hipSetDevice(device_list[i]));
        HIPCHECK(hipMemcpyAsync(src_device_buffers[i], src_host_buffers[i],
                                buff_size, hipMemcpyHostToDevice,
                                device_streams[i]));
        HIPCHECK(hipMemcpyAsync(dst_device_buffers[i], dst_host_buffers[i],
                                buff_size * num_gpus, hipMemcpyHostToDevice,
                                device_streams[i]));
    }
    {
        for (size_t i = 0; i < num_gpus; i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            CallAllGather(reinterpret_cast<T*>(src_device_buffers[i]),
                          reinterpret_cast<T*>(dst_device_buffers[i]), buff_len,
                          rccl_comms[i], device_streams[i]);
        }

        for (size_t i = 0; i < num_gpus; i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipMemcpyAsync(dst_host_buffers[i], dst_device_buffers[i],
                                    buff_size * num_gpus, hipMemcpyDeviceToHost,
                                    device_streams[i]));
            HIPCHECK(hipStreamSynchronize(device_streams[i]));
        }

        {
            for (size_t i = 0; i < num_gpus; i++) {
                for (size_t j = 0; j < num_gpus; j++) {
                    validate(reinterpret_cast<T*>(src_host_buffers[i]),
                             &reinterpret_cast<T*>(
                                 dst_host_buffers[j])[buff_len * i],
                             buff_len, 1, 0);
                }
            }
        }
    }
}

void RandomGatherTest(std::vector<int>& device_list, int num_tests) {
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

    std::vector<void*> src_device_buffers(num_gpus);
    std::vector<void*> src_host_buffers(num_gpus);
    std::vector<void*> dst_device_buffers(num_gpus);
    std::vector<void*> dst_host_buffers(num_gpus);

    std::vector<hipStream_t> device_streams(num_gpus);

    for (int i = 0; i < device_list.size(); i++) {
        hipHostMalloc(
            &(src_host_buffers[i]),
            max_allocated_memory);  // = reinterpret_cast<void*>(new signed
                                    // char[max_allocated_memory]);
        hipHostMalloc(&(dst_host_buffers[i]),
                      max_allocated_memory *
                          num_gpus);  // = reinterpret_cast<void*>(new signed
                                      // char[max_allocated_memory*num_gpus]);
    }

    {  // used new scope to force current-device guard to destruct after
       // changing active device
        CurrDeviceGuard_t g;
        for (int i = 0; i < device_list.size(); i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamCreate(&device_streams[i]));
            HIPCHECK(hipMalloc(&(src_device_buffers[i]), max_allocated_memory));
            HIPCHECK(hipMalloc(&(dst_device_buffers[i]),
                               max_allocated_memory * num_gpus));
        }
    }

    for (auto pbuff_len = buffer_lengths.begin();
         pbuff_len != buffer_lengths.end(); pbuff_len++) {
        DoAllGather<signed char>(device_list, device_streams, rccl_comms,
                                 src_host_buffers, src_device_buffers,
                                 dst_host_buffers, dst_device_buffers,
                                 *pbuff_len);
        DoAllGather<unsigned char>(device_list, device_streams, rccl_comms,
                                   src_host_buffers, src_device_buffers,
                                   dst_host_buffers, dst_device_buffers,
                                   *pbuff_len);
        DoAllGather<signed short>(device_list, device_streams, rccl_comms,
                                  src_host_buffers, src_device_buffers,
                                  dst_host_buffers, dst_device_buffers,
                                  *pbuff_len);
        DoAllGather<unsigned short>(device_list, device_streams, rccl_comms,
                                    src_host_buffers, src_device_buffers,
                                    dst_host_buffers, dst_device_buffers,
                                    *pbuff_len);
        DoAllGather<signed int>(device_list, device_streams, rccl_comms,
                                src_host_buffers, src_device_buffers,
                                dst_host_buffers, dst_device_buffers,
                                *pbuff_len);
        DoAllGather<unsigned int>(device_list, device_streams, rccl_comms,
                                  src_host_buffers, src_device_buffers,
                                  dst_host_buffers, dst_device_buffers,
                                  *pbuff_len);
        DoAllGather<signed long>(device_list, device_streams, rccl_comms,
                                 src_host_buffers, src_device_buffers,
                                 dst_host_buffers, dst_device_buffers,
                                 *pbuff_len);
        DoAllGather<unsigned long>(device_list, device_streams, rccl_comms,
                                   src_host_buffers, src_device_buffers,
                                   dst_host_buffers, dst_device_buffers,
                                   *pbuff_len);
        DoAllGather<float>(device_list, device_streams, rccl_comms,
                           src_host_buffers, src_device_buffers,
                           dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllGather<double>(device_list, device_streams, rccl_comms,
                            src_host_buffers, src_device_buffers,
                            dst_host_buffers, dst_device_buffers, *pbuff_len);

        //        DoAllGather<__fp16>(device_list, device_streams, rccl_comms,
        //        host_buffers, device_buffers, dst_host_buffer,
        //        dst_device_buffer, *pbuff_len);
    }

    // free allocted buffers on both host and device

    for (auto iter = src_device_buffers.begin();
         iter != src_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for (auto iter = src_host_buffers.begin(); iter != src_host_buffers.end();
         iter++) {
        //        delete reinterpret_cast<signed char*>(*iter);
        HIPCHECK(hipHostFree(*iter));
    }

    for (auto iter = dst_device_buffers.begin();
         iter != dst_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for (auto iter = dst_host_buffers.begin(); iter != dst_host_buffers.end();
         iter++) {
        //        delete reinterpret_cast<signed char*>(*iter);
        HIPCHECK(hipHostFree(*iter));
    }
}

void GatherTestSize(std::vector<int>& device_list, size_t size_in_bytes) {
    size_t num_gpus = device_list.size();
    EnableDevicePeerAccess(device_list);

    std::vector<rcclComm_t> rccl_comms(num_gpus);
    RCCLCHECK(rcclCommInitAll(rccl_comms.data(), num_gpus, device_list.data()));

    std::vector<void*> src_device_buffers(num_gpus);
    std::vector<void*> src_host_buffers(num_gpus);
    std::vector<void*> dst_device_buffers(num_gpus);
    std::vector<void*> dst_host_buffers(num_gpus);

    std::vector<hipStream_t> device_streams(num_gpus);

    for (int i = 0; i < device_list.size(); i++) {
        src_host_buffers[i] =
            reinterpret_cast<void*>(new signed char[size_in_bytes]);
        dst_host_buffers[i] =
            reinterpret_cast<void*>(new signed char[size_in_bytes * num_gpus]);
    }

    {  // used new scope to force current-device guard to destruct after
       // changing active device
        CurrDeviceGuard_t g;
        for (int i = 0; i < device_list.size(); i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamCreate(&device_streams[i]));
            HIPCHECK(hipMalloc(&(src_device_buffers[i]), size_in_bytes));
            HIPCHECK(
                hipMalloc(&(dst_device_buffers[i]), size_in_bytes * num_gpus));
        }
    }

    DoAllGather<signed char>(device_list, device_streams, rccl_comms,
                             src_host_buffers, src_device_buffers,
                             dst_host_buffers, dst_device_buffers,
                             size_in_bytes);
    DoAllGather<unsigned char>(device_list, device_streams, rccl_comms,
                               src_host_buffers, src_device_buffers,
                               dst_host_buffers, dst_device_buffers,
                               size_in_bytes);
    DoAllGather<signed short>(device_list, device_streams, rccl_comms,
                              src_host_buffers, src_device_buffers,
                              dst_host_buffers, dst_device_buffers,
                              size_in_bytes);
    DoAllGather<unsigned short>(device_list, device_streams, rccl_comms,
                                src_host_buffers, src_device_buffers,
                                dst_host_buffers, dst_device_buffers,
                                size_in_bytes);
    DoAllGather<signed int>(device_list, device_streams, rccl_comms,
                            src_host_buffers, src_device_buffers,
                            dst_host_buffers, dst_device_buffers,
                            size_in_bytes);
    DoAllGather<unsigned int>(device_list, device_streams, rccl_comms,
                              src_host_buffers, src_device_buffers,
                              dst_host_buffers, dst_device_buffers,
                              size_in_bytes);
    DoAllGather<signed long>(device_list, device_streams, rccl_comms,
                             src_host_buffers, src_device_buffers,
                             dst_host_buffers, dst_device_buffers,
                             size_in_bytes);
    DoAllGather<unsigned long>(device_list, device_streams, rccl_comms,
                               src_host_buffers, src_device_buffers,
                               dst_host_buffers, dst_device_buffers,
                               size_in_bytes);
    DoAllGather<float>(device_list, device_streams, rccl_comms,
                       src_host_buffers, src_device_buffers, dst_host_buffers,
                       dst_device_buffers, size_in_bytes);
    DoAllGather<double>(device_list, device_streams, rccl_comms,
                        src_host_buffers, src_device_buffers, dst_host_buffers,
                        dst_device_buffers, size_in_bytes);

    //        DoAllGather<__fp16>(device_list, device_streams, rccl_comms,
    //        host_buffers, device_buffers, dst_host_buffer, dst_device_buffer,
    //        *pbuff_len);

    // free allocted buffers on both host and device

    for (auto iter = src_device_buffers.begin();
         iter != src_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for (auto iter = src_host_buffers.begin(); iter != src_host_buffers.end();
         iter++) {
        delete reinterpret_cast<signed char*>(*iter);
    }

    for (auto iter = dst_device_buffers.begin();
         iter != dst_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for (auto iter = dst_host_buffers.begin(); iter != dst_host_buffers.end();
         iter++) {
        delete reinterpret_cast<signed char*>(*iter);
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        if (strcmp(argv[1], "-r") == 0) {
            if (argc > 4) {
                int num_tests = atoi(argv[2]);
                int num_gpus = atoi(argv[3]);
                std::vector<int> device_list(num_gpus);
                if (argc == num_gpus + 4) {
                    for (int i = 0; i < num_gpus; i++) {
                        device_list[i] = atoi(argv[i + 4]);
                    }
                    RandomGatherTest(device_list, num_tests);
                } else {
                    print_out(
                        "The size of gpus in list is less than specified "
                        "length");
                }
            }
            return 0;
        }
    }
    if (argc == 3) {
        int num_gpus = atoi(argv[1]);
        size_t size_in_bytes = atoi(argv[2]);
        std::vector<int> device_list(num_gpus);
        for (int i = 0; i < num_gpus; i++) {
            device_list[i] = i;
        }
        std::cout << num_gpus << " " << size_in_bytes << std::endl;
        GatherTestSize(device_list, size_in_bytes);
        return 0;
    }
    if (argc != 3 || argc != 5) {
        std::cout << "Usage: ./a.out -r <num tests> <num gpus> <list of gpus>"
                  << std::endl;
        std::cout << "./a.out -r 99 3 1 2 3" << std::endl;
        std::cout << "./a.out <num gpus> <number of elements>" << std::endl;
        std::cout << "Example: ./a.out -s 4 1024" << std::endl;
        std::cout << "[-r] enabled validation for certain number of gpus, "
                     "elements, ops and datatypes"
                  << std::endl;
        return 0;
    }
}
