/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl/rccl.h"
#include <algorithm>
#include <iostream>
#include <list>
#include <typeinfo>
#include <vector>
#include "common.h"
#include "validation/validate.h"

void CallAllReduce(signed char* psrc_buff, signed char* pdst_buff,
                   size_t buff_len, rcclRedOp_t op, rcclComm_t comm,
                   hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclChar, op, comm,
                            stream));
}

void CallAllReduce(signed int* psrc_buff, signed int* pdst_buff,
                   size_t buff_len, rcclRedOp_t op, rcclComm_t comm,
                   hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclInt, op, comm,
                            stream));
}

void CallAllReduce(signed long* psrc_buff, signed long* pdst_buff,
                   size_t buff_len, rcclRedOp_t op, rcclComm_t comm,
                   hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclInt64, op, comm,
                            stream));
}

void CallAllReduce(unsigned long* psrc_buff, unsigned long* pdst_buff,
                   size_t buff_len, rcclRedOp_t op, rcclComm_t comm,
                   hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclUint64, op, comm,
                            stream));
}

void CallAllReduce(__fp16* psrc_buff, __fp16* pdst_buff, size_t buff_len,
                   rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclHalf, op, comm,
                            stream));
}

void CallAllReduce(float* psrc_buff, float* pdst_buff, size_t buff_len,
                   rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclFloat, op, comm,
                            stream));
}

void CallAllReduce(double* psrc_buff, double* pdst_buff, size_t buff_len,
                   rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(psrc_buff, pdst_buff, buff_len, rcclDouble, op,
                            comm, stream));
}

template <typename T>
void DoAllReduce(std::vector<int>& device_list,
                 std::vector<hipStream_t>& htod_streams,
                 std::vector<hipStream_t>& op_streams,
                 std::vector<hipStream_t>& dtoh_streams,
                 std::vector<rcclComm_t>& rccl_comms,
                 std::vector<void*>& src_host_buffers,
                 std::vector<void*>& src_device_buffers,
                 std::vector<void*>& dst_host_buffers,
                 std::vector<void*>& dst_device_buffers, size_t buff_size) {
    size_t buff_len = buff_size / sizeof(T);
    size_t num_gpus = device_list.size();

    std::vector<hipEvent_t> events(num_gpus);

    for (int i = 0; i < device_list.size(); i++) {
        for (size_t j = 0; j < buff_len; j++) {
            reinterpret_cast<T*>(src_host_buffers[i])[j] =
                static_cast<T>(kbuffer_values[device_list[i]]);
            reinterpret_cast<T*>(dst_host_buffers[i])[j] =
                static_cast<T>(kbuffer_values[device_list[i]]);
        }

        HIPCHECK(hipEventCreate(&events[i]));
        HIPCHECK(hipSetDevice(device_list[i]));
        HIPCHECK(hipMemcpyAsync(src_device_buffers[i], src_host_buffers[i],
                                buff_size, hipMemcpyHostToDevice,
                                htod_streams[i]));
        HIPCHECK(hipEventRecord(events[i], htod_streams[i]));
    }
    for (auto p_ops = umap_rccl_op.begin(); p_ops != umap_rccl_op.end();
         p_ops++) {
        for (size_t i = 0; i < num_gpus; i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamWaitEvent(op_streams[i], events[i], 0));
            CallAllReduce(reinterpret_cast<T*>(src_device_buffers[i]),
                          reinterpret_cast<T*>(dst_device_buffers[i]), buff_len,
                          p_ops->second, rccl_comms[i], op_streams[i]);
            HIPCHECK(hipEventRecord(events[i], op_streams[i]));
        }

        for (size_t i = 0; i < num_gpus; i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamWaitEvent(dtoh_streams[i], events[i], 0));
            HIPCHECK(hipMemcpyAsync(dst_host_buffers[i], dst_device_buffers[i],
                                    buff_size, hipMemcpyDeviceToHost,
                                    dtoh_streams[i]));
            HIPCHECK(hipEventRecord(events[i], dtoh_streams[i]));
        }
        for (size_t i = 0; i < num_gpus; i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipEventSynchronize(events[i]));
        }

        if (p_ops->second == rcclSum) {
            T sum_val = static_cast<T>(0);
            for (auto pdevice_index = device_list.begin();
                 pdevice_index != device_list.end(); pdevice_index++) {
                sum_val += static_cast<T>(kbuffer_values[*pdevice_index]);
            }
            for (size_t i = 0; i < num_gpus; i++) {
                validate(reinterpret_cast<T*>(dst_host_buffers[i]), sum_val,
                         buff_len, 1, 0);
            }
        }
        if (p_ops->second == rcclProd) {
            T prod_val = static_cast<T>(1);
            for (auto pdevice_index = device_list.begin();
                 pdevice_index != device_list.end(); pdevice_index++) {
                prod_val *= static_cast<T>(kbuffer_values[*pdevice_index]);
            }
            for (size_t i = 0; i < num_gpus; i++) {
                validate(reinterpret_cast<T*>(dst_host_buffers[i]), prod_val,
                         buff_len, 1, 0);
            }
        }
        if (p_ops->second == rcclMax) {
            T max_val = static_cast<T>(0);
            for (auto pdevice_index = device_list.begin();
                 pdevice_index != device_list.end(); pdevice_index++) {
                T tmp_val = static_cast<T>(kbuffer_values[*pdevice_index]);
                max_val = max_val > tmp_val ? max_val : tmp_val;
            }
            for (size_t i = 0; i < num_gpus; i++) {
                validate(reinterpret_cast<T*>(dst_host_buffers[i]), max_val,
                         buff_len, 1, 0);
            }
        }
        if (p_ops->second == rcclMin) {
            T min_val = static_cast<T>(100);
            for (auto pdevice_index = device_list.begin();
                 pdevice_index != device_list.end(); pdevice_index++) {
                T tmp_val = static_cast<T>(kbuffer_values[*pdevice_index]);
                min_val = min_val < tmp_val ? min_val : tmp_val;
            }
            for (size_t i = 0; i < num_gpus; i++) {
                validate(reinterpret_cast<T*>(dst_host_buffers[i]), min_val,
                         buff_len, 1, 0);
            }
        }
    }
}

void RandomReduceTest(std::vector<int>& device_list, int num_tests) {
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

    std::vector<hipStream_t> htod_streams(num_gpus);
    std::vector<hipStream_t> op_streams(num_gpus);
    std::vector<hipStream_t> dtoh_streams(num_gpus);

    for (int i = 0; i < device_list.size(); i++) {
        HIPCHECK(hipHostMalloc(&src_host_buffers[i], max_allocated_memory));
        HIPCHECK(hipHostMalloc(&dst_host_buffers[i], max_allocated_memory));
    }

    {  // used new scope to force current-device guard to destruct after
       // changing active device
        CurrDeviceGuard_t g;
        for (int i = 0; i < device_list.size(); i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamCreate(&htod_streams[i]));
            HIPCHECK(hipStreamCreate(&op_streams[i]));
            HIPCHECK(hipStreamCreate(&dtoh_streams[i]));
            HIPCHECK(hipMalloc(&(src_device_buffers[i]), max_allocated_memory));
            HIPCHECK(hipMalloc(&(dst_device_buffers[i]), max_allocated_memory));
        }
    }

    for (auto pbuff_len = buffer_lengths.begin();
         pbuff_len != buffer_lengths.end(); pbuff_len++) {
        DoAllReduce<signed char>(device_list, htod_streams, op_streams,
                                 dtoh_streams, rccl_comms, src_host_buffers,
                                 src_device_buffers, dst_host_buffers,
                                 dst_device_buffers, *pbuff_len);
        DoAllReduce<signed int>(device_list, htod_streams, op_streams,
                                dtoh_streams, rccl_comms, src_host_buffers,
                                src_device_buffers, dst_host_buffers,
                                dst_device_buffers, *pbuff_len);
        DoAllReduce<signed long>(device_list, htod_streams, op_streams,
                                 dtoh_streams, rccl_comms, src_host_buffers,
                                 src_device_buffers, dst_host_buffers,
                                 dst_device_buffers, *pbuff_len);
        DoAllReduce<unsigned long>(device_list, htod_streams, op_streams,
                                   dtoh_streams, rccl_comms, src_host_buffers,
                                   src_device_buffers, dst_host_buffers,
                                   dst_device_buffers, *pbuff_len);
        DoAllReduce<float>(device_list, htod_streams, op_streams, dtoh_streams,
                           rccl_comms, src_host_buffers, src_device_buffers,
                           dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<double>(device_list, htod_streams, op_streams, dtoh_streams,
                            rccl_comms, src_host_buffers, src_device_buffers,
                            dst_host_buffers, dst_device_buffers, *pbuff_len);
        DoAllReduce<__fp16>(device_list, htod_streams, op_streams, dtoh_streams,
                            rccl_comms, src_host_buffers, src_device_buffers,
                            dst_host_buffers, dst_device_buffers, *pbuff_len);
    }

    // free allocted buffers on both host and device

    for (auto iter = src_device_buffers.begin();
         iter != src_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for (auto iter = src_host_buffers.begin(); iter != src_host_buffers.end();
         iter++) {
        HIPCHECK(hipHostFree(*iter));
    }

    for (auto iter = dst_device_buffers.begin();
         iter != dst_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for (auto iter = dst_host_buffers.begin(); iter != dst_host_buffers.end();
         iter++) {
        HIPCHECK(hipHostFree(*iter));
    }
}

void ReduceTestSize(std::vector<int>& device_list, size_t size_in_bytes) {
    size_t num_gpus = device_list.size();
    EnableDevicePeerAccess(device_list);

    std::vector<rcclComm_t> rccl_comms(num_gpus);
    RCCLCHECK(rcclCommInitAll(rccl_comms.data(), num_gpus, device_list.data()));

    std::vector<void*> src_device_buffers(num_gpus);
    std::vector<void*> src_host_buffers(num_gpus);
    std::vector<void*> dst_device_buffers(num_gpus);
    std::vector<void*> dst_host_buffers(num_gpus);

    std::vector<hipStream_t> htod_streams(num_gpus);
    std::vector<hipStream_t> op_streams(num_gpus);
    std::vector<hipStream_t> dtoh_streams(num_gpus);

    for (int i = 0; i < device_list.size(); i++) {
        HIPCHECK(hipHostMalloc(&src_host_buffers[i], size_in_bytes));
        HIPCHECK(hipHostMalloc(&dst_host_buffers[i], size_in_bytes));
    }

    {  // used new scope to force current-device guard to destruct after
       // changing active device
        CurrDeviceGuard_t g;
        for (int i = 0; i < device_list.size(); i++) {
            HIPCHECK(hipSetDevice(device_list[i]));
            HIPCHECK(hipStreamCreate(&htod_streams[i]));
            HIPCHECK(hipStreamCreate(&op_streams[i]));
            HIPCHECK(hipStreamCreate(&dtoh_streams[i]));
            HIPCHECK(hipMalloc(&(src_device_buffers[i]), size_in_bytes));
            HIPCHECK(hipMalloc(&(dst_device_buffers[i]), size_in_bytes));
        }
    }

    DoAllReduce<signed char>(device_list, htod_streams, op_streams,
                             dtoh_streams, rccl_comms, src_host_buffers,
                             src_device_buffers, dst_host_buffers,
                             dst_device_buffers, size_in_bytes);
    DoAllReduce<signed int>(device_list, htod_streams, op_streams, dtoh_streams,
                            rccl_comms, src_host_buffers, src_device_buffers,
                            dst_host_buffers, dst_device_buffers,
                            size_in_bytes);
    DoAllReduce<signed long>(device_list, htod_streams, op_streams,
                             dtoh_streams, rccl_comms, src_host_buffers,
                             src_device_buffers, dst_host_buffers,
                             dst_device_buffers, size_in_bytes);
    DoAllReduce<unsigned long>(device_list, htod_streams, op_streams,
                               dtoh_streams, rccl_comms, src_host_buffers,
                               src_device_buffers, dst_host_buffers,
                               dst_device_buffers, size_in_bytes);
    DoAllReduce<float>(device_list, htod_streams, op_streams, dtoh_streams,
                       rccl_comms, src_host_buffers, src_device_buffers,
                       dst_host_buffers, dst_device_buffers, size_in_bytes);
    DoAllReduce<double>(device_list, htod_streams, op_streams, dtoh_streams,
                        rccl_comms, src_host_buffers, src_device_buffers,
                        dst_host_buffers, dst_device_buffers, size_in_bytes);
    DoAllReduce<__fp16>(device_list, htod_streams, op_streams, dtoh_streams,
                        rccl_comms, src_host_buffers, src_device_buffers,
                        dst_host_buffers, dst_device_buffers, size_in_bytes);

    // free allocted buffers on both host and device

    for (auto iter = src_device_buffers.begin();
         iter != src_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for (auto iter = src_host_buffers.begin(); iter != src_host_buffers.end();
         iter++) {
        HIPCHECK(hipHostFree(*iter));
    }

    for (auto iter = dst_device_buffers.begin();
         iter != dst_device_buffers.end(); iter++) {
        HIPCHECK(hipFree(*iter));
    }
    for (auto iter = dst_host_buffers.begin(); iter != dst_host_buffers.end();
         iter++) {
        HIPCHECK(hipHostFree(*iter));
    }

    for (int i = 0; i < device_list.size(); i++) {
        HIPCHECK(hipStreamDestroy(htod_streams[i]));
        HIPCHECK(hipStreamDestroy(op_streams[i]));
        HIPCHECK(hipStreamDestroy(dtoh_streams[i]));
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
                    RandomReduceTest(device_list, num_tests);
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
        ReduceTestSize(device_list, size_in_bytes);
        return 0;
    }
    if (argc != 3 || argc != 5) {
        std::cout << "Usage: ./a.out -r <num tests> <num gpus> <list of gpus>"
                  << std::endl;
        std::cout << "./a.out -r 99 3 1 2 3" << std::endl;
        std::cout << "Usage: ./a.out <num gpus> <size of buffers in bytes>" << std::endl;
        std::cout << "./a.out 4 1024" << std::endl;
        return 0;
    }
}
