// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include "nccl.h"
#include <vector>
#include <iostream>
#include "rocprofiler-sdk-roctx/version.h"
#include <rocprofiler-register/rocprofiler-register.h>


#define ROCP_REG_VERSION                                                                           \
    ROCPROFILER_REGISTER_COMPUTE_VERSION_3(                                                        \
        ROCTX_VERSION_MAJOR, ROCTX_VERSION_MINOR, ROCTX_VERSION_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(rccl, ROCP_REG_VERSION)

ncclResult_t ncclAllGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAllReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclAllToAll_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream);

ncclResult_t ncclAllToAllv_impl(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);

ncclResult_t ncclBroadcast_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);

ncclResult_t ncclReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclReduceScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);

ncclResult_t ncclSend_impl(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclRecv_impl(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

namespace rccl {
    typedef uint64_t rccl_range_id_t;
    typedef ncclResult_t (*ncclAllGatherfn_t)(const void* sendbuff, void* recvbuff, size_t sendcount,
        ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
    typedef ncclResult_t (*ncclAllReduce_fn_t)(const void* sendbuff, void* recvbuff, size_t count,
        ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
    typedef ncclResult_t (*ncclAllToAll_fn_t)(const void* sendbuff, void* recvbuff, size_t count,
        ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
    typedef ncclResult_t (*ncclAllToAllv_fn_t)(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
        void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
        ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
    typedef ncclResult_t (*ncclBroadcast_fn_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
        ncclComm_t comm, cudaStream_t stream);
    typedef ncclResult_t (*ncclGather_fn_t)(const void* sendbuff, void* recvbuff, size_t sendcount,
        ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);
    typedef ncclResult_t (*ncclReduce_fn_t)(const void* sendbuff, void* recvbuff, size_t count,
        ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
    typedef ncclResult_t (*ncclReduceScatter_fn_t)(const void* sendbuff, void* recvbuff, size_t recvcount,
        ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
    typedef ncclResult_t (*ncclScatter_fn_t)(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
        ncclComm_t comm, hipStream_t stream);
    typedef ncclResult_t (*ncclSend_fn_t)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
        ncclComm_t comm, cudaStream_t stream);
    typedef ncclResult_t (*ncclRecv_fn_t)(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
        ncclComm_t comm, cudaStream_t stream);


    struct rcclApiFuncTable
    {
        uint64_t               size;
        ncclAllGatherfn_t      ncclAllGather_fn;
        ncclAllReduce_fn_t     ncclAllReduce_fn;
        ncclAllToAll_fn_t      ncclAllToAll_fn;
        ncclAllToAllv_fn_t     ncclAllToAllv_fn;
        ncclBroadcast_fn_t     ncclBroadcast_fn;
        ncclGather_fn_t        ncclGather_fn;
        ncclReduce_fn_t        ncclReduce_fn;
        ncclReduceScatter_fn_t ncclReduceScatter_fn;
        ncclScatter_fn_t       ncclScatter_fn;
        ncclSend_fn_t          ncclSend_fn;
        ncclRecv_fn_t          ncclRecv_fn;
    };

    constexpr size_t
    compute_table_offset(size_t n)
    {
        return (sizeof(uint64_t) + (n * sizeof(void*)));
    }

    constexpr size_t
    compute_table_size(size_t nmembers)
    {
        return (sizeof(uint64_t) + (nmembers * sizeof(void*)));
    }

    #define RCCL_ASSERT_OFFSET(TABLE, MEMBER, IDX)                                                    \
        static_assert(offsetof(TABLE, MEMBER) == compute_table_offset(IDX),                            \
                    "Do not re-arrange the table members")

    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllGather_fn,        0);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllReduce_fn,        1);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllToAll_fn,         2);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllToAllv_fn,        3);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclBroadcast_fn,        4);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclGather_fn,           5);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclReduce_fn,           6);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclReduceScatter_fn,    7);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclScatter_fn,          8);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclSend_fn,             9);
    RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclRecv_fn,             10);

    #undef RCCL_ASSERT_OFFSET

    static_assert(
        sizeof(rcclApiFuncTable) == compute_table_size(11),
        "Update table major/step version and add a new offset assertion if this fails to compile");

    static std::array<unsigned char, sizeof(rcclApiFuncTable)> m_buffer;


    rcclApiFuncTable* RcclGetFunctionTable_impl()
    {
        static rcclApiFuncTable tbl = {sizeof(rcclApiFuncTable),
                                        &ncclAllGather_impl,
                                        &ncclAllReduce_impl,
                                        &ncclAllToAll_impl,
                                        &ncclAllToAllv_impl,
                                        &ncclBroadcast_impl,
                                        &ncclGather_impl,
                                        &ncclReduce_impl,
                                        &ncclReduceScatter_impl,
                                        &ncclScatter_impl,
                                        &ncclSend_impl,
                                        &ncclRecv_impl};

        std::array<void*, 1> table_array{&tbl};
        rocprofiler_register_library_indentifier_t lib_id      = rocprofiler_register_library_indentifier_t{};
        rocprofiler_register_error_code_t rocp_reg_status =
            rocprofiler_register_library_api_table("rccl",
                                                &ROCPROFILER_REGISTER_IMPORT_FUNC(rccl),
                                                ROCP_REG_VERSION,
                                                table_array.data(),
                                                table_array.size(),
                                                &lib_id);

        INFO( NCCL_COLL,"[rocprofiler-sdk-rccl][ = %d ] rocprofiler-register returned code = %d : %s", getpid(),
               rocp_reg_status, rocprofiler_register_error_string(rocp_reg_status));

        if(rocp_reg_status != ROCP_REG_SUCCESS && rocp_reg_status != ROCP_REG_NO_TOOLS)
            WARN( "[rocprofiler-sdk-rccl][%d] rocprofiler-register failed with error code %d : %s",  getpid(),
            rocp_reg_status, rocprofiler_register_error_string(rocp_reg_status));

        return &tbl;
    }

    const rcclApiFuncTable* RcclGetFunctionTable()
    {
        static rcclApiFuncTable* tbl = RcclGetFunctionTable_impl();
        return tbl;
    }
}

