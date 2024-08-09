
#include "nccl.h"
#include <vector>
#include <iostream>
#include "api_trace.h"
#include "core.h"
#if defined(RCCL_ROCPROFILER_REGISTER) && RCCL_ROCPROFILER_REGISTER > 0
    #include <rocprofiler-register/rocprofiler-register.h>

#define ROCP_REG_VERSION                                                                           \
    ROCPROFILER_REGISTER_COMPUTE_VERSION_3(                                                        \
        RCCL_VERSION_MAJOR, RCCL_VERSION_MINOR, RCCL_VERSION_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(rccl, ROCP_REG_VERSION)
#endif

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclAllToAllv, const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, hipStream_t stream);


namespace rccl {

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
        #if defined(RCCL_ROCPROFILER_REGISTER) && RCCL_ROCPROFILER_REGISTER > 0
            std::array<void*, 1> table_array{&tbl};
            rocprofiler_register_library_indentifier_t lib_id      = rocprofiler_register_library_indentifier_t{};
            rocprofiler_register_error_code_t rocp_reg_status =
                rocprofiler_register_library_api_table("rccl",
                                                    &ROCPROFILER_REGISTER_IMPORT_FUNC(rccl),
                                                    ROCP_REG_VERSION,
                                                    table_array.data(),
                                                    table_array.size(),
                                                    &lib_id);

            // INFO( NCCL_COLL,"[rocprofiler-sdk-rccl][ = %d ] rocprofiler-register returned code = %d : %s", getpid(),
            //     rocp_reg_status, rocprofiler_register_error_string(rocp_reg_status));

            // if(rocp_reg_status != ROCP_REG_SUCCESS && rocp_reg_status != ROCP_REG_NO_TOOLS)
            //     WARN( "[rocprofiler-sdk-rccl][%d] rocprofiler-register failed with error code %d : %s",  getpid(),
            //     rocp_reg_status, rocprofiler_register_error_string(rocp_reg_status));
        #endif

        return &tbl;
    }

    const rcclApiFuncTable* RcclGetFunctionTable()
    {
	 static rcclApiFuncTable* tbl = nullptr;
         if(tbl == nullptr)
             tbl = RcclGetFunctionTable_impl();

        return tbl;
    }
}


ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
      return ::rccl::RcclGetFunctionTable()->ncclAllGather_fn(sendbuff, recvbuff, sendcount,
                                     datatype, comm, stream);
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
      return ::rccl::RcclGetFunctionTable()->ncclAllReduce_fn(sendbuff, recvbuff, count,
                                     datatype, op, comm, stream);
}

ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream) {
    return ::rccl::RcclGetFunctionTable()->ncclAllToAll_fn(sendbuff, recvbuff, count,
                                     datatype, comm, stream);
}

ncclResult_t ncclAllToAllv(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream) {
      return ::rccl::RcclGetFunctionTable()->ncclAllToAllv_fn(sendbuff, sendcounts, sdispls, recvbuff,
        recvcounts, rdispls, datatype, comm, stream);
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
      return ::rccl::RcclGetFunctionTable()->ncclBroadcast_fn(sendbuff, recvbuff, count, datatype, root,
      comm, stream);
}

ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream) {
      return ::rccl::RcclGetFunctionTable()->ncclGather_fn(sendbuff, recvbuff, sendcount, datatype,
          root, comm, stream);
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
      return ::rccl::RcclGetFunctionTable()->ncclReduce_fn(sendbuff, recvbuff, count, datatype,
          op, root, comm, stream);
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
      return ::rccl::RcclGetFunctionTable()->ncclReduceScatter_fn(sendbuff, recvbuff, recvcount,
        datatype, op, comm, stream);
}

ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream) {
      return ::rccl::RcclGetFunctionTable()->ncclScatter_fn(sendbuff, recvbuff, recvcount, datatype, root,
        comm, stream);
}

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
      return ::rccl::RcclGetFunctionTable()->ncclSend_fn(sendbuff, count, datatype, peer,
        comm, stream);
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
      return ::rccl::RcclGetFunctionTable()->ncclRecv_fn(recvbuff, count, datatype, peer,
        comm, stream);
}


