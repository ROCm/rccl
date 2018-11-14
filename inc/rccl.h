/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rccl.h
 * @brief rccl library header
 *
 * This header contains all RCCL data structures and APIs
 *
 * @author Aditya Atluri
 */

#pragma once

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

//! Data type RCCL operation is done on
//! rcclDataType_t is passed to a rccl operation such as rcclAllReduce, where
//! reduction happens on following data types described
/* Data types */
typedef enum {
    rcclChar       = 0,   //!< signed char
    rcclInt        = 1,   //!< signed integer
    rcclHalf       = 2,   //!< half-precision floating point
    rcclFloat      = 3,   //!< single-precision floating point
    rcclDouble     = 4,   //!< double-precision floating point
    rcclInt64      = 5,   //!< signed 64-bit integer
    rcclUint64     = 6,   //!< unsigned 64-bit integer
    rccl_NUM_TYPES = 7    //!< total number of data types supported by rccl
} rcclDataType_t;

//! Return status from RCCL calls
//! When a RCCL API is executed, one of the following codes are returned by the
//! API. Meaning of each code is written below
typedef enum {
    rcclSuccess = 0,        //!< Success
    rcclUnhandledHipError,  //!< An error occured while using HIP which is not
                            //!< handled
    rcclSystemError,        //!< System error
    rcclInternalError,      //!< RCCL internal error
    rcclInvalidDevicePointer,  //!< One of the pointers passed to RCCL API is
                               //!< not valid
    rcclInvalidRank,  //!< Rank of the gpu being accessed by RCCL API is not
                      //!< valid
    rcclUnsupportedDeviceCount,  //!< More gpus are requested than prescribed
    rcclDeviceNotFound,      //!< Device the API is requested on is not found
    rcclInvalidDeviceIndex,  //!< Invalid HIP device index
    rcclLibWrapperNotSet,    //!< Wrapper library is not set
    rcclHipMallocFailed,     //!< HIP memory allocation failed
    rcclRankMismatch,        //!< Rank mismatch
    rcclInvalidArgument,     //!< One of the arguments is not valid
    rcclInvalidType,         //!< rcclDataType_t passed in to API is not valid
    rcclInvalidOperation,    //!< rcclOp_t passed in to API is not valid
    rccl_NUM_RESULTS
} rcclResult_t;

//! Type of reduction operation
//! For doing reduction operation, one of the following value is passed to RCCL
//! reduction APIs.
typedef enum {
    rcclSum = 0,  //!< Addition sum = a + b
    rcclProd,     //!< Multiplication mul = a * b
    rcclMax,      //!< Maximum max = a > b ? a : b
    rcclMin,      //!< Minimum min = a < b ? a : b
    rccl_NUM_OPS  //!< Total number of ops RCCL supports
} rcclRedOp_t;

//! rcclComm_t is communicator structure intialized for each gpu in the clique
//! which stores relevant gpu information to do a RCCL operation.
typedef struct RcclComm_t* rcclComm_t;

//! rcclUniqueId is initialized per clique and is used to create communicators
//! for each gpu
typedef struct RcclUniqueId* rcclUniqueId;

//! Returns RCCL API status as string

//! \param [in] result
const char* rcclGetErrorString(rcclResult_t result);

//! Generates rcclUniqueId

//! \param [in] uniqueId Memory location to rcclUniqueId created by application
rcclResult_t rcclGetUniqueId(rcclUniqueId* uniqueId);

//! Create RCCL communicator for a GPU based on rank, commId and number of
//! devices in the clique

//! \param [out] comm Memory location to rcclComm_t created by application
//! \param [in] ndev Number of devices in the clique
//! \param [in] commId rcclUniqueId with which the communicator is created with
//! \param [in] rank Rank of the gpu the communicator will be created with
rcclResult_t rcclCommInitRank(rcclComm_t* comm, int ndev, rcclUniqueId commId,
                              int rank);

//! Create an array of communicators from device list of length ndev

//! \param [out] comm Memory location to ndev number of communicators
//! \param [in] ndev Number of devices forming clique
//! \param [in] devlist List of HIP device indices
rcclResult_t rcclCommInitAll(rcclComm_t* comm, int ndev, int* devlist);

//! Get HIP device index from communicator

//! \param [in] comm
//! \param [out] dev HIP device index associated with communicator
rcclResult_t rcclCommCuDevice(rcclComm_t comm, int* dev);

//! Get rank from communicator

//! \param [in] comm
//! \param [out] rank Rank of the gpu associated with communicator
rcclResult_t rcclCommUserRank(rcclComm_t comm, int* rank);

//! Get number of gpus in clique the communicator participates in

//! \param [in] comm
//! \param [out] count Number of gpus present in communicators clique
rcclResult_t rcclCommCount(rcclComm_t comm, int* count);

//! Destroy communicator

//! \param [in] comm Communicator to be destroyed
rcclResult_t rcclCommDestroy(rcclComm_t comm);

//! Does reduction op on sendbuff on all gpus and store the result to all gpus
//! on recvbuff Reduction op (rcclRedOp_t) is done on data (of data type
//! rcclDataType_t) in sendbuff of length = count on all gpus and stored in
//! recvbuff on all gpus. The operation is launched on the stream provided. To
//! do in-place all-reduce, pass in same pointers to recvbuff and sendbuff (for
//! in-place all-reduce, sendbuff = recvbuff)

//! \param [in] sendbuff Source buffer
//! \param [in] recvbuff Destination buffer
//! \param [in] count Number of elements in buffer
//! \param [in] datatype Data type of buffers
//! \param [in] op Reduction operation on buffers
//! \param [in] comm Communicator for current gpu
//! \param [in] stream HIP stream the op launches on
rcclResult_t rcclAllReduce(const void* sendbuff, void* recvbuff, int count,
                           rcclDataType_t datatype, rcclRedOp_t op,
                           rcclComm_t comm, hipStream_t stream);

//! Data (of data type rcclDataType_t) present in root gpus buff of length count
//! is broadcasted to all other gpus. The operation is launched on stream
//! provided.

//! \param [in] buff Source buffer for root gpu, destination buffer for non-root
//! gpus \param [in] count Number of elements in buffer \param [in] datatype
//! Data type of buffers \param [in] root HIP device index of the root gpu
//! \param [in] comm Communicator for current gpu
//! \param [in] stream HIP stream the op launches on
rcclResult_t rcclBcast(void* buff, int count, rcclDataType_t datatype, int root,
                       rcclComm_t comm, hipStream_t stream);

//! Does reduction op on sendbuff on all gpus and stores result in recvbuff of
//! root gpu Reduction op (rcclRedOp_t) is done on data (of data type
//! rcclDataType_t) in sendbuff of length = count on all gpus and store in
//! recvbuff on root gpu. The operation is launched on the stream provided. The
//! value to recvbuff for non-root gpus can be null.

//! \param [in] sendbuff Source buffer
//! \param [in] recvbuff Destination buffer
//! \param [in] count Number of elements in buffer
//! \param [in] datatype Data type of buffers
//! \param [in] op Reduction operation on buffers
//! \param [in] root HIP device index of the root gpu
//! \param [in] comm Communicator for current gpu
//! \param [in] stream HIP stream the op launches on
rcclResult_t rcclReduce(const void* sendbuff, void* recvbuff, int count,
                        rcclDataType_t datatype, rcclRedOp_t op, int root,
                        rcclComm_t comm, hipStream_t stream);

//! Each gpu gathers values from other gpus' sendbuff into its recvbuff.
//! Size of recvbuff needs to be count*num_of_gpus.
//! The data is ordered by comm's device ranking.

//! \param [in] sendbuff Source buffer
//! \param [in] recvbuff Destination buffer
//! \param [in] count Number of elements in buffer
//! \param [in] datatype Data type of buffers
//! \param [in] comm Communicator for current gpu
//! \param [in] stream HIP stream the op launches on
rcclResult_t rcclAllGather(const void* sendbuff, int count, rcclDataType_t datatype,
                            void* recvbuff, rcclComm_t comm, hipStream_t stream);

//! Does reduction op on sendbuff on all gpus and scatter the result to all gpus'
//! recvbuff. Reduction op (rcclRedOp_t) is done on data (of data type
//! rcclDataType_t) in sendbuff of length = recvcount*num_gpus on all gpus and
//! stored in recvbuff with length = recvcount on all gpus. The operation is
//! launched on the stream provided. To do in-place reduce scatter, pass in
//! same pointers to recvbuff and sendbuff (sendbuff = recvbuff)

//! \param [in] sendbuff Source buffer
//! \param [in] recvbuff Destination buffer
//! \param [in] recvcount Number of elements in receive buffer
//! \param [in] datatype Data type of buffers
//! \param [in] op Reduction operation on buffers
//! \param [in] comm Communicator for current gpu
//! \param [in] stream HIP stream the op launches on
rcclResult_t rcclReduceScatter(const void* sendbuff, void* recvbuff, int recvcount,
                           rcclDataType_t datatype, rcclRedOp_t op,
                           rcclComm_t comm, hipStream_t stream);
#ifdef __cplusplus
}  // end extern "C"
#endif
