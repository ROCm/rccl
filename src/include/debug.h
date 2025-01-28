/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_DEBUG_H_
#define NCCL_INT_DEBUG_H_

#include "nccl.h"
#include "nccl_common.h"
#include <stdio.h>

#include <pthread.h>

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16
#define LOG_VERBS
extern int ncclDebugLevel;
extern FILE *ncclDebugFile;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) __attribute__ ((format (printf, 5, 6)));

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
extern char ncclLastError[];

#define VERSION(...) ncclDebugLog(NCCL_LOG_VERSION, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define TRACE_CALL(...) ncclDebugLog(NCCL_LOG_TRACE, NCCL_CALL, __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclDebugLog(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#else
#define TRACE(...)
#endif

#ifdef LOG_VERBS
#define LOG_SEND_WR(WR, QP_NUM, SRC, DST)                           \
  do {                                                              \
    struct ibv_send_wr* wr_ptr = WR;                                \
    int wr_index = 0;                                               \
    while (wr_ptr != nullptr) {                                     \
      ncclDebugLog(NCCL_LOG_TRACE, NCCL_VERBS, __func__, __LINE__,  \
                  "Posted send wr_id=%lu, wr_indx=%d, qp_num=%d, src_nic=%d, dst_nic=%d, opcode=%d, send_flags=%d, imm_data=%d, remote_addr=%lx, rkey=%x, length=%d, lkey=%x",\
                  wr_ptr->wr_id, wr_index, QP_NUM, SRC, DST, wr_ptr->opcode,\
                  wr_ptr->send_flags, wr_ptr->imm_data,             \
                  wr_ptr->wr.rdma.remote_addr, wr_ptr->wr.rdma.rkey,\
                  wr_ptr->sg_list ? wr_ptr->sg_list->length : 0,    \
                  wr_ptr->sg_list ? wr_ptr->sg_list->lkey : 0);     \
      wr_ptr = wr_ptr->next;                                        \
      wr_index++;                                                   \
    }                                                               \
  } while(0);
#define LOG_RECV_WR(WR, DST)                                      \
  do {                                                            \
    struct ibv_recv_wr* wr_ptr = WR;                              \
    int wr_index = 0;                                             \
    while (wr_ptr != nullptr) {                                   \
      ncclDebugLog(NCCL_LOG_TRACE, NCCL_VERBS, __func__, __LINE__,\
                   "Posted recv wr_id=%lu, wr_indx=%d, dst=%d, num_sge=%d, sg_list_length=%d, sg_list_lkey=%x",\
                    wr_ptr->wr_id, wr_index, DST, wr_ptr->num_sge,\
                    wr_ptr->sg_list ? wr_ptr->sg_list->length : 0,\
                    wr_ptr->sg_list ? wr_ptr->sg_list->lkey : 0); \
      wr_ptr = wr_ptr->next;                                      \
      wr_index++;                                                 \
    }                                                             \
  } while(0);
#else
#define LOG_SEND_WR(...)
#define LOG_RECV_WR(...)
#endif

void ncclSetThreadName(pthread_t thread, const char *fmt, ...);

#endif
