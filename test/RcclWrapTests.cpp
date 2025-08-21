/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <comm.h>
#include <graph/topo.h>

#include "TestBed.hpp"


namespace RcclUnitTesting
{

  // Helper function to test the static expose check
  ncclResult_t testStaticExposeCheck() {
    RCCL_STATIC_EXPOSE_CHECK();
    return ncclSuccess;
  }

  TEST(Rcclwrap, RcclFuncMaxSendRecvCount) {
    ncclResult_t staticCheckResult = testStaticExposeCheck();
    #ifdef RCCL_EXPOSE_STATIC
    EXPECT_EQ(staticCheckResult, ncclSuccess);
    #else
    EXPECT_EQ(staticCheckResult, ncclInvalidUsage);
    #endif

    size_t maxCount = 0;
    ncclResult_t result = rcclFuncMaxSendRecvCount(ncclFuncAllReduce, 4, 1024, maxCount);
    EXPECT_EQ(maxCount, 1024);
    EXPECT_EQ(result, ncclSuccess);
  }

  TEST(Rcclwrap, RcclUpdateCollectiveProtocol_UsesLL128WhenInRange) {
    setenv("NCCL_PROTO", "", 1); // Trigger auto selection mode
    unsetenv("NCCL_PROTO");

    ncclComm_t comm = new ncclComm();
    *comm = {};
    // Manually populate minimal fields for comm
    comm->nRanks = 1;
    comm->nNodes = 2;  // triggers inter-node logic
    comm->rank=0;
    comm->topo =  new ncclTopoSystem();
    *comm->topo = {};
    comm->topo->ll128Enabled=true;
    comm->topo->nodes[GPU].nodes[0] = {};
    comm->topo->nodes[GPU].count = 1;
    strncpy(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942", sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn));

    int idx = rcclGetTunableIndex(ncclFuncAllReduce);
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX] = 512;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX] = 1024;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MIN_IDX] = 256;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MAX_IDX] = 2048;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_FACTOR_IDX] = 1;

    ncclTaskColl info = {};
    // Manually populate minimal fields for info
    info.func = ncclFuncAllReduce;
    info.protocol = NCCL_PROTO_UNDEF;

    size_t nBytes = 1024;

    rcclUpdateCollectiveProtocol(comm, nBytes, &info);
    EXPECT_TRUE(info.protocol == NCCL_PROTO_LL128 || info.protocol == NCCL_PROTO_LL);

    delete comm->topo;
    delete comm;
  }

  TEST(Rcclwrap, RcclUpdateCollectiveProtocol_WarnsOnGfx942Arch) {
    setenv("NCCL_PROTO", "", 1);
    unsetenv("NCCL_PROTO");

    ncclComm_t comm = new ncclComm();
    *comm = {};
    // Manually populate minimal fields for comm
    comm->nRanks = 1;
    comm->nNodes = 2;  // triggers inter-node logic
    comm->rank=0;
    comm->topo =  new ncclTopoSystem();
    comm->topo->ll128Enabled=true;
    comm->topo->nodes[GPU].nodes[0] = {};
    strncpy(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942", sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn));

    int idx = rcclGetTunableIndex(ncclFuncAllReduce);
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX] = RCCL_LL_LIMITS_UNDEFINED;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX] = RCCL_LL_LIMITS_UNDEFINED;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MIN_IDX] = RCCL_LL_LIMITS_UNDEFINED;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_MAX_IDX] = RCCL_LL_LIMITS_UNDEFINED;
    comm->minMaxLLRange[idx][NCCL_PROTO_LL128][RCCL_PROTOCOL_FACTOR_IDX] = RCCL_LL_LIMITS_UNDEFINED;

    ncclTaskColl info = {};
    // Manually populate minimal fields for info
    info.func = ncclFuncAllReduce;
    info.protocol = NCCL_PROTO_UNDEF;
    size_t nBytes = 1024; // 1024 per rank for 4 ranks

    rcclUpdateCollectiveProtocol(comm, nBytes, &info);
    EXPECT_EQ(info.protocol, NCCL_PROTO_UNDEF);

    delete comm->topo;
    delete comm;
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_HonorsUserProtocolEnv) {   //Why does this pass if it does not enter the else if block
  setenv("NCCL_PROTO", "1", 1);  // Simulate manual override

  ncclComm_t comm = new ncclComm();
  *comm = {};
  // Manually populate minimal fields for comm
  comm->nRanks = 1;
  comm->nNodes = 2;  // triggers inter-node logic
  comm->rank=0;
  comm->topo =  new ncclTopoSystem(); //(struct ncclTopoSystem*)calloc(1, sizeof(struct ncclTopoSystem));
  *comm->topo = {};
  comm->topo->ll128Enabled=true;
  comm->topo->nodes[GPU].nodes[0] = {};
  strncpy(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942", sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn));

  ncclTaskColl info = {};
  // Manually populate minimal fields for info
  info.func = ncclFuncAllReduce;
  info.protocol = NCCL_PROTO_UNDEF;
  size_t nBytes = 1024; // 1024 per rank for 4 ranks

  rcclUpdateCollectiveProtocol(comm, nBytes, &info);
  EXPECT_EQ(info.protocol, NCCL_PROTO_UNDEF);

  delete comm->topo;
  delete comm;
}

TEST(Rcclwrap, RcclUpdateCollectiveProtocol_SimpleFallbackWhenNoRanges) {
  setenv("NCCL_PROTO", "", 1); // Trigger auto selection mode
  unsetenv("NCCL_PROTO");

  ncclComm_t comm = new ncclComm();
  *comm = {};
  // Manually populate minimal fields for comm
  comm->nRanks = 1;
  comm->nNodes = 2;  // triggers inter-node logic
  comm->rank=0;
  comm->topo =  new ncclTopoSystem(); //(struct ncclTopoSystem*)calloc(1, sizeof(struct ncclTopoSystem));
  *comm->topo = {};
  comm->topo->ll128Enabled=true;
  comm->topo->nodes[GPU].nodes[0] = {};
  comm->topo->nodes[GPU].count = 1;
  strncpy(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942", sizeof(comm->topo->nodes[GPU].nodes[0].gpu.gcn));

  int idx = rcclGetTunableIndex(ncclFuncAllReduce);
  comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX] = 512;
  comm->minMaxLLRange[idx][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX] = 1024;


  // Manually populate minimal fields for info
  ncclTaskColl info = {};
  info.func = ncclFuncAllReduce;
  info.protocol = NCCL_PROTO_UNDEF;
  size_t nBytes = 2048; // 1024 per rank for 4 ranks

  rcclUpdateCollectiveProtocol(comm, nBytes, &info);
  EXPECT_EQ(info.protocol, NCCL_PROTO_SIMPLE);

  delete comm->topo;
  delete comm;
}
} //RcclUnitTesting
