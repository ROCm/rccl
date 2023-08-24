// Copyright (c) 2023 Arista Networks, Inc.  All rights reserved.
// Licensed under the MIT License.

#include "TestBed.hpp"

#include "flow_export.h"
#include "proxy.h"

#include <atomic>

ncclResult_t
testExportComm( const Comm_v1 * comm ) {
   // TODO: Capture the comm info and assert in the test that this matches what
   // we expect.
   return ncclSuccess;
}

extern FlowExport_v1 * flowExport_v1;
extern std::atomic< uint64_t > * currCommHash;

FlowExport_v1 testFlowExport{
   .exportComm = testExportComm,
};

void SetupMockFlowExport() {
   // Fake the plugin setup without actually dynamically loading a shared lib.
   flowExport_v1 = &testFlowExport;
   currCommHash = new std::atomic< uint64_t >[ NCCL_MAX_LOCAL_RANKS ]();
}

void TeardownMockFlowExport() {
   flowExport_v1 = nullptr;
   delete[] currCommHash;
   currCommHash = nullptr;
}

namespace RcclUnitTesting {

TEST( FlowExport, CommExport ) {
   SetupMockFlowExport();

   TestBed testBed;

   // Configuration
   std::vector< ncclFunc_t > const funcTypes = { ncclCollAllReduce };
   std::vector< ncclDataType_t > const dataTypes = { ncclFloat32 };
   std::vector< ncclRedOp_t > const redOps = { ncclSum };
   std::vector< int > const roots = { 0 };
   std::vector< int > const numElements = { 393216, 384 };
   std::vector< bool > const inPlaceList = { false };
   std::vector< bool > const managedMemList = { false };
   std::vector< bool > const useHipGraphList = { false };

   testBed.RunSimpleSweep( funcTypes,
                           dataTypes,
                           redOps,
                           roots,
                           numElements,
                           inPlaceList,
                           managedMemList,
                           useHipGraphList );
   testBed.Finalize();

   TeardownMockFlowExport();
}

} // namespace RcclUnitTesting
