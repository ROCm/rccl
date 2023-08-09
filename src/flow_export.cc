// Copyright (c) 2023 Arista Networks, Inc.  All rights reserved.
// Licensed under the MIT License.

#include "flow_export.h"

#include "net.h"

#include <atomic>
#include <dlfcn.h>

NCCL_PARAM( FlowExport, "FLOW_EXPORT", 0 );

FlowExport_v1 * flowExport_v1 = nullptr;
std::atomic< uint64_t > * currCommHash = nullptr;

namespace {

inline uint32_t
tpRank( struct ncclComm & comm, int rank ) {
   return rank == -1 ? UINT32_MAX : comm.topParentRanks[ rank ];
}

} // namespace

ncclResult_t
ncclInitFlowExport() {
   if ( !ncclParamFlowExport() ) {
      INFO( NCCL_INIT, "flow export disabled" );
      return ncclSuccess;
   }

   if ( flowExport_v1 ) {
      INFO( NCCL_INIT, "flow export already initialized" );
      return ncclSuccess;
   }

   const char * pluginName = getenv( "FLOW_EXPORT_PLUGIN_LIB" );
   if ( !pluginName || !pluginName[ 0 ] ) {
      pluginName = "libnccl_flow_export_v1.so";
   }
   void * lib = dlopen( pluginName, RTLD_NOW | RTLD_LOCAL );
   if ( !lib ) {
      std::string err = dlerror();
      WARN( "failed to load flow export plugin %s: %s", pluginName, err.c_str() );
      return ncclSystemError;
   }
   INFO( NCCL_INIT, "loaded flow export plugin %s", pluginName );

   flowExport_v1 =
      reinterpret_cast< FlowExport_v1 * >( dlsym( lib, "flowExport_v1" ) );
   if ( !flowExport_v1 ) {
      std::string err = dlerror();
      WARN( "failed to find flowExport_v1 struct: %s", err.c_str() );
      return ncclSystemError;
   }

   currCommHash = new std::atomic< uint64_t >[ NCCL_MAX_LOCAL_RANKS ]();

   if ( !flowExport_v1->init ) {
      INFO( NCCL_INIT, "flowExport_v1.init not defined, skipping" );
      return ncclSuccess;
   }

   ncclResult_t result = flowExport_v1->init();
   if ( result != ncclSuccess ) {
      WARN( "flowExport_v1.init() failed: code %d", result );
   }
   return result;
}

ncclResult_t
ncclExitFlowExport() {
   ncclResult_t result = ncclSuccess;

   if ( flowExport_v1 && flowExport_v1->exit ) {
      result = flowExport_v1->exit();
      if ( result == ncclSuccess ) {
         INFO( NCCL_INIT, "unloaded flow export plugin" );
      } else {
         WARN( "flowExport_v1.exit() failed: code %d", result );
      }
   }

   flowExport_v1 = nullptr;
   delete[] currCommHash;
   currCommHash = nullptr;

   return result;
}

ncclResult_t
ncclExportFlow( const connectionMetaData_t & cmd,
                struct ncclSocket & ctrlSock,
                struct ncclSocket socks[],
                int nSocks ) {
   if ( !flowExport_v1 || !flowExport_v1->exportFlow ) {
      return ncclSuccess;
   }

   Flow_v1 flow;

   flow.commHash = currCommHash[ cmd.tpLocalRank ];
   flow.direction = cmd.send ? SEND : RECEIVE;
   flow.srcRank = cmd.tpRank;
   flow.srcLocalRank = cmd.tpLocalRank;
   flow.dstRank = cmd.tpRemoteRank;
   flow.channel = cmd.channelId;
   flow.topology = cmd.connIndex == 0 ? COLLECTIVE : P2P;

   for ( int i = 0; i <= nSocks; ++i ) {
      struct ncclSocket & sock = ( i == nSocks ) ? ctrlSock : socks[ i ];
      flow.trafficType = UDP; // TODO
      flow.qPair = 0; // TODO

      // This assumes v4 :(
      struct sockaddr_in srcAddr;
      bzero( &srcAddr, sizeof( srcAddr ) );
      socklen_t len = sizeof( srcAddr );
      if ( getsockname(
              sock.fd, reinterpret_cast< struct sockaddr * >( &srcAddr ), &len ) ) {
         perror( "getsockname() failed" );
         return ncclSystemError;
      }
      flow.srcIp = ntohl( srcAddr.sin_addr.s_addr );
      flow.srcPort = ntohs( srcAddr.sin_port );
      flow.dstIp = ntohl( sock.addr.sin.sin_addr.s_addr );
      flow.dstPort = ntohs( sock.addr.sin.sin_port );

      ncclResult_t result = flowExport_v1->exportFlow( &flow );
      if ( result != ncclSuccess ) {
         WARN( "flowExport_v1.exportFlow() failed: code %d", result );
         return result;
      }
   }

   return ncclSuccess;
}

ncclResult_t
ncclExportComm( struct ncclComm & comm ) {
   if ( !flowExport_v1 || !flowExport_v1->exportComm ) {
      return ncclSuccess;
   }

   uint32_t tpLocalRank = comm.topParentLocalRanks[ comm.localRank ];
   currCommHash[ tpLocalRank ] = comm.commHash;

   Comm_v1 commExport;
   commExport.commHash = comm.commHash;

   commExport.nRanks = comm.nRanks;
   commExport.rank = comm.rank;
   commExport.tpRank = tpRank( comm, comm.rank );

   commExport.nLocalRanks = comm.localRanks;
   commExport.localRank = comm.localRank;
   commExport.tpLocalRank = tpLocalRank;

   commExport.nNodes = comm.nNodes;
   commExport.node = comm.node;

   commExport.nChannels = comm.nChannels;

   for ( int c = 0; c < comm.nChannels; ++c ) {
      struct ncclChannel & channel = comm.channels[ c ];
      commExport.ring[ c ].tpPrev = tpRank( comm, channel.ring.prev );
      commExport.ring[ c ].tpNext = tpRank( comm, channel.ring.next );
      commExport.tree[ c ].tpUp = tpRank( comm, channel.tree.up );
      for ( int n = 0; n < NCCL_MAX_TREE_ARITY; ++n ) {
         commExport.tree[ c ].tpDown[ n ] = tpRank( comm, channel.tree.down[ n ] );
      }
   }

   ncclResult_t result = flowExport_v1->exportComm( &commExport );
   if ( result != ncclSuccess ) {
      WARN( "flowExport_v1.exportComm() failed: code %d", result );
   }

   return result;
}

ncclResult_t
ncclExportDone( struct ncclComm & comm ) {
   int tpLocalRank = comm.topParentLocalRanks[ comm.localRank ];
   ncclResult_t result = ncclSuccess;
   if ( flowExport_v1 && flowExport_v1->exportDone ) {
      result = flowExport_v1->exportDone( currCommHash[ tpLocalRank ],
                                          tpRank( comm, comm.rank ) );
      if ( result != ncclSuccess ) {
         WARN( "flowExport_v1.exportDone() failed: code %d", result );
      }
   }
   currCommHash[ tpLocalRank ] = 0;
   return result;
}
