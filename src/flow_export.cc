// Copyright (c) 2023 Arista Networks, Inc.  All rights reserved.
// Licensed under the MIT License.

#include "flow_export.h"

#include "ibvwrap.h"
#include "net.h"

#include <atomic>
#include <dlfcn.h>

NCCL_PARAM( FlowExport, "FLOW_EXPORT", 0 );

// Make visible for tests.
__attribute__ ((visibility("default"))) FlowExport_v1 * flowExport_v1 = nullptr;
__attribute__ ((visibility("default"))) std::atomic< uint64_t > * currCommHash = nullptr;

namespace {

inline uint32_t
tpRank( struct ncclComm & comm, int rank ) {
   return rank == -1 ? UINT32_MAX : comm.topParentRanks[ rank ];
}

inline void
initFlowV1( const connectionMetaData_t & cmd, Flow_v1 & flow ) {
   flow.commHash = currCommHash[ cmd.tpLocalRank ];
   flow.direction = cmd.send ? SEND : RECEIVE;
   flow.srcRank = cmd.tpRank;
   flow.srcLocalRank = cmd.tpLocalRank;
   flow.dstRank = cmd.tpRemoteRank;
   flow.channel = cmd.channelId;
   flow.topology = cmd.connIndex == 0 ? COLLECTIVE : P2P;
}

inline void
gidToIpAddr( const union ibv_gid & gid, IpGenAddr & genAddr ) {
   // A gid.raw is 16 bytes, the size of an IPv6 address. If the address has a
   // prefix of ::ffff:0:0/96 (i.e., the first 10 bytes are zero, followed by
   // two 0xff bytes) bytes), then it actually represents an IPv4 address, which
   // is held in the last 4 bytes.
   // See: https://en.wikipedia.org/wiki/IPv6_address#Transition_from_IPv4
   static const uint8_t ipv4Prefix[ 12 ] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff
   };
   if ( gid.global.subnet_prefix == 0 && memcmp( gid.raw, ipv4Prefix, 12 ) == 0 ) {
      // IPv4
      genAddr.isIpv4 = true;
      memcpy( &genAddr.addr.ipv4Addr.s_addr, gid.raw + 12, 4 );
   } else {
      // IPv6
      genAddr.isIpv4 = false;
      memcpy( genAddr.addr.ipv6Addr.s6_addr, gid.raw, 16 );
   }
}

inline bool
fdToIpAddr( int fd, struct sockaddr * addr, socklen_t len ) {
   bzero( addr, len );
   if ( getsockname( fd, addr, &len ) ) {
      perror( "getsockname() failed" );
      return false;
   }
   return true;
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
ncclExportSocketFlow( const connectionMetaData_t & cmd,
                      struct ncclSocket & ctrlSock,
                      struct ncclSocket socks[],
                      int nSocks ) {
   if ( !flowExport_v1 || !flowExport_v1->exportFlow ) {
      return ncclSuccess;
   }

   Flow_v1 flow;
   initFlowV1( cmd, flow );

   flow.trafficType = TCP;
   flow.srcQPair = 0;
   flow.dstQPair = 0;

   for ( int i = 0; i <= nSocks; ++i ) {
      struct ncclSocket & sock = ( i == nSocks ) ? ctrlSock : socks[ i ];
      if ( sock.addr.sa.sa_family == AF_INET ) {
         struct sockaddr_in srcAddr;
         if ( !fdToIpAddr( sock.fd,
                           reinterpret_cast< struct sockaddr * >( &srcAddr ),
                           sizeof( srcAddr ) ) ) {
            return ncclSystemError;
         }
         flow.srcIp.isIpv4 = true;
         flow.srcIp.addr.ipv4Addr = srcAddr.sin_addr;
         flow.srcPort = ntohs( srcAddr.sin_port );

         flow.dstIp.isIpv4 = true;
         flow.dstIp.addr.ipv4Addr = sock.addr.sin.sin_addr;
         flow.dstPort = ntohs( sock.addr.sin.sin_port );
      } else if ( sock.addr.sa.sa_family == AF_INET6 ) {
         struct sockaddr_in6 srcAddr;
         if ( !fdToIpAddr( sock.fd,
                           reinterpret_cast< struct sockaddr * >( &srcAddr ),
                           sizeof( srcAddr ) ) ) {
            return ncclSystemError;
         }
         flow.srcIp.isIpv4 = false;
         flow.srcIp.addr.ipv6Addr = srcAddr.sin6_addr;
         flow.srcPort = ntohs( srcAddr.sin6_port );

         flow.dstIp.isIpv4 = false;
         flow.dstIp.addr.ipv6Addr = sock.addr.sin6.sin6_addr;
         flow.dstPort = ntohs( sock.addr.sin6.sin6_port );
      } else {
         WARN( "nclExportSocketFlow: socket of unknown address family %d",
               sock.addr.sa.sa_family );
         continue;
      }

      ncclResult_t result = flowExport_v1->exportFlow( &flow );
      if ( result != ncclSuccess ) {
         WARN( "flowExport_v1.exportFlow() failed: code %d", result );
         return result;
      }
   }

   return ncclSuccess;
}

ncclResult_t
ncclExportIbFlow( const connectionMetaData_t & cmd,
                  const union ibv_gid & localGid,
                  const union ibv_gid & remoteGid,
                  struct ibv_qp ** srcQPairs,
                  const uint32_t * dstQPairs,
                  int nQPairs ) {
   if ( !flowExport_v1 || !flowExport_v1->exportFlow ) {
      return ncclSuccess;
   }

   Flow_v1 flow;
   initFlowV1( cmd, flow );

   flow.trafficType = ROCEv2;
   gidToIpAddr( localGid, flow.srcIp );
   gidToIpAddr( remoteGid, flow.dstIp );
   flow.srcPort = 0;
   flow.dstPort = 0;

   for ( int i = 0; i < nQPairs; ++i ) {
      flow.srcQPair = srcQPairs[ i ]->qp_num;
      flow.dstQPair = dstQPairs[ i ];

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
   if ( currCommHash ) {
      currCommHash[ tpLocalRank ] = 0;
   }
   return result;
}
