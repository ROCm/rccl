/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#include "hip/hip_runtime.h"
#include "rccl.h"
#include "mpi.h"


#define MPICHECK(cmd) do {                          \
    int e = cmd;				    \
    if( e != MPI_SUCCESS ) {			    \
      printf("Failed: MPI error %s:%d '%d'\n",	    \
	     __FILE__,__LINE__, e);		    \
      exit(EXIT_FAILURE);			    \
    }						    \
  } while(0)


#define HIPCHECK(cmd) do {                         \
    hipError_t e = cmd;				   \
    if( e != hipSuccess ) {				 \
      printf("Failed: HIP error %s:%d '%s'\n",		 \
	     __FILE__,__LINE__,hipGetErrorString(e));	 \
      exit(EXIT_FAILURE);				 \
    }							 \
  } while(0)


#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;			    \
    if (r!= ncclSuccess) {				  \
      printf("Failed, NCCL error %s:%d '%s'\n",		  \
	     __FILE__,__LINE__,ncclGetErrorString(r));	  \
      exit(EXIT_FAILURE);				  \
    }							  \
  } while(0)

static void init_sendbuf (float *sendbuf, int count, int val)
{
  for (int i = 0; i < count; i++) {
    sendbuf[i] = (float)val+1;
  }
}

static void init_zero (float *recvbuf, int count)
{
  for (int i = 0; i < count; i++) {
    recvbuf[i] = 0.0;
  }
}

static bool check_recvbuf (float *recvbuf, int count, int ndevices)
{
  bool result = true;
  float expected=0.0;
  for (int i=0; i<ndevices; i++){
    expected += (float)i+1;
  }
  for (int i = 0; i < count; i++) {
    if (recvbuf[i] != expected) {
      result = false;
      printf("Element %d is %f expected %f\n", i, recvbuf[i], expected);
      break;
    }
  }
  return result;
}

static void print_help()
{
    printf("Usage: rccl-allreduce-mpi-multirank <distMode> <startDev> <numDevs> \n");
    printf("   all arguments are optional, but have to be provided in this order\n");
    printf("   distMode   : 0 - 1 (default: 0 - block distribution of rank to devices)\n");
    printf("   startDev   : id of first Device to use  (default: 0) \n");
    printf("   numDevs    : number of Devices to use   (default: 2) \n");
}

static int distmode=0;
static int startdev=0;
static int numdevices=2;
static int maxdevices=0;

static void devicemode_init( int argc, char **argv)
{
    char *modeexpl[4];
    int myRank;
    MPICHECK(MPI_Comm_rank (MPI_COMM_WORLD, &myRank));

    modeexpl[0] = strdup("0: contiguous assignment of ranks to devices");
    modeexpl[1] = strdup("1: round robin assignment of ranks to devices");

    if (argc > 1 ) {
        distmode = atoi(argv[1]);
    }
    if (argc > 2 ) {
        startdev = atoi(argv[2]);
    }
    if ( argc > 3 ) {
        numdevices = atoi(argv[3]);
    }
    if ( distmode > 1) {
        if ( myRank == 0 ) {
            printf("Unknown distribution mode %d. Known distribution modes are 0-1\n", distmode);
            print_help();
        }
        MPI_Abort (MPI_COMM_WORLD, -1);
    }
    HIPCHECK(hipGetDeviceCount(&maxdevices));
    if ( numdevices > maxdevices) {
        if ( myRank == 0 ) {
            printf("Requesting %d devices, %d devices available. Aborting.\n", numdevices, maxdevices);
            print_help();
        }
        MPI_Abort (MPI_COMM_WORLD, -1);
    }
    if ( startdev > maxdevices-1) {
        if ( myRank == 0 ) {
            printf("Startdevice is %d, max. number of devices is %d. Valid values are 0 - %d\n", startdev, maxdevices, maxdevices-1);
            print_help();
        }
        MPI_Abort (MPI_COMM_WORLD, -1);
    }

    if ( myRank == 0 ) {
        printf("Using binding mode %s\n", modeexpl[distmode]);
        printf("Starting devices is %d, %d devices used.\n\n", startdev, numdevices);
    }
}

static bool report_binding=true;

static void device_set(int id, int nDev)
{
    int dev=0;
    if (distmode == 0 ) {
        int tmp = (id*numdevices)/nDev;
        dev = (startdev+tmp)%maxdevices;
    }
    else if (distmode == 1) {
        dev = (startdev+id)%numdevices;
    }

    HIPCHECK(hipSetDevice(dev));
    if (report_binding) {
        printf("Rank %d using device %d\n", id, dev);
        if ( id == nDev-1) {
            report_binding=false;
        }
    }
}


int main(int argc, char* argv[])
{
  int size = 32*1024*1024;
  int myRank, nRanks, localRank = 0;
  ncclUniqueId id;
  ncclComm_t comm;
  float *h_sendbuff, *h_recvbuff;
  float *sendbuff, *recvbuff;
  hipStream_t s;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  //get deviceId to be used for each rank, e.g. localRank%numberOfDevices
  devicemode_init( argc, argv);
  int nDev = numdevices;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  //initializing RCCL
  device_set(myRank, nRanks);
  NCCLCHECK(ncclCommInitRankMulti(&comm, nRanks, id, myRank, myRank));

  //allocate buffers
  HIPCHECK(hipMalloc(&sendbuff, size * sizeof(float)));
  h_sendbuff = (float*) malloc ( size *sizeof(float));
  init_sendbuf(h_sendbuff, size, myRank);
  HIPCHECK(hipMemcpy(sendbuff, h_sendbuff, size * sizeof(float), hipMemcpyDefault));

  HIPCHECK(hipMalloc(&recvbuff, size * sizeof(float)));
  h_recvbuff = (float*) malloc ( size *sizeof(float));
  init_zero(h_recvbuff, size);
  HIPCHECK(hipMemcpy(recvbuff, h_recvbuff, size * sizeof(float), hipMemcpyDefault));

  HIPCHECK(hipStreamCreate(&s));

  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat,
                          ncclSum, comm, s));

  //completing NCCL operation by synchronizing on the HIP stream
  HIPCHECK(hipStreamSynchronize(s));

  //check result

  HIPCHECK(hipMemcpy(h_recvbuff, recvbuff, size*sizeof(float), hipMemcpyDefault));
  bool res = check_recvbuf(h_recvbuff, size, nRanks);
  printf("[%d] Checking buffer result is %s\n", myRank, res == true ? "correct" : "wrong" );

  //free buffers
  HIPCHECK(hipFree(sendbuff));
  free (h_sendbuff);
  HIPCHECK(hipFree(recvbuff));
  free (h_recvbuff);

  //finalizing NCCL
  ncclCommDestroy(comm);
  HIPCHECK(hipStreamDestroy(s));

  //finalizing MPI
  printf("[MPI Rank %d] Success \n", myRank);
  MPICHECK(MPI_Finalize());
  return 0;
}
