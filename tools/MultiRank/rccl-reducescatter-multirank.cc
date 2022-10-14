/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
#include <stdio.h>
#include "hip/hip_runtime.h"
#include "rccl.h"

#define HIPCHECK(cmd) do {                         \
  hipError_t e = cmd;                              \
  if( e != hipSuccess ) {                          \
    printf("Failed: HIP error %s:%d '%s'\n",             \
        __FILE__,__LINE__,hipGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static void init_sendbuf (float *sendbuf, int count, int val)
{
    for (int i = 0; i < count; i++) {
        sendbuf[i] = (float)val;
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


static int distmode=0;
static int startdev=0;
static int numdevices=2;
static int maxdevices=0;
static int ranksperdev=1;

static void print_help()
{
    printf("Usage: rccl-reducescatter-multirank <distMode> <startDev> <numDevs> <ranksPerDev> \n");
    printf("   all arguments are optional, but have to be provided in this order\n");
    printf("   distMode   : 0 - 3 (default: 0 - all ranks are on different devices)\n");
    printf("   startDev   : id of first Device to use  (default: 0) \n");
    printf("   numDevs    : number of Devices to use   (default: 2) \n");
    printf("   ranksPerDev: number of Ranks per Device (default: 1) \n");
}

static void devicemode_init( int argc, char **argv)
{
    char *modeexpl[4];

    modeexpl[0] = strdup("0: all ranks are on different devices");
    modeexpl[1] = strdup("1: all ranks are on same device");
    modeexpl[2] = strdup("2: contiguous assignment of ranks to devices");
    modeexpl[3] = strdup("3: round robin assignment of ranks to devices");

    if (argc > 1 ) {
        distmode = atoi(argv[1]);
    }
    if (argc > 2 ) {
        startdev = atoi(argv[2]);
    }
    if ( argc > 3 ) {
        numdevices = atoi(argv[3]);
    }
    if ( argc > 4 ) {
        ranksperdev = atoi(argv[4]);
    }

    if ( distmode > 3) {
        printf("Unknown distribution mode %d. Known distribution modes are 0-3\n", distmode);
        print_help();
        exit(-1);
    }
    HIPCHECK(hipGetDeviceCount(&maxdevices));
    if ( numdevices > maxdevices) {
        printf("Requesting %d devices, %d devices available. Aborting.\n", numdevices, maxdevices);
        print_help();
        exit(-1);
    }
    if ( startdev > maxdevices-1) {
        printf("Startdevice is %d, max. number of devices is %d. Valid values are 0 - %d\n", startdev, maxdevices, maxdevices-1);
        print_help();
        exit(-1);
    }

    if (distmode == 1) numdevices  = 1;
    if (distmode == 0) ranksperdev = 1;

    printf("Using binding mode %s\n", modeexpl[distmode]);
    printf("Starting devices is %d, %d devices used, %d ranks per device.\n\n", startdev, numdevices, ranksperdev);
}

static bool report_binding=true;

static void device_set(int id, int nDev)
{
    int dev=0;
    if (distmode == 0 )
        dev = (startdev+id)%numdevices;
    else if (distmode == 1) {
        dev = startdev;
    }
    else if (distmode == 2) {
        int tmp = (id*numdevices)/nDev;
        dev = (startdev+tmp)%maxdevices;
    }
    else if (distmode == 3) {
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
  int nDev;

  devicemode_init( argc, argv);
  nDev = numdevices * ranksperdev;

  int sendsize = 32*1024*1024;
  int recvsize  = sendsize / nDev;

  //allocating and initializing device buffers
  float** h_sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** sendbuff   = (float**)malloc(nDev * sizeof(float*));
  float** h_recvbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff   = (float**)malloc(nDev * sizeof(float*));
  hipStream_t* s     = (hipStream_t*)malloc(sizeof(hipStream_t)*nDev);
  ncclComm_t* comms  = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nDev);

  for (int i = 0; i < nDev; ++i) {
      device_set(i, nDev);

      HIPCHECK(hipMalloc(sendbuff+i, sendsize * sizeof(float)));
      h_sendbuff[i] = (float*) malloc (sendsize *sizeof(float));
      init_sendbuf(h_sendbuff[i], sendsize, i+1);
      HIPCHECK(hipMemcpy(sendbuff[i], h_sendbuff[i], sendsize * sizeof(float), hipMemcpyDefault));

      HIPCHECK(hipMalloc(recvbuff+i, recvsize*sizeof(float)));
      h_recvbuff[i] = (float*) malloc (recvsize *sizeof(float));
      HIPCHECK(hipMemset(recvbuff[i], 0, recvsize*sizeof(float)));
      HIPCHECK(hipStreamSynchronize(NULL));
      HIPCHECK(hipStreamCreate(s+i));
  }


  //initializing NCCL
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++ ) {
      device_set(i, nDev);
      NCCLCHECK(ncclCommInitRankMulti(&comms[i], nDev, id, i, i));
  }
  NCCLCHECK(ncclGroupEnd());

  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
      NCCLCHECK(ncclReduceScatter((const void*)sendbuff[i], (void*)recvbuff[i], recvsize, ncclFloat, ncclSum, comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());


  //synchronizing on HIP streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
      device_set(i, nDev);
      HIPCHECK(hipStreamSynchronize(s[i]));
  }

  for (int i = 0; i < nDev; ++i) {
      device_set(i, nDev);
      HIPCHECK(hipMemcpy(h_recvbuff[i], recvbuff[i], recvsize*sizeof(float), hipMemcpyDefault ));
      bool res = check_recvbuf(h_recvbuff[i], recvsize, nDev);
      printf("Checking buffer %d result is %s\n",i, res == true ? "correct" : "wrong" );
  }

  //free buffers
  for (int i = 0; i < nDev; ++i) {
      device_set(i, nDev);
      HIPCHECK(hipFree(sendbuff[i]));
      free (h_sendbuff[i]);
      HIPCHECK(hipFree(recvbuff[i]));
      free (h_recvbuff[i]);
  }

  //finalizing RCCL
  for(int i = 0; i < nDev; ++i) {
      ncclCommDestroy(comms[i]);
      HIPCHECK(hipStreamDestroy(s[i]));
  }

  free (h_sendbuff);
  free (sendbuff);
  free (h_recvbuff);
  free (recvbuff);
  free (s);
  free (comms);

  printf("Success \n");
  return 0;
}
