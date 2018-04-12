/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include <atomic>
#include <map>
#include "rcclCheck.h"

#define CHUNK_DWORD     (1048572/2)
#define CHUNK_DWORDx4   CHUNK_DWORD/4
#define CHUNK_SIZE      CHUNK_DWORD*sizeof(int)

struct DeviceControl_t {
    struct DeviceControl_t *prevPeer;
    struct DeviceControl_t *nextPeer;
    std::atomic<uint32_t> chunkId;
    std::atomic<void*> srcBuffer;
    std::atomic<void*> dstBuffer;
    void *controlBuffer;
    uint32_t hipCurrentDeviceId;
};

struct RcclComm_t;

class DevTrackerPool_t{
private:
    int *deviceIds;
    int numDevices;
public:
    int activeDevices;
    std::map<size_t, DeviceControl_t*> Pool;
    DevTrackerPool_t() : deviceIds(nullptr), numDevices(0) {}
    ~DevTrackerPool_t() {
        delete deviceIds;
    }

    DevTrackerPool_t(const int *devIds, int numDevices);
    RcclComm_t *AddDevice(int device, int rank, int ndev);
    void PrintAll();
    DeviceControl_t *getPoolByDevID(int devId);
};

struct RcclComm_t {
public:
    DevTrackerPool_t *pool;
    DeviceControl_t *Track;
    hipEvent_t event;
    int numDevices;
    int device;
    int rank;
    ~RcclComm_t() {
        HIPCHECK(hipFree(Track->controlBuffer));
        HIPCHECK(hipEventDestroy(event));
    }
};

    DevTrackerPool_t::DevTrackerPool_t(const int* devIds, int numDevices) : numDevices(numDevices), activeDevices(numDevices) {
        int userDevId;
        HIPCHECK(hipGetDevice(&userDevId));

        deviceIds = new int[numDevices];
        memcpy(deviceIds, devIds, numDevices*sizeof(int));

        struct DeviceControl_t *tmp;

        for(int i=0;i<numDevices;i++){
            HIPCHECK(hipHostMalloc(&tmp, sizeof(DeviceControl_t), hipHostMallocCoherent));
            Pool[i] = tmp;
            Pool[i]->prevPeer = nullptr;
            Pool[i]->nextPeer = nullptr;
            Pool[i]->chunkId = 0;
            Pool[i]->hipCurrentDeviceId = devIds[i];
            Pool[i]->srcBuffer = nullptr;
            Pool[i]->dstBuffer = nullptr;
            Pool[i]->controlBuffer = nullptr;
        }

        struct DeviceControl_t *dptr;
        void *controlBuffer;
        HIPCHECK(hipSetDevice(deviceIds[0]));
        HIPCHECK(hipHostGetDevicePointer((void**)&dptr, Pool[0], 0));
        HIPCHECK(hipMalloc(&controlBuffer, numDevices*CHUNK_SIZE));
        Pool[0]->controlBuffer = controlBuffer;
        if(numDevices != 1) {
            Pool[1]->prevPeer = dptr;
        } else {
            Pool[0]->prevPeer = dptr;
        }
        Pool[numDevices-1]->nextPeer = dptr;

        for(unsigned i=1;i<numDevices;i++) {
            HIPCHECK(hipSetDevice(deviceIds[i]));
            HIPCHECK(hipHostGetDevicePointer((void**)&dptr, Pool[i], 0));
            HIPCHECK(hipMalloc(&controlBuffer, numDevices*CHUNK_SIZE));
            Pool[i]->controlBuffer = controlBuffer;
            Pool[(i+1)%numDevices]->prevPeer = dptr;
            Pool[(i-1)%numDevices]->nextPeer = dptr;
        }

        HIPCHECK(hipSetDevice(userDevId));
    }

    RcclComm_t* DevTrackerPool_t::AddDevice(int device, int rank, int ndev) {
        RcclComm_t* retComm = new RcclComm_t;
        numDevices = ndev;
        retComm->numDevices = ndev;
        retComm->device = device;
        retComm->rank = rank;
        struct DeviceControl_t *dctrl;
        HIPCHECK(hipHostMalloc(&dctrl, sizeof(DeviceControl_t), hipHostMallocCoherent));
        void *controlBuffer;
        HIPCHECK(hipMalloc(&controlBuffer, ndev*CHUNK_SIZE));
        dctrl->controlBuffer = controlBuffer;
        dctrl->srcBuffer = 0;
        dctrl->dstBuffer = 0;
        dctrl->prevPeer = nullptr;
        dctrl->nextPeer = nullptr;
        dctrl->chunkId = 0;
        dctrl->hipCurrentDeviceId = device;

        if(Pool.find(rank) != Pool.end()) {
            // clean existing entry
        } else {
            Pool[rank] = dctrl;
        }

        if(Pool.size() == ndev) {
            Pool[1]->prevPeer = Pool[0];
            Pool[ndev-1]->nextPeer = Pool[0];
            for(int i=1;i<ndev;i++) {
                Pool[(i+1)%ndev]->prevPeer = Pool[i];
                Pool[(i-1)%ndev]->nextPeer = Pool[i];
            }
        }
        retComm->Track = dctrl;
        return retComm;
    }

    void DevTrackerPool_t::PrintAll() {
        for(int i=0;i<numDevices;i++) {
            std::cout<<"On Device: "<<deviceIds[i]<<std::endl;
            std::cout<<Pool[i]->prevPeer<<std::endl;
            std::cout<<Pool[i]->nextPeer<<std::endl;
            std::cout<<Pool[i]->chunkId<<std::endl;
            std::cout<<Pool[i]->dstBuffer<<std::endl;
            std::cout<<Pool[i]->srcBuffer<<std::endl;
        }
    }

    struct DeviceControl_t* DevTrackerPool_t::getPoolByDevID(int devId) {
        for(int i=0;i<numDevices;i++) {
            if(devId == deviceIds[i]) {
                return Pool[i];
            }
        }
        return nullptr;
    }


