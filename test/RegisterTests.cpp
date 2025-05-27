#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "register.h"
#include "StandaloneUtils.hpp"
  
namespace RcclUnitTesting  
{ 
  TEST(Register, GraphRegisterDeregister)
  {
    // Check for multi-gpu
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 1) {
      GTEST_SKIP() << "This test requires at least 1 device.";
    }

    // Initialize the comms
    std::vector<ncclComm_t> comms(numDevices);
    NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

    // Create buffers on each device
    const size_t bufferSize = 1024 * 1024; // 1 MB
    std::vector<void*> deviceBuffers(numDevices);
    for (int i = 0; i < numDevices; i++) {
      HIPCALL(hipSetDevice(i));
      HIPCALL(hipMalloc(&deviceBuffers[i], bufferSize));
    }

    // Register buffers with ncclGraphRegister
    std::vector<void *> regRecords(numDevices);
    for (int i = 0; i < numDevices; i++) {
      HIPCALL(hipSetDevice(i));
      NCCLCHECK(ncclCommGraphRegister(comms[i], deviceBuffers[i], bufferSize, &regRecords[i]));
      // Verify registration succeeded
      ASSERT_NE(regRecords[i], nullptr);
    }
    
    // Test that we can deregister
    for (int i = 0; i < numDevices; i++) { 
      HIPCALL(hipSetDevice(i));
      NCCLCHECK(ncclCommGraphDeregister(comms[i], (struct ncclReg *)regRecords[i]));
    }

    // Clean up resources
    for (int i = 0; i < numDevices; i++) {
      HIPCALL(hipSetDevice(i));
      HIPCALL(hipFree(deviceBuffers[i]));
    }
  }
}