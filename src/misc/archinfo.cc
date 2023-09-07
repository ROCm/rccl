/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "archinfo.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

void gcnArchNameFormat(char* gcnArchName, char* out) {
  // this function parses the char array from the device properties into something easier to handle.
  // as the gcnArchName attribute looks something like: "gfx900:xnack+:blah-:etc-"
  char *gcnArchNameToken = strtok(gcnArchName, ":");
  strcpy(gcnArchNameToken, out);
}

void gcnArchConvertToGcnArchName(int gcnArch, char* gcnArchName) {
  // gcnArch is deprecated and we should instead use gcnArchName; however, some data files still have
  // the older gcnArch value.  There's only a handful of architectures that were coded prior to deprecation,
  // so we handle those cases here.
  //char gcnArchName[256] = {0}; // why 256?  Because that's what gcnArchName gives us, so we're matching it.
  gcnArchName[6] = 0;
  switch (gcnArch) {
    case 906:
      strncpy(gcnArchName, "gfx906", 6);
      break;
    case 908:
      strncpy(gcnArchName, "gfx908", 6);
      break;
    case 910:
      // this is actually 90a
      strncpy(gcnArchName, "gfx90a", 6);
      break;
    default:
      // throw an error here or however we're handling these runtime exceptions.
  }
}

int getGcnArchName(int deviceId, char* out) {
  // this is a generic call in to get a consistent gcnArchName regardless of which version of rocm we're using.
  // or which version of rocm we're using.
  hipDeviceProp_t devProp;
  hipError_t status = hipGetDeviceProperties(&devProp, deviceId);
  if (status != hipSuccess) {
    //std::cerr << "Encountered HIP error getting device properties: "
    //          << hipGetErrorString(status) << "\n";
    exit(-1);
  }
#ifdef HIP_NO_GCNARCHNAME
  // we're using a HIP version before 3.7.
  gcnArchConvertToGcnArchName(devProp.gcnArch, out);
  return 1;
#else
  gcnArchNameFormat(devProp.gcnArchName, out);
  return 0;
#endif
}

double getDeviceWallClockRateInKhz(int deviceId) {
  #ifdef HIP_NO_WALLCLOCKRATE // we're using an older version of ROCM/HIP and must try according to the name instead.
    if (strncmp("gfx94", getGcnArchName(deviceId), 5) == 0)
      return 1.0E5;
    else
      return 2.5E4;
#else
    // So instead of using gcnArch to detect the model, we'll query the frequency directly via HIP.
    hipDeviceProp_t devProp;
    hipError_t status = hipGetDeviceProperties(&devProp, deviceId);
    int gpu_rtc_freq;
    // don't be alarmed; this is the one that gives us the number we actually want, not hipDeviceAttributeClockRate
    hipDeviceGetAttribute(&gpu_rtc_freq, hipDeviceAttributeWallClockRate, deviceId);
    if (gpu_rtc_freq != 0.0) // hey, it worked!
      return (double) gpu_rtc_freq;
    else
      return 1.0E5; // TODO: remove me before merging PR; current workaround for buggy return.
  #endif
}
