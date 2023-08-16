#ifndef ARCHINFO_H_
#define ARCHINFO_H_

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

char *gcnArchNameFormat(char *gcnArchName);
char *gcnArchConvertToGcnArchName(int gcnArch);
char *getGcnArchName(int deviceId);
double getDeviceWallClockRateInKhz(int deviceId);

#endif // ARCHINFO_H
