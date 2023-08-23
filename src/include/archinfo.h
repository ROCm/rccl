#ifndef ARCHINFO_H_
#define ARCHINFO_H_

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

void gcnArchNameFormat(char *gcnArchName, char* out);
void gcnArchConvertToGcnArchName(int gcnArch, char* out);
int getGcnArchName(int deviceId, char* out);
double getDeviceWallClockRateInKhz(int deviceId);

#endif // ARCHINFO_H
