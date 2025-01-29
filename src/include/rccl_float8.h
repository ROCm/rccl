/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_HIP_FP8_H
#define HIP_INCLUDE_HIP_HIP_FP8_H

#include <hip/hip_common.h>

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
// We only have fnuz defs for now, which are not supported by other platforms
    #if __has_include(<hip/amd_detail/amd_hip_fp8.h>)
        #include <hip/amd_detail/amd_hip_fp8.h>
    // For older versions of ROCm that do not include hip_fp8.h,
    // we provide a local version of the header file as a fallback.
    #else
        #include <amd_hip_fp8.h>
    #endif
#endif



inline std::ostream& operator<<(std::ostream& os, const __hip_fp8_e4m3& f8)
{
    return os << float(f8);
}

inline std::ostream& operator<<(std::ostream& os, const __hip_fp8_e5m2& bf8)
{
    return os << float(bf8);
}

inline __host__ __device__ float operator*(__hip_fp8_e4m3 a, __hip_fp8_e4m3 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(__hip_fp8_e5m2 a, __hip_fp8_e5m2 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(__hip_fp8_e4m3 a, float b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(__hip_fp8_e5m2 a, float b)
{
    return float(a) * float(b);
}

#endif  // HIP_INCLUDE_HIP_HIP_FP8_H
