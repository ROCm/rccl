/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

// This is only okay when the protocol buffer is allocated in uncached memory.
#if defined(__gfx942__) && defined(HIP_UNCACHED_MEMORY) && !defined(DISABLE_CHEAP_THREADFENCE)
#define RCCL_CHEAP_THREADFENCE_OK_SOMETIMES 1
#else
#define RCCL_CHEAP_THREADFENCE_OK_SOMETIMES 0
#endif 

template<bool UseCheaperThreadFence>
inline __device__ void gfx9ThreadFence();

template<>
inline __device__ void gfx9ThreadFence<true>() {
    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
    asm volatile("buffer_inv sc0 sc1");
}

template<>
inline __device__ void gfx9ThreadFence<false>() {
    __threadfence();
}
