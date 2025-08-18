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


namespace rccl{

using RCCLVecU64_2 = uint64_t __attribute__((ext_vector_type(2)));

#if (defined(__gfx942__) || defined(__gfx950))
#define store_bytepack16_global(addr, value) \
  asm ("global_store_dwordx4 %0, %1 off sc0 sc1 nt " :: "v"((rccl::RCCLVecU64_2*)(addr)), "v"((value.u64_vec2)))

#define load_bytepack16_global(addr, ans) \
  ans.u64_vec2 = __builtin_nontemporal_load((rccl::RCCLVecU64_2*)(addr)) 
  
#else
#define store_bytepack16_global(addr, value) \
  __builtin_nontemporal_store((value).u64[0], (uint64_t*)(addr));  \
  __builtin_nontemporal_store((value).u64[1], (uint64_t*)(addr)+1);

#define load_bytepack16_global(addr, ans) \
  ans.u64[0] = __builtin_nontemporal_load((uint64_t*)addr); \
  ans.u64[1] = __builtin_nontemporal_load((uint64_t*)addr+1); 
#endif

}