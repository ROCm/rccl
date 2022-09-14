/*
* Copyright 2009-2020  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef NVTX_IMPL_GUARD_CUDA
#error Never include this file directly -- it is automatically included by nvToolsExtCuda.h (except when NVTX_NO_IMPL is defined).
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void (NVTX_API * nvtxNameCuDeviceA_impl_fntype)(hipDevice_t device, const char* name);
typedef void (NVTX_API * nvtxNameCuDeviceW_impl_fntype)(hipDevice_t device, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCuContextA_impl_fntype)(hipCtx_t context, const char* name);
typedef void (NVTX_API * nvtxNameCuContextW_impl_fntype)(hipCtx_t context, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCuStreamA_impl_fntype)(hipStream_t stream, const char* name);
typedef void (NVTX_API * nvtxNameCuStreamW_impl_fntype)(hipStream_t stream, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCuEventA_impl_fntype)(hipEvent_t event, const char* name);
typedef void (NVTX_API * nvtxNameCuEventW_impl_fntype)(hipEvent_t event, const wchar_t* name);

NVTX_DECLSPEC void NVTX_API nvtxNameCuDeviceA(hipDevice_t device, const char* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCuDeviceA_impl_fntype local = (nvtxNameCuDeviceA_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCuDeviceA_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCuDeviceW(hipDevice_t device, const wchar_t* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCuDeviceW_impl_fntype local = (nvtxNameCuDeviceW_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCuDeviceW_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCuContextA(hipCtx_t context, const char* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCuContextA_impl_fntype local = (nvtxNameCuContextA_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCuContextA_impl_fnptr;
    if(local!=0)
        (*local)(context, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCuContextW(hipCtx_t context, const wchar_t* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCuContextW_impl_fntype local = (nvtxNameCuContextW_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCuContextW_impl_fnptr;
    if(local!=0)
        (*local)(context, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCuStreamA(hipStream_t stream, const char* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCuStreamA_impl_fntype local = (nvtxNameCuStreamA_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCuStreamA_impl_fnptr;
    if(local!=0)
        (*local)(stream, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCuStreamW(hipStream_t stream, const wchar_t* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCuStreamW_impl_fntype local = (nvtxNameCuStreamW_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCuStreamW_impl_fnptr;
    if(local!=0)
        (*local)(stream, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCuEventA(hipEvent_t event, const char* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCuEventA_impl_fntype local = (nvtxNameCuEventA_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCuEventA_impl_fnptr;
    if(local!=0)
        (*local)(event, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCuEventW(hipEvent_t event, const wchar_t* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCuEventW_impl_fntype local = (nvtxNameCuEventW_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCuEventW_impl_fnptr;
    if(local!=0)
        (*local)(event, name);
#endif /*NVTX_DISABLE*/
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

