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

#include "rccl_libpath_info.h"
#include <stdlib.h>
#include <dlfcn.h>
#include <link.h>
#include <string.h>

void getRcclLibPath(char *libPath, int pathLen) {
    const char *rcclLib = "librccl.so";
    //Open lib handle
    void *dlHandle = dlopen(rcclLib, RTLD_LAZY);
    //Return Unknown if opening handle was not successful
    if (!dlHandle) {
        strncpy(libPath, "Unknown", pathLen);
        return;
    }
    //Retrive lib info
    struct link_map *dlLinkMap;
    if (dlinfo(dlHandle, RTLD_DI_LINKMAP, &dlLinkMap) != 0) {
        strncpy(libPath, "Unknown", pathLen);
        return;
    }
    //Retrieval successful. Copy to libPath
    strncpy(libPath, dlLinkMap->l_name, pathLen - 1);
    libPath[pathLen - 1] = '\0';
    //Close handle
    dlclose(dlHandle);
}