/*
Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include "ShmObject.h"
#include <string>

// Template specializations for sem_t objects which require additional initialization
template<>
ncclResult_t ShmObject<sem_t>::Close()
{
    size_t numMutexes = m_shmSize / sizeof(sem_t);

    for (size_t i = 0; i < numMutexes; i++)
    {
        sem_destroy(static_cast<sem_t*>(&m_shmPtr[i]));
    }

    int retVal = shm_unlink(m_shmName.c_str());
    if (retVal == -1 && errno != ENOENT)
    {
        WARN("Call to shm_unlink in ShmObject failed : %s", strerror(errno));
        return ncclSystemError;
    }

    return ncclSuccess;
}
