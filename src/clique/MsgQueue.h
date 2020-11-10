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

#ifndef RCCL_MSG_QUEUE_HPP_
#define RCCL_MSG_QUEUE_HPP_

#include <string>

#include "nccl.h"
#include "core.h"

struct MsgBuffer
{
  long msg_type;
  char msg_text[1];
};

ncclResult_t MsgQueueGetId(std::string name, int projid, bool exclusive, int& msgid);
ncclResult_t MsgQueueSend(int msgid, const void* msgp, size_t msgsz, int msgflg);
ncclResult_t MsgQueueRecv(int msgid, void* msgp, size_t msgsz, long msgtyp, bool wait);
ncclResult_t MsgQueueClose(std::string name, int projid);

#endif
