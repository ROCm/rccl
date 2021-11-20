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

#include "MsgQueue.h"
#include <chrono>

#define MSG_QUEUE_PERM S_IRUSR | S_IWUSR
#define MSG_QUEUE_MODE O_RDWR
#define MSG_SIZE 1
#define MSG_QUEUE_TIMEOUT 60

ncclResult_t MsgQueueGetId(std::string const& name, bool exclusive, mqd_t& mq_desc)
{
  int flag = (exclusive == true ? O_CREAT | O_EXCL : O_CREAT);
  struct mq_attr attr;
  attr.mq_maxmsg = 10;
  attr.mq_msgsize = MSG_SIZE;
  attr.mq_flags = 0;

  std::string mq_name = "/" + name;
  mq_desc = mq_open(mq_name.c_str(), flag | MSG_QUEUE_MODE, MSG_QUEUE_PERM, &attr);

  // Check if we're trying to create message queue and it already exists; if so, delete existing queue
  if (mq_desc == -1 && exclusive == true && errno == EBUSY)
  {
    NCCLCHECK(MsgQueueClose(name, mq_desc, true));
    SYSCHECKVAL(mq_open(mq_name.c_str(), flag | MSG_QUEUE_MODE, MSG_QUEUE_PERM, attr), "mq_open", mq_desc);
  }
  else if (mq_desc == -1)
  {
    WARN("Call to MsgQueueGetId failed : %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t MsgQueueSend(mqd_t const& mq_desc, const char* msgp, size_t msgsz)
{
  SYSCHECK(mq_send(mq_desc, msgp, msgsz, 0), "mq_send");
  return ncclSuccess;
}

ncclResult_t MsgQueueRecv(mqd_t const& mq_desc, char* msgp, size_t msgsz)
{
  SYSCHECK(mq_receive(mq_desc, msgp, msgsz, NULL), "mq_receive");
  return ncclSuccess;
}

ncclResult_t MsgQueueWaitUntilEmpty(mqd_t const& mq_desc)
{
  mq_attr attr;
  mq_getattr(mq_desc, &attr);

  auto start = std::chrono::steady_clock::now();
  while(attr.mq_curmsgs > 0)
  {
    SYSCHECK(mq_getattr(mq_desc, &attr), "mq_getattr");
    if(std::chrono::steady_clock::now() - start > std::chrono::seconds(MSG_QUEUE_TIMEOUT))
    {
      WARN("Message Queue timed out waiting for all ranks to receive messages.");
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

ncclResult_t MsgQueueClose(std::string const& name, mqd_t& mq_desc, bool unlink)
{
  if (unlink)
  {
    NCCLCHECK(MsgQueueUnlink(name));
  }
  SYSCHECK(mq_close(mq_desc), "mq_close");
  return ncclSuccess;
}

ncclResult_t MsgQueueUnlink(std::string const& name)
{
  std::string mq_name = "/" + name;
  SYSCHECK(mq_unlink(mq_name.c_str()), "mq_unlink");
  return ncclSuccess;
}
