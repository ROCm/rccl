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