#include "MsgQueue.h"

#include <sys/ipc.h>
#include <sys/msg.h>

#define MSG_QUEUE_PERM 0666

ncclResult_t MsgQueueGetId(std::string name, int projid, bool exclusive, int& msgid)
{
  key_t key;
  SYSCHECKVAL(ftok(name.c_str(), projid), "ftok", key);
  int flag = (exclusive == true ? IPC_CREAT | IPC_EXCL : IPC_CREAT);

  msgid = msgget(key, MSG_QUEUE_PERM | flag);
  // Check if we're trying to create message queue and it already exists; if so, delete existing queue
  if (msgid == -1 && exclusive == true && errno == EEXIST)
  {
    NCCLCHECK(MsgQueueClose(name, projid));
    SYSCHECKVAL(msgget(key, MSG_QUEUE_PERM | flag), "msgget", msgid);
  }
  else if (msgid == -1)
  {
    WARN("Call to MsgQueueGetId failed : %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t MsgQueueSend(int msgid, const void* msgp, size_t msgsz, int msgflg)
{
  SYSCHECK(msgsnd(msgid, msgp, msgsz, msgflg), "msgsnd");
  return ncclSuccess;
}

ncclResult_t MsgQueueRecv(int msgid, void* msgp, size_t msgsz, long msgtyp, bool wait)
{
  int msgflg = (wait == false ? IPC_NOWAIT : 0);
  SYSCHECK(msgrcv(msgid, msgp, msgsz, msgtyp, msgflg), "msgrcv");
  return ncclSuccess;
}

ncclResult_t MsgQueueClose(std::string name, int projid)
{
  key_t key;
  int msgid;
  key = ftok(name.c_str(), projid);
  SYSCHECKVAL(msgget(key, IPC_CREAT), "msgget", msgid);
  SYSCHECK(msgctl(msgid, IPC_RMID, NULL), "msgctl");
  return ncclSuccess;
}
