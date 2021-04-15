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

#ifndef NCCL_SHM_OBJECT_H_
#define NCCL_SHM_OBJECT_H_

#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <type_traits>
#include <semaphore.h>

#include "MsgQueue.h"
#include "nccl.h"
#include "core.h"
#include "shm.h"

// ShmObject abstracts away the nitty-gritty when multiple processes need to handle opening a shared
// memory object at the same time.

static ncclResult_t shmSetupExclusive(const char* shmname, const int shmsize, int* fd, void** ptr, int create) {
  *fd = shm_open(shmname, O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR);
  if (*fd == -1) return ncclSystemError;
  if (create) SYSCHECK(shm_allocate(*fd, shmsize), "posix_fallocate");
  SYSCHECK(shm_map(*fd, shmsize, ptr), "mmap");
  close(*fd);
  *fd = -1;
  if (create) memset(*ptr, 0, shmsize);
  return ncclSuccess;
}

template <typename T>
class ShmObject
{
public:
ShmObject(size_t size, std::string fileName, int rank, int numRanks, int projid) :
  m_shmSize(size),
    m_shmName(fileName),
    m_rank(rank),
    m_numRanks(numRanks),
    m_projid(projid),
    m_alloc(false),
    m_shmPtr(nullptr) {}

  ShmObject() {}

  ~ShmObject() {}

  ncclResult_t Open();

  ncclResult_t Close()
  {
    if (m_alloc)
    {
      if (m_rank == 0)
      {
        std::string tmpFileName = "/tmp/" + m_shmName;
        remove(tmpFileName.c_str());
      }
      int retVal = shm_unlink(m_shmName.c_str());
      if (retVal == -1 && errno != ENOENT)
      {
        WARN("Call to shm_unlink in ShmObject failed : %s", strerror(errno));
        return ncclSystemError;
      }
    }
    return ncclSuccess;
  }

  T*& Get()
  {
    return m_shmPtr;
  }
protected:
  ncclResult_t BroadcastMessage(mqd_t& mq_desc, bool pass)
  {
    char msg_text[1];
    msg_text[0] = (pass == 0 ? 'F': 'P');
    for (int rank = 0; rank < m_numRanks; rank++)
    {
      INFO(NCCL_INIT, "Root broadcasting to %d", rank);
      if (rank == m_rank) continue;
      NCCLCHECK(MsgQueueSend(mq_desc, &msg_text[0], sizeof(msg_text)));
    }
    return ncclSuccess;
  }

  // tag for dispatch
      template<class U>
        struct OpenTag{};

      ncclResult_t InitIfSemaphore(OpenTag<int> tag);
      ncclResult_t InitIfSemaphore(OpenTag<uint32_t> tag);
      ncclResult_t InitIfSemaphore(OpenTag<hipIpcMemHandle_t> tag);
      ncclResult_t InitIfSemaphore(OpenTag<sem_t> tag);
      ncclResult_t InitIfSemaphore(OpenTag<std::pair<hipIpcMemHandle_t,size_t>> tag);

      size_t      m_shmSize;
      std::string m_shmName;
      int         m_rank;
      int         m_numRanks;
      int         m_projid;
      bool        m_alloc;
      T*          m_shmPtr;
};

template <typename T>
ncclResult_t ShmObject<T>::Open()
{
  if (m_alloc == false)
  {
    int shmFd;
    int protection = PROT_READ | PROT_WRITE;
    int visibility = MAP_SHARED;

    INFO(NCCL_INIT, "Rank %d in Open for %s\n", m_rank, m_shmName.c_str());
    mqd_t mq_desc;
    NCCLCHECK(MsgQueueGetId(m_shmName, false, mq_desc));
    INFO(NCCL_INIT, "Rank %d got msgqueue id for %s\n", m_rank, m_shmName.c_str());
    if (m_rank == 0)
    {
      ncclResult_t resultSetup = shmSetupExclusive(m_shmName.c_str(), m_shmSize, &shmFd, (void**)&m_shmPtr, 1);
      ncclResult_t resultSemInit = InitIfSemaphore(OpenTag<T>{});
      if ((resultSetup != ncclSuccess && errno != EEXIST) || (resultSemInit != ncclSuccess))
      {
        NCCLCHECK(BroadcastMessage(mq_desc, false));
        WARN("Call to ShmObject::Open in root rank failed : %s", strerror(errno));
        return ncclSystemError;
      }
      NCCLCHECK(BroadcastMessage(mq_desc, true));

      mq_attr attr;
      mq_getattr (mq_desc, &attr);
      INFO(NCCL_INIT, "Number of messages in queue %s: %ld\n", m_shmName.c_str(), attr.mq_curmsgs);
      while (attr.mq_curmsgs > 0)
      {
        mq_getattr (mq_desc, &attr);
      }
      NCCLCHECK(MsgQueueClose(m_shmName));
    }
    else
    {
      mq_attr attr;
      mq_getattr (mq_desc, &attr);
      INFO(NCCL_INIT, "Number of messages in queue %s: %ld\n", m_shmName.c_str(), attr.mq_curmsgs);
      char msg_text[1];
      NCCLCHECK(MsgQueueRecv(mq_desc, &msg_text[0], sizeof(msg_text)));
      if (msg_text[0] == 'P')
      {
        INFO(NCCL_INIT, "Rank %d got message for %s\n", m_rank, m_shmName.c_str());
        NCCLCHECK(shmSetup(m_shmName.c_str(), m_shmSize, &shmFd, (void**)&m_shmPtr, 0));
        INFO(NCCL_INIT, "Rank %d finished shmSetup\n", m_rank);
      }
      else
      {
        WARN("Call to shm_open from non-root rank in ShmObject failed : %s", strerror(errno));
        return ncclSystemError;
      }
    }
    m_alloc = true;
  }
  else
  {
    WARN("Cannot allocate ShmObject twice.\n");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

template<typename T>
ncclResult_t ShmObject<T>::InitIfSemaphore(OpenTag<int> tag)
{
  return ncclSuccess;
}

template<typename T>
ncclResult_t ShmObject<T>::InitIfSemaphore(OpenTag<unsigned int> tag)
{
  return ncclSuccess;
}

template<typename T>
ncclResult_t ShmObject<T>::InitIfSemaphore(OpenTag<hipIpcMemHandle_t> tag)
{
  return ncclSuccess;
}

template<typename T>
ncclResult_t ShmObject<T>::InitIfSemaphore(OpenTag<std::pair<hipIpcMemHandle_t,size_t>> tag)
{
  return ncclSuccess;
}

template<typename T>
ncclResult_t ShmObject<T>::InitIfSemaphore(OpenTag<sem_t> tag)
{
  size_t numMutexes = m_shmSize / sizeof(sem_t);

  for (size_t i = 0; i < numMutexes; i++)
  {
    SYSCHECK(sem_init(static_cast<sem_t*>(&m_shmPtr[i]), 1, 1), "sem_init");
  }
  return ncclSuccess;
}
#endif
