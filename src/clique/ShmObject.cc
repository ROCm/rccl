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
