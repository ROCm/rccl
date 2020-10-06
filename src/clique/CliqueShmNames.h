#ifndef NCCL_CLIQUE_SHM_NAMES_H_
#define NCCL_CLIQUE_SHM_NAMES_H_

#include <string>
#include <map>

static std::map<std::string, std::string> CliqueShmNames =
{
    {"SharedCounters", "RcclCounters"  },
    {"Mutexes"       , "RcclMutexes"   },
    {"IpcHandles"    , "RcclIpcHandles"},
    {"Barriers"      , "RcclBarriers"  }
};

#endif
