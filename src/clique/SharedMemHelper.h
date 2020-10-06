#ifndef SHAREDMEMHELPER_H
#define SHAREDMEMHELPER_H


class SharedMemHelper
{
public:
  SharedMemHelper(int rank, int numRanks, int numEntries);

  ncclStatus_t Init(std::string const& baseFilename);

  ncclStatus_t


protected:
  bool m_initialized;
  int m_rank;
  int m_numRanks;
};

#endif
