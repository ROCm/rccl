/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <unistd.h>
#include "CollectiveArgs.hpp"
#include "StandaloneUtils.hpp"
#include <rccl/rccl.h>


std::string executeCommand(const char* cmd) {
    std::string result;
    FILE* pipe = popen(cmd, "r");

    if (!pipe) {
        std::cerr << "Error executing command: " << cmd << std::endl;
        return result;
    }

    char buffer[128];
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL) {
            result += buffer;
        }
    }

    pclose(pipe);
    return result;
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::istringstream iss(str);

    std::string line;
    while(std::getline(iss, line, delimiter)) {
        result.push_back(line);
    }

    return result;
}


ArchInfo parseMetadata(const std::vector<std::string>& list) {
    ArchInfo archInfo;
    KernelInfo currKernelInfo;
    
    std::regex amdhsaTargetRegex("amdhsa.target:\\s+(?:'?)amdgcn-amd-amdhsa--(\\w+)(?:'?)");
    std::regex kernelNameRegex("\\.name:\\s+(\\w+)");
    std::regex privateSegmentSizeRegex("\\.private_segment_fixed_size:\\s+(\\d+)");
    
    for (const auto& line : list) {
        std::smatch match;

        if (std::regex_search(line, match, amdhsaTargetRegex)) {
            archInfo.archName = match[1];
        } else if (std::regex_search(line, match, kernelNameRegex)) {
            currKernelInfo.name = match[1];
        } else if (std::regex_search(line, match, privateSegmentSizeRegex)) {
            currKernelInfo.privateSegmentFixedSize = std::stoi(match[1]);
        }
        
        if (!currKernelInfo.name.empty() && currKernelInfo.privateSegmentFixedSize != 0) {
            archInfo.kernels.push_back(currKernelInfo);
            currKernelInfo = {}; // Empty kernelInfo
        }
    }
    
    return archInfo;
}

namespace RcclUnitTesting
{

void fork_and_launch_rccl(int nranks,  int collID, const std::vector<int>& sendBuff, std::vector<int>& recvBuff, const std::vector<int>& expected, bool use_managed_mem){
    std::vector<pid_t> children(nranks, 0);
    std::vector<std::vector<int>> childPipes(nranks, std::vector<int>(2,0));
    ncclUniqueId id;

    for(int r = 0; r < nranks; ++r){
      if(pipe(childPipes[r].data()) == -1)
        ERROR("child %i pipe Failed\n", r);
    } 

    auto createNCCLid = [&](int rank){
        ncclGetUniqueId(&id);
        close(childPipes[rank][0]);
        write(childPipes[rank][1], &id, sizeof(ncclUniqueId));
        close(childPipes[rank][1]);
    };

    auto getNCCLidFromParent = [&](int rank){
      close(childPipes[rank][1]); //close write to child0
      read(childPipes[rank][0], &id, sizeof(ncclUniqueId));
      close(childPipes[rank][0]);
    };

    auto getAndDistributeNCCLid = [&](int nranks){
      close(childPipes[0][1]); //close write to child0
      read(childPipes[0][0], &id, sizeof(ncclUniqueId)); //read from child0
      for(int r = 1; r < nranks; ++r){
        write(childPipes[r][1], &id, sizeof(ncclUniqueId));
        close(childPipes[r][1]);
      }
    };

    for(int r = 0; r < nranks; ++r){
      children[r] = fork();
      if(children[r] == 0){
        int ngpus = 0;
        HIPCALL(hipGetDeviceCount(&ngpus));
        if(ngpus != nranks){
          exit(0);
        }
        //child processes
        if(r == 0)
          createNCCLid(r);
        else
          getNCCLidFromParent(r);

        call_rccl(id, collID, r, nranks, sendBuff, recvBuff, use_managed_mem);
        for(int i = 0; i < recvBuff.size(); ++i){
          ASSERT_EQ(recvBuff[i], expected[i]);
        }
        exit(0);
      }
    }

    getAndDistributeNCCLid(nranks);
    
    for(int r = 0; r < nranks; ++r)
      wait(NULL); // Wait for all children
}

void call_rccl(ncclUniqueId id, int collID, int rank, int nranks, const std::vector<int>& send, std::vector<int>& recv, bool use_managed_mem){
    switch(collID){
        case ncclCollAllReduce:
        break;

        case ncclCollAllGather:
        break;
        default:
            ERROR("This collective is not implemented for call_RCCL routine");
    }
    
    HIPCALL(hipSetDevice(rank));
    hipStream_t stream;
    HIPCALL(hipStreamCreate(&stream));
    ncclComm_t comm;
    
    

    NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
    int *sendbuff;
    int *recvbuff;
    void *sendRegHandle;
    void *recvRegHandle;
    

    
    size_t sendSize = 0;
    size_t recvSize = 0;

     switch(collID){
      case ncclCollAllReduce:
        sendSize = send.size();
        recvSize = recv.size();
        break;
      case ncclCollAllGather:
        sendSize = send.size();
        recvSize = nranks*send.size();
        break;
      default: exit(0);
    }

    if(!use_managed_mem){
      HIPCALL(hipMalloc((void **)&sendbuff, sendSize * sizeof(int)));
      HIPCALL(hipMalloc((void **)&recvbuff, recvSize * sizeof(int)));
    }
    else{
      HIPCALL(hipMallocManaged((void **)&sendbuff, sendSize * sizeof(int)));
      HIPCALL(hipMallocManaged((void **)&recvbuff, recvSize * sizeof(int)));
    }    
   
    NCCLCHECK(ncclCommRegister(comm, sendbuff, sendSize * sizeof(int), &sendRegHandle));
    NCCLCHECK(ncclCommRegister(comm, recvbuff, recvSize * sizeof(int), &recvRegHandle));

    HIPCALL(hipMemcpy(sendbuff, send.data(), sizeof(int) * sendSize, hipMemcpyHostToDevice));
    HIPCALL(hipMemcpy(recvbuff, recv.data(), sizeof(int) *recvSize, hipMemcpyHostToDevice));

    switch(collID){
      case ncclCollAllReduce:
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, sendSize, ncclInt, ncclSum, comm, stream));
        break;
      case ncclCollAllGather:
        NCCLCHECK(ncclAllGather(sendbuff, recvbuff, sendSize, ncclInt, comm, stream));
        break;
      default: exit(0);
    }

    HIPCALL(hipStreamSynchronize(stream));
    HIPCALL(hipMemcpy(recv.data(), recvbuff, sizeof(int) * recvSize, hipMemcpyDeviceToHost));
    
    NCCLCHECK(ncclCommDeregister(comm, sendRegHandle));
    NCCLCHECK(ncclCommDeregister(comm, recvRegHandle));

    HIPCALL(hipFree(sendbuff));
    HIPCALL(hipFree(recvbuff));
    ncclCommDestroy(comm);
  }

}