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

void call_RCCL(ncclUniqueId id, int collID, int rank, int nranks, std::vector<int>& send, std::vector<int>& recv, bool managed){
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

    if(!managed){
      NCCLCHECK(ncclMemAlloc((void **)&sendbuff, sendSize * sizeof(int)));
      NCCLCHECK(ncclMemAlloc((void **)&recvbuff, recvSize * sizeof(int)));
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

    NCCLCHECK(ncclMemFree(sendbuff));
    NCCLCHECK(ncclMemFree(recvbuff));
    ncclCommDestroy(comm);
  }

}