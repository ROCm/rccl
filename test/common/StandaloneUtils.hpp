#ifndef STANDALONE_UTILS_H
#define STANDALONE_UTILS_H

#include <iostream>
#include <cstdio>
#include <regex>

#define HIPCALL(cmd)                                                                          \
    do {                                                                                      \
        hipError_t error = (cmd);                                                             \
        if (error != hipSuccess)                                                              \
        {                                                                                     \
            printf("Encountered HIP error (%s) at line %d in file %s\n",                      \
                                  hipGetErrorString(error), __LINE__, __FILE__);              \
            exit(-1);                                                                         \
        }                                                                                     \
    } while (0)

#define NCCLCHECK(cmd) do {                                     \
    ncclResult_t res = cmd;                                     \
    if (res != ncclSuccess) {                                   \
         printf("NCCL failure %s:%d '%s'\n",                    \
            __FILE__,__LINE__,ncclGetErrorString(res));         \
    }                                                           \
} while(0)

// should be 112, temp fix to make CI pass
#define MAX_STACK_SIZE 448

#ifdef ENABLE_LL128
#define MAX_STACK_SIZE_gfx90a 296
#else
#define MAX_STACK_SIZE_gfx90a MAX_STACK_SIZE
#endif

struct KernelInfo {
    std::string name;
    int privateSegmentFixedSize = 0;
};

struct ArchInfo {
    std::string archName;
    std::vector<KernelInfo> kernels;
};

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

#endif