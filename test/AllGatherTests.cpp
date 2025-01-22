/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(AllGather, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1048576, 500};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16, ncclFloat64, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {586};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {104857, 264};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt8, ncclInt64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {958};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1039203, 2500};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {896};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, UserBufferRegistration)
  {          
    setenv("UT_PROCESS_MASK", "2", 1);
    const int nranks = 8;
    std::vector<pid_t> children(nranks);
    std::vector<std::vector<int>> childPipes(nranks, std::vector<int>(2,0));
    std::vector<bool> ResultsCorrect(nranks, false);
    ncclUniqueId id;

    size_t count = 32;
    std::vector<int> sendBuff(count, 0);
    std::vector<int> recvBuff(nranks*count, 0);
    std::vector<int> expected(nranks*count, 0);

    for (int i = 0; i < count; ++i){
        sendBuff[i] = i;
    }

    for(int r = 0; r < nranks; ++r)
      for (int i = 0; i < count; ++i)
        expected[r*count + i] = sendBuff[i];
    
    for(int r = 0; r < nranks; ++r){
      if(pipe(childPipes[r].data()) == -1)
        printf("child %i pipe Failed\n", r);
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
        //child processes
        if(r == 0)
          createNCCLid(r);
        else
          getNCCLidFromParent(r);

        call_RCCL(id, ncclCollAllGather, r, nranks, sendBuff, recvBuff);
        for(int i = 0; i < recvBuff.size(); ++i)
          ASSERT_EQ(recvBuff[i], expected[i]);
        break;
      }
    }

    getAndDistributeNCCLid(nranks);
    for(int r = 0; r < nranks; ++r)
      wait(NULL); // Wait for all children
  }

  TEST(AllGather, ManagedMemUserBufferRegistration)
  {          
    setenv("UT_PROCESS_MASK", "2", 1);
    const int nranks = 8;
    std::vector<pid_t> children(nranks);
    std::vector<std::vector<int>> childPipes(nranks, std::vector<int>(2,0));
    std::vector<bool> ResultsCorrect(nranks, false);
    ncclUniqueId id;

    size_t count = 32;
    std::vector<int> sendBuff(count, 0);
    std::vector<int> recvBuff(nranks*count, 0);
    std::vector<int> expected(count, 0);
    const bool useManagedMem = true;

    for (int i = 0; i < count; ++i){
        sendBuff[i] = i;
    }

    for(int r = 0; r < nranks; ++r)
      for (int i = 0; i < count; ++i)
        expected[r*count + i] = sendBuff[i];

    for(int r = 0; r < nranks; ++r){
      if(pipe(childPipes[r].data()) == -1)
        printf("child %i pipe Failed\n", r);
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
        //child processes
        if(r == 0)
          createNCCLid(r);
        else
          getNCCLidFromParent(r);
        
        call_RCCL(id, ncclCollAllGather, r, nranks, sendBuff, recvBuff, useManagedMem);
        for(int i = 0; i < recvBuff.size(); ++i)
          ASSERT_EQ(recvBuff[i], expected[i]);
        break;
      }
    }

    getAndDistributeNCCLid(nranks);
    for(int r = 0; r < nranks; ++r)
      wait(NULL); // Wait for all children
  }
}
