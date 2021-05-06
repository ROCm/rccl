/*************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TEST_ALLREDUCEGROUP_MULTI_PROCESS_HPP
#define TEST_ALLREDUCEGROUP_MULTI_PROCESS_HPP

#include "CorrectnessTest.hpp"
#include "test_AllReduceMultiProcess.hpp"
#include <string>

namespace CorrectnessTests
{
    class AllReduceGroupMultiProcessCorrectnessTest : public MultiProcessCorrectnessTest
    {
    public:
        void TestGroupCalls(int process, std::vector<int> const& ranks, std::vector<Dataset*>& datasets, std::vector<ncclFunc_t> const& funcs, bool& pass)
        {
            ncclGroupStart();
            for (int i = 0; i < ranks.size(); i++)
            {
                SetUpPerProcess(ranks[i], funcs, comms[ranks[i]], streams[ranks[i]], datasets);
                if (numDevices > numDevicesAvailable)
                {
                    break;
                }
            }
            ncclGroupEnd();

            if (numDevices > numDevicesAvailable)
            {
                pass = true;
                return;
            }

            int numProcesses = numDevices / ranks.size();
            Barrier barrier(process, numProcesses, std::atoi(getenv("NCCL_COMM_ID")));

            for (int i = 0; i < ranks.size(); i++)
            {
                for (int j = 0; j < datasets.size(); j++)
                {
                    FillDatasetWithPattern(*datasets[j], ranks[i]);
                }
            }

            int const root = 0;

            for (int i = 0; i < 3; i++)
            {
              AllReduceMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[i], barrier, op, ranks);
            }
            barrier.Wait();

            size_t const byteCount = datasets[0]->NumBytes() / numDevices;
            size_t const elemCount = numElements / numDevices;

            ncclGroupStart();
            // AllReduce
            for (int i = 0; i < ranks.size(); i++)
            {
                int rank = ranks[i];
                for (int j = 0; j < 3; j++)
                {
                  ncclAllReduce(datasets[j]->inputs[rank], datasets[j]->outputs[rank],
                                numElements, dataType, op, comms[rank], streams[rank]);
                }
            }
            // Signal end of group call
            ncclGroupEnd();

            for (int i = 0; i < ranks.size(); i++)
            {
                HIP_CALL(hipSetDevice(ranks[i]));
                HIP_CALL(hipStreamSynchronize(streams[ranks[i]]));
            }

            for (int i = 0; i < funcs.size(); i++)
            {
                for (int j = 0; j < ranks.size(); j++)
                {
                    pass = ValidateResults(*datasets[i], ranks[j], root);
                    if (!pass)
                    {
                        break;
                    }
                }
                barrier.Wait();
                for (int j = 0; j < ranks.size(); j++)
                {
                    datasets[i]->Release(ranks[j]);
                }
            }

            for (int i = 0; i < ranks.size(); i++)
            {
                TearDownPerProcess(comms[ranks[i]], streams[ranks[i]]);
            }
        }
    };
}

#endif
