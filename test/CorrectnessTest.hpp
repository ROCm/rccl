/*************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef CORRECTNESSTEST_HPP
#define CORRECTNESSTEST_HPP

#include <cstdio>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include "rccl.h"
#include "../include/rccl_bfloat16.h"

#define HIP_CALL(x) ASSERT_EQ(x, hipSuccess)
#define NCCL_CALL(x) ASSERT_EQ(x, ncclSuccess)

namespace CorrectnessTests
{
    // Performs the various basic reduction operations
    template <typename T>
    T ReduceOp(ncclRedOp_t const op, T const A, T const B)
    {
        switch (op)
        {
        case ncclSum:  return A + B;
        case ncclProd: return A * B;
        case ncclMax:  return std::max(A, B);
        case ncclMin:  return std::min(A, B);
        default:
            fprintf(stderr, "[ERROR] Unsupported reduction operator (%d)\n", op);
            exit(0);
        }
    }

    // Returns the number of bytes per element for each supported datatype
    static int DataTypeToBytes(ncclDataType_t const dataType)
    {
        switch (dataType)
        {
        case ncclInt8:   return 1;
        case ncclUint8:  return 1;
        case ncclInt32:  return 4;
        case ncclUint32: return 4;
        case ncclInt64:  return 8;
        case ncclUint64: return 8;
        case ncclFloat16: return 2;
        case ncclFloat32: return 4;
        case ncclFloat64: return 8;
        case ncclBfloat16: return 2;
        default:
            fprintf(stderr, "[ERROR] Unsupported datatype (%d)\n", dataType);
            exit(0);
        }
    }

    // Encapsulates all the memory used per devices for collectives, as well as reference results
    struct Dataset
    {
        int                 numDevices;  // Number of devices participating
        size_t              numElements; // Number of elements per array
        ncclDataType_t      dataType;    // Data type of each input/output pointer
        bool                inPlace;     // Whether or not output pointers are same as input pointers

        std::vector<void *> inputs;      // Input pointers (1 per device)
        std::vector<void *> outputs;     // Output pointers (1 per device)
                                         // May be identical to input pointers for in-place tests
        std::vector<void *> expected;    // Expected output (1 per device)

        size_t NumBytes() const
        {
            return numElements * DataTypeToBytes(dataType);
        }

        void Initialize(int            const numDevices_,
                        size_t         const numElements_,
                        ncclDataType_t const dataType_,
                        bool           const inPlace_)
        {
            numDevices  = numDevices_;
            numElements = numElements_;
            dataType    = dataType_;
            inPlace     = inPlace_;

            inputs.resize(numDevices);
            outputs.resize(numDevices);
            expected.resize(numDevices);

            // Allocate per-device memory
            size_t const numBytes = NumBytes();

            for (int i = 0; i < numDevices; i++)
            {
                HIP_CALL(hipSetDevice(i));
                HIP_CALL(hipMalloc((void **)&inputs[i], numBytes));
                if (inPlace)
                    outputs[i] = inputs[i];
                else
                    HIP_CALL(hipMalloc((void **)&outputs[i], numBytes));

                expected[i] = malloc(numBytes);
            }
        }

        // Explicit memory release to avoid double-free from subDatasets
        void Release()
        {
            for (int i = 0; i < outputs.size(); i++)
            {
                if (!inPlace) hipFree(outputs[i]);
                hipFree(inputs[i]);
                free(expected[i]);
            }

            outputs.clear();
        }

        // Creates a dataset by pointing to an existing dataset
        // Primarily to allow for testing with different starting byte-alignments
        void ExtractSubDataset(size_t const startElement,
                               size_t const lastElement,
                               Dataset& subDataset)
        {
            ASSERT_LE(startElement, lastElement);
            ASSERT_LT(lastElement, numElements);

            subDataset.numDevices  = numDevices;
            subDataset.numElements = lastElement - startElement + 1;
            subDataset.dataType    = dataType;
            subDataset.inPlace     = inPlace;

            subDataset.inputs.resize(numDevices);
            subDataset.outputs.resize(numDevices);
            subDataset.expected.resize(numDevices);

            size_t const byteOffset = (startElement * DataTypeToBytes(dataType));
            for (int i = 0; i < numDevices; i++)
            {
                subDataset.inputs[i]   = (int8_t *)inputs[i] + byteOffset;
                subDataset.outputs[i]  = (int8_t *)outputs[i] + byteOffset;
                subDataset.expected[i] = (int8_t *)expected[i] + byteOffset;
            }
        }
    };

    typedef std::tuple<ncclRedOp_t    /* op          */,
                       ncclDataType_t /* dataType    */,
                       size_t         /* numElements */,
                       int            /* numDevices  */,
                       bool           /* inPlace     */> TestTuple;

    // Base class for each collective test
    // - Each test is instantiated with a different TestTuple
    class CorrectnessTest : public testing::TestWithParam<TestTuple>
    {
    protected:

        // This code is called per test-tuple
        void SetUp() override
        {
            // Check for fine-grained env variable (otherwise will hang)
            if (!getenv("HSA_FORCE_FINE_GRAIN_PCIE"))
            {
                printf("Must set HSA_FORCE_FINE_GRAIN_PCIE=1 prior to execution\n");
                exit(0);
            }

            // Make the test tuple parameters accessible
            std::tie(op, dataType, numElements, numDevices, inPlace) = GetParam();

            // Collect the number of available GPUs
            HIP_CALL(hipGetDeviceCount(&numDevicesAvailable));

            // Only proceed with testing if there are enough GPUs
            if (numDevices > numDevicesAvailable)
            {
                fprintf(stdout, "[  SKIPPED ] Test requires %d devices (only %d available)\n",
                        numDevices, numDevicesAvailable);

                // Modify the number of devices so that tear-down doesn't occur
                // This is temporary until GTEST_SKIP() becomes available
                numDevices = 0;
                numDevicesAvailable = -1;
                return;
            }

            // Initialize communicators
            comms.resize(numDevices);
            NCCL_CALL(ncclCommInitAll(comms.data(), numDevices, NULL));

            // Create streams
            streams.resize(numDevices);
            for (int i = 0; i < numDevices; i++)
            {
                HIP_CALL(hipSetDevice(i));
                HIP_CALL(hipStreamCreate(&streams[i]));
            }
        }

        // Clean up per TestTuple
        void TearDown() override
        {
            // Release communicators and streams
            for (int i = 0; i < numDevices; i++)
            {
                NCCL_CALL(ncclCommDestroy(comms[i]));
                HIP_CALL(hipStreamDestroy(streams[i]));
            }
        }

        void FillDatasetWithPattern(Dataset& dataset)
        {
            int8_t*   arrayI1 = (int8_t   *)malloc(dataset.NumBytes());
            uint8_t*  arrayU1 = (uint8_t  *)arrayI1;
            int32_t*  arrayI4 = (int32_t  *)arrayI1;
            uint32_t* arrayU4 = (uint32_t *)arrayI1;
            int64_t*  arrayI8 = (int64_t  *)arrayI1;
            uint64_t* arrayU8 = (uint64_t *)arrayI1;
            float*    arrayF4 = (float    *)arrayI1;
            double*   arrayF8 = (double   *)arrayI1;
            rccl_bfloat16* arrayB2 = (rccl_bfloat16 *)arrayI1;

            // NOTE: Currently half-precision float tests are unsupported due to half being supported
            //       on GPU only and not host

            // Fills input  data[i][j] with (i + j) % 6
            // - Keeping range small to reduce likelihood of overflow
            // - Sticking with floating points values that are perfectly representable
            for (int i = 0; i < dataset.numDevices; i++)
            {
                for (int j = 0; j < dataset.numElements; j++)
                {
                    int    valueI = (i + j) % 6;
                    float  valueF = (float)valueI;

                    switch (dataset.dataType)
                    {
                    case ncclInt8:    arrayI1[j] = valueI; break;
                    case ncclUint8:   arrayU1[j] = valueI; break;
                    case ncclInt32:   arrayI4[j] = valueI; break;
                    case ncclUint32:  arrayU4[j] = valueI; break;
                    case ncclInt64:   arrayI8[j] = valueI; break;
                    case ncclUint64:  arrayU8[j] = valueI; break;
                    case ncclFloat32: arrayF4[j] = valueF; break;
                    case ncclFloat64: arrayF8[j] = valueF; break;
                    case ncclBfloat16: arrayB2[j] = rccl_bfloat16(valueF); break;
                    default:
                        fprintf(stderr, "[ERROR] Unsupported datatype\n");
                        exit(0);
                    }
                }

                HIP_CALL(hipSetDevice(i));
                HIP_CALL(hipMemcpy(dataset.inputs[i], arrayI1, dataset.NumBytes(), hipMemcpyHostToDevice));

                // Fills output data[i][j] with 0 (if not inplace)
                if (!dataset.inPlace)
                    HIP_CALL(hipMemset(dataset.outputs[i], 0, dataset.NumBytes()));
            }

            free(arrayI1);
        }

        void Synchronize() const
        {
            // Wait for reduction to complete
            for (int i = 0; i < numDevices; i++)
            {
                HIP_CALL(hipSetDevice(i));
                HIP_CALL(hipStreamSynchronize(streams[i]));
            }
        }

        void ValidateResults(Dataset const& dataset) const
        {
            int8_t*   outputI1 = (int8_t   *)malloc(dataset.NumBytes());
            uint8_t*  outputU1 = (uint8_t  *)outputI1;
            int32_t*  outputI4 = (int32_t  *)outputI1;
            uint32_t* outputU4 = (uint32_t *)outputI1;
            int64_t*  outputI8 = (int64_t  *)outputI1;
            uint64_t* outputU8 = (uint64_t *)outputI1;
            float*    outputF4 = (float    *)outputI1;
            double*   outputF8 = (double   *)outputI1;
            rccl_bfloat16* outputB2 = (rccl_bfloat16 *)outputI1;

            bool isMatch = true;

            // Loop over each device's output and compare it to the expected output
            // (Each collective operation computes its own expected results)
            for (int i = 0; i < dataset.numDevices && isMatch; i++)
            {
                HIP_CALL(hipMemcpy(outputI1, dataset.outputs[i], dataset.NumBytes(), hipMemcpyDeviceToHost));

                int8_t*   expectedI1 = (int8_t   *)dataset.expected[i];
                uint8_t*  expectedU1 = (uint8_t  *)expectedI1;
                int32_t*  expectedI4 = (int32_t  *)expectedI1;
                uint32_t* expectedU4 = (uint32_t *)expectedI1;
                int64_t*  expectedI8 = (int64_t  *)expectedI1;
                uint64_t* expectedU8 = (uint64_t *)expectedI1;
                float*    expectedF4 = (float    *)expectedI1;
                double*   expectedF8 = (double   *)expectedI1;
                rccl_bfloat16* expectedB2 = (rccl_bfloat16 *)expectedI1;

                for (int j = 0; j < dataset.numElements && isMatch; j++)
                {
                    switch (dataset.dataType)
                    {
                    case ncclInt8:    isMatch &= (outputI1[j] == expectedI1[j]); break;
                    case ncclUint8:   isMatch &= (outputU1[j] == expectedU1[j]); break;
                    case ncclInt32:   isMatch &= (outputI4[j] == expectedI4[j]); break;
                    case ncclUint32:  isMatch &= (outputU4[j] == expectedU4[j]); break;
                    case ncclInt64:   isMatch &= (outputI8[j] == expectedI8[j]); break;
                    case ncclUint64:  isMatch &= (outputU8[j] == expectedU8[j]); break;
                    case ncclFloat32: isMatch &= (outputF4[j] == expectedF4[j]); break;
                    case ncclFloat64: isMatch &= (outputF8[j] == expectedF8[j]); break;
                    case ncclBfloat16: isMatch &= (outputB2[j] == expectedB2[j]); break;
                    default:
                        fprintf(stderr, "[ERROR] Unsupported datatype\n");
                        exit(0);
                    }

                    if (!isMatch)
                    {
                        switch (dataset.dataType)
                        {
                        case ncclInt8:
                            printf("Expected %d.  Output %d on device %d[%d]\n", outputI1[j], expectedI1[j], i, j); break;
                        case ncclUint8:
                            printf("Expected %u.  Output %u on device %d[%d]\n", outputU1[j], expectedU1[j], i, j); break;
                        case ncclInt32:
                            printf("Expected %d.  Output %d on device %d[%d]\n", outputI4[j], expectedI4[j], i, j); break;
                        case ncclUint32:
                            printf("Expected %u.  Output %u on device %d[%d]\n", outputU4[j], expectedU4[j], i, j); break;
                        case ncclInt64:
                            printf("Expected %ld.  Output %ld on device %d[%d]\n", outputI8[j], expectedI8[j], i, j); break;
                        case ncclUint64:
                            printf("Expected %lu.  Output %lu on device %d[%d]\n", outputU8[j], expectedU8[j], i, j); break;
                        case ncclFloat32:
                            printf("Expected %f.  Output %f on device %d[%d]\n", outputF4[j], expectedF4[j], i, j); break;
                        case ncclFloat64:
                            printf("Expected %lf.  Output %lf on device %d[%d]\n", outputF8[j], expectedF8[j], i, j); break;
                        case ncclBfloat16:
                            printf("Expected %f.  Output %f on device %d[%d]\n", (float)outputB2[j], (float)expectedB2[j], i, j); break;
                        default:
                            fprintf(stderr, "[ERROR] Unsupported datatype\n");
                            exit(0);
                        }
                    }
                }
                ASSERT_EQ(isMatch, true);
            }
        }

        // Passed in parameters from TestTuple
        ncclRedOp_t              op;
        ncclDataType_t           dataType;
        size_t                   numElements;
        int                      numDevices;
        bool                     inPlace;

        int                      numDevicesAvailable;
        std::vector<ncclComm_t>  comms;
        std::vector<hipStream_t> streams;
    };

}

#endif
