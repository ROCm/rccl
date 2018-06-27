/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

//
// This header provides device functions for reduce ops
// which are used in rcclAllReduce and rcclReduce implementations
//

template<typename DataType_t, typename VectorType_t>
inline __device__ void OpMax(VectorType_t& out, VectorType_t& in1, VectorType_t& in2) {
    const int num = sizeof(VectorType_t)/sizeof(DataType_t);
    for(int i = 0; i < num; i++) {
        out[i] = in1[i] > in2[i] ? in1[i] : in2[i];
    }
}

template<typename DataType_t, typename VectorType_t>
inline __device__ void OpMin(VectorType_t& out, VectorType_t& in1, VectorType_t& in2) {
    const int num = sizeof(VectorType_t)/sizeof(DataType_t);
    for(int i = 0; i < num; i++) {
        out[i] = in1[i] < in2[i] ? in1[i] : in2[i];
    }
}
