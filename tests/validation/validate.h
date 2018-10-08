/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once
#include <cmath>
#include <iostream>

#define CHECKVAL(got, expected, index)                            \
    std::cerr << "[L: " << __LINE__ << "] Bad Data at: " << index \
              << " Expected: " << expected << " but Got: " << got \
              << std::endl;

template <typename T>
inline bool cmp(T left, T right);

#define EPSF 1e-5f
#define EPS 1e-5
template <>
inline bool cmp(float left, float right) {
    return (fabs(left - right) < EPSF);
}

template <>
inline bool cmp(double left, double right) {
    return (fabs(left - right) < EPS);
}

template <typename T>
inline bool cmp(T left, T right) {
    return left == right;
}

template <typename T>
inline bool validate(T* ptr, T val, size_t len, size_t tail = 1,
                     size_t offset = 0);

template <typename T>
inline bool validate(T* got, T* expected, size_t len, size_t tail = 1,
                     size_t offset = 0);

template <typename T>
inline bool validate(T* ptr, T val, size_t len, size_t tail, size_t offset) {
    size_t i = offset;
    size_t tailIter = 0;
    while (i < len) {
        if (!cmp(ptr[i], val)) {
            if (tailIter < tail) {
                CHECKVAL(ptr[i], val, i);
                tailIter++;
            } else {
                return false;
            }
        }
        i++;
    }
    return true;
}

template <>
inline bool validate(__fp16* ptr, __fp16 val, size_t len, size_t tail, size_t offset) {
    size_t i = offset;
    size_t tailIter = 0;
    while (i < len) {
        if (!cmp(ptr[i], val)) {
            if (tailIter < tail) {
                CHECKVAL(float(ptr[i]), float(val), i);
                tailIter++;
            } else {
                return false;
            }
        }
        i++;
    }
    return true;
}

template <typename T>
inline bool validate(T* got, T* expected, size_t len, size_t tail,
                     size_t offset) {
    size_t i = offset;
    size_t tailIter = 0;
    while (i < len) {
        if (!cmp(got[i], expected[i])) {
            if (tailIter < tail) {
                CHECKVAL(got[i], expected[i], i);
                tailIter++;
            } else {
                return false;
            }
        }
        i++;
    }
    return true;
}

template <>
inline bool validate(__fp16* got, __fp16* expected, size_t len, size_t tail,
                     size_t offset) {
    size_t i = offset;
    size_t tailIter = 0;
    while (i < len) {
        if (!cmp(float(got[i]), float(expected[i]))) {
            if (tailIter < tail) {
                CHECKVAL(float(got[i]), float(expected[i]), i);
                tailIter++;
            } else {
                return false;
            }
        }
        i++;
    }
    return true;
}
