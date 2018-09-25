/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclDataTypes.h
 * @brief This file contains data types required for rccl operations
 *
 * Clang based vector types are defined here
 *
 * @author Aditya Atluri
 */

#pragma once

typedef signed char rccl_char16_t __attribute__((ext_vector_type(16)));
typedef unsigned char rccl_uchar16_t __attribute__((ext_vector_type(16)));
typedef signed short rccl_short8_t __attribute__((ext_vector_type(8)));
typedef unsigned short rccl_ushort8_t __attribute__((ext_vector_type(8)));
typedef signed int rccl_int4_t __attribute__((ext_vector_type(4)));
typedef unsigned int rccl_uint4_t __attribute__((ext_vector_type(4)));
typedef signed long rccl_long2_t __attribute__((ext_vector_type(2)));
typedef unsigned long rccl_ulong2_t __attribute__((ext_vector_type(2)));

typedef __fp16 rccl_half_t;
typedef __fp16 rccl_half8_t __attribute__((ext_vector_type(8)));
typedef float rccl_float4_t __attribute__((ext_vector_type(4)));
typedef double rccl_double2_t __attribute__((ext_vector_type(2)));
