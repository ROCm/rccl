/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdint.h>

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache);

#define NCCL_PARAM(name, env, deftVal) \
  int64_t ncclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false)) { \
      ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache); \
    } \
    return cache; \
  }

#define RCCL_PARAM_DECLARE(name) \
int64_t rcclParam##name()

#define RCCL_PARAM(name, env, default_value) \
pthread_mutex_t rcclParamMutex##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t rcclParam##name() { \
  static_assert(default_value != -1LL, "default value cannot be -1"); \
  static int64_t value = -1LL; \
  int64_t localValue; \
  pthread_mutex_lock(&rcclParamMutex##name); \
  localValue = value; \
  char* en = getenv("RCCL_TEST_ENV_VARS"); \
  if (value == -1LL || (en && (strcmp(en, "ENABLE") == 0))){  \
    value = default_value; \
    char* str = getenv("RCCL_" env); \
    if (str && strlen(str) > 0) { \
      errno = 0; \
      int64_t v = strtoll(str, NULL, 0); \
      if (errno) { \
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lu.", str, "RCCL_" env, value); \
      } else { \
        value = v; \
        INFO(NCCL_ALL,"%s set by environment to %lu.", "RCCL_" env, value);  \
      } \
    } \
    localValue = value; \
  } \
  pthread_mutex_unlock(&rcclParamMutex##name); \
  return localValue; \
}

#endif
