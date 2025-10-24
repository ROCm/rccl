#ifndef RCCL_HSA_MEM_H_
#define RCCL_HSA_MEM_H_

/**
 * @file rccl_hsa_mem.h
 * @brief RCCL extended memory allocation utilities for HIP/HSA interoperability.
 *
 * This header provides helper APIs for allocating and freeing GPU memory
 * with specific flags or placement control using the HSA runtime.
 *
 * It integrates HIP and HSA memory management to allow advanced allocation
 * modes such as uncached device memory or memory allocated on a specific device.
 */

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <pthread.h>

/**
 * @brief Allocate GPU memory with extended HSA/HIP flags on a specific device.
 *
 * This function allocates device memory with custom behavior, similar to
 * `hipExtMallocWithFlags()`, but extended to allow explicit device selection
 * and integration with the HSA memory segment model.
 *
 * @param[out] ptr          Pointer to the allocated memory.
 * @param[in]  size         Size of memory to allocate in bytes.
 * @param[in]  flags        HIP allocation flags (e.g., `hipDeviceMallocUncached`).
 * @param[in]  device       Target HIP device ID to allocate on.
 *
 * @return
 *   - `hipSuccess` on success.
 *   - Appropriate `hipError_t` on failure.
 *
 * @note
 *   - The underlying allocation uses `hsa_amd_memory_pool_allocate()`.
 *   - Querying pointer attributes through hip api doesnt show device ID as allocation code path is bypassing hip layer
 */
hipError_t rcclExtMallocWithFlagsOnDevice(void** ptr, size_t size, uint32_t flags, int device);

/**
 * @brief Free memory allocated by RCCL or via rcclExtMallocWithFlagsOnDevice.
 *
 * This function safely frees HSA/HIP memory previously allocated using
 * `rcclExtMallocWithFlagsOnDevice()` or similar custom allocators.
 *
 * @param[in] ptr   Pointer to memory to free.
 *
 * @return
 *   - `hipSuccess` on success.
 *   - Appropriate `hipError_t` on failure.
 *
 * @note
 */
hipError_t rcclExtFree_impl(void* ptr);

#endif // RCCL_HSA_MEM_H_
