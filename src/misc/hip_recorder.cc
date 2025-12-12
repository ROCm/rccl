/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#include <hip/hip_runtime.h>  // Must be first for hipStream_t, hipEvent_t, hipError_t
#include "nccl.h"             // NCCL types (ncclDataType_t, ncclComm_t, etc.)

// Forward declare ncclInfo (internal RCCL type used in recorder.h)
struct ncclInfo;

#include "recorder.h"         // Depends on both HIP and NCCL types
#include <dlfcn.h>
#include <mutex>

// Function pointer types for real HIP functions
typedef hipError_t (*hipStreamSynchronize_t)(hipStream_t);
typedef hipError_t (*hipDeviceSynchronize_t)(void);
typedef hipError_t (*hipEventSynchronize_t)(hipEvent_t);
typedef hipError_t (*hipEventRecord_t)(hipEvent_t, hipStream_t);
typedef hipError_t (*hipStreamWaitEvent_t)(hipStream_t, hipEvent_t, unsigned int);
typedef hipError_t (*hipEventCreate_t)(hipEvent_t*);
typedef hipError_t (*hipEventCreateWithFlags_t)(hipEvent_t*, unsigned int);
typedef hipError_t (*hipEventDestroy_t)(hipEvent_t);

// Get real HIP function pointers using dlsym
static hipStreamSynchronize_t real_hipStreamSynchronize = nullptr;
static hipDeviceSynchronize_t real_hipDeviceSynchronize = nullptr;
static hipEventSynchronize_t real_hipEventSynchronize = nullptr;
static hipEventRecord_t real_hipEventRecord = nullptr;
static hipStreamWaitEvent_t real_hipStreamWaitEvent = nullptr;
static hipEventCreate_t real_hipEventCreate = nullptr;
static hipEventCreateWithFlags_t real_hipEventCreateWithFlags = nullptr;
static hipEventDestroy_t real_hipEventDestroy = nullptr;

// Thread-safe initialization
static std::once_flag init_flag;

static void do_init_hip_functions() {
  real_hipStreamSynchronize = (hipStreamSynchronize_t)dlsym(RTLD_NEXT, "hipStreamSynchronize");
  real_hipDeviceSynchronize = (hipDeviceSynchronize_t)dlsym(RTLD_NEXT, "hipDeviceSynchronize");
  real_hipEventSynchronize = (hipEventSynchronize_t)dlsym(RTLD_NEXT, "hipEventSynchronize");
  real_hipEventRecord = (hipEventRecord_t)dlsym(RTLD_NEXT, "hipEventRecord");
  real_hipStreamWaitEvent = (hipStreamWaitEvent_t)dlsym(RTLD_NEXT, "hipStreamWaitEvent");
  real_hipEventCreate = (hipEventCreate_t)dlsym(RTLD_NEXT, "hipEventCreate");
  real_hipEventCreateWithFlags = (hipEventCreateWithFlags_t)dlsym(RTLD_NEXT, "hipEventCreateWithFlags");
  real_hipEventDestroy = (hipEventDestroy_t)dlsym(RTLD_NEXT, "hipEventDestroy");
}

// Initialize function pointers once (thread-safe)
static void init_hip_functions() {
  std::call_once(init_flag, do_init_hip_functions);
}

extern "C" {

// Intercept hipStreamSynchronize
hipError_t hipStreamSynchronize(hipStream_t stream) {
  init_hip_functions();
  
  // Record the call BEFORE executing it (to capture intent)
  rccl::Recorder::instance().record(rccl::rrHipStreamSynchronize, stream, (hipEvent_t)nullptr);
  
  // Execute the real HIP function
  hipError_t result = real_hipStreamSynchronize(stream);
  
  return result;
}

// Intercept hipDeviceSynchronize
hipError_t hipDeviceSynchronize(void) {
  init_hip_functions();
  
  // Record the call - cast nullptr to resolve overload ambiguity
  rccl::Recorder::instance().record(rccl::rrHipDeviceSynchronize, (hipStream_t)nullptr, (hipEvent_t)nullptr);
  
  // Execute the real HIP function
  hipError_t result = real_hipDeviceSynchronize();
  
  return result;
}

// Intercept hipEventSynchronize
hipError_t hipEventSynchronize(hipEvent_t event) {
  init_hip_functions();
  
  // Record the call - cast nullptr to resolve overload ambiguity
  rccl::Recorder::instance().record(rccl::rrHipEventSynchronize, (hipStream_t)nullptr, event);
  
  // Execute the real HIP function
  hipError_t result = real_hipEventSynchronize(event);
  
  return result;
}

// Intercept hipEventRecord
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  init_hip_functions();
  
  // Record the call
  rccl::Recorder::instance().record(rccl::rrHipEventRecord, stream, event);
  
  // Execute the real HIP function
  hipError_t result = real_hipEventRecord(event, stream);
  
  return result;
}

// Intercept hipStreamWaitEvent
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  init_hip_functions();
  
  // Record the call
  rccl::Recorder::instance().record(rccl::rrHipStreamWaitEvent, stream, event);
  
  // Execute the real HIP function
  hipError_t result = real_hipStreamWaitEvent(stream, event, flags);
  
  return result;
}

// Intercept hipEventCreate
hipError_t hipEventCreate(hipEvent_t* event) {
  init_hip_functions();
  
  // Execute the real HIP function first to get the event pointer
  hipError_t result = real_hipEventCreate(event);
  
  // Record the call with the newly created event (stream is nullptr)
  if (result == hipSuccess && event) {
    rccl::Recorder::instance().record(rccl::rrHipEventCreate, (hipStream_t)nullptr, *event);
  }
  
  return result;
}

// Intercept hipEventCreateWithFlags
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
  init_hip_functions();
  
  // Execute the real HIP function first to get the event pointer
  hipError_t result = real_hipEventCreateWithFlags(event, flags);
  
  // Record the call with the newly created event (stream is nullptr)
  if (result == hipSuccess && event) {
    rccl::Recorder::instance().record(rccl::rrHipEventCreate, (hipStream_t)nullptr, *event);
  }
  
  return result;
}

// Intercept hipEventDestroy
hipError_t hipEventDestroy(hipEvent_t event) {
  init_hip_functions();
  
  // Record the call before destroying (stream is nullptr)
  rccl::Recorder::instance().record(rccl::rrHipEventDestroy, (hipStream_t)nullptr, event);
  
  // Execute the real HIP function
  hipError_t result = real_hipEventDestroy(event);
  
  return result;
}

} // extern "C"

