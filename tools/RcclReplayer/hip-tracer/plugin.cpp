/* Copyright (c) Advanced Micro Devices, Inc. */
/*
 * RCCL HIP Tracer Plugin
 *
 * Uses roctracer to intercept HIP synchronization calls and records them
 * into RCCL's trace via the exported C API.
 */

#include <cstdio>
#include <atomic>

#include <roctracer/roctracer.h>
#include <hip/amd_detail/hip_prof_str.h>

#include "rccl_hip_api.h"

// RCCL exported C API (linked at load time)
extern "C" {
    int rccl_hip_recording_enabled(void);
    void rccl_record_hip_call(int type, void* stream, void* event);
}

namespace {

constexpr uint32_t ACTIVITY_DOMAIN_HIP_API = 3;

std::atomic<bool> initialized{false};
std::atomic<bool> recording_active{false};
std::atomic<bool> recording_checked{false};

void hip_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    // Check recording status once
    if (!recording_checked.load(std::memory_order_acquire)) {
        bool expected = false;
        if (recording_checked.compare_exchange_strong(expected, true)) {
            if (rccl_hip_recording_enabled()) {
                recording_active.store(true, std::memory_order_release);
                fprintf(stderr, "[RCCL HIP Tracer] Recording enabled\n");
            }
        }
    }

    if (!recording_active.load(std::memory_order_relaxed)) return;

    const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
    if (!data || data->phase != ACTIVITY_API_PHASE_EXIT) return;

    void* stream = nullptr;
    void* event = nullptr;
    int type = -1;

    switch (cid) {
        case HIP_API_ID_hipStreamSynchronize:
            stream = data->args.hipStreamSynchronize.stream;
            type = RCCL_HIP_STREAM_SYNCHRONIZE;
            break;
        case HIP_API_ID_hipDeviceSynchronize:
            type = RCCL_HIP_DEVICE_SYNCHRONIZE;
            break;
        case HIP_API_ID_hipEventSynchronize:
            event = data->args.hipEventSynchronize.event;
            type = RCCL_HIP_EVENT_SYNCHRONIZE;
            break;
        case HIP_API_ID_hipEventRecord:
            event = data->args.hipEventRecord.event;
            stream = data->args.hipEventRecord.stream;
            type = RCCL_HIP_EVENT_RECORD;
            break;
        case HIP_API_ID_hipStreamWaitEvent:
            stream = data->args.hipStreamWaitEvent.stream;
            event = data->args.hipStreamWaitEvent.event;
            type = RCCL_HIP_STREAM_WAIT_EVENT;
            break;
        case HIP_API_ID_hipEventCreate:
            event = *data->args.hipEventCreate.event;
            type = RCCL_HIP_EVENT_CREATE;
            break;
        case HIP_API_ID_hipEventCreateWithFlags:
            event = *data->args.hipEventCreateWithFlags.event;
            type = RCCL_HIP_EVENT_CREATE;
            break;
        case HIP_API_ID_hipEventDestroy:
            event = data->args.hipEventDestroy.event;
            type = RCCL_HIP_EVENT_DESTROY;
            break;
        default:
            return;
    }

    rccl_record_hip_call(type, stream, event);
}

__attribute__((constructor))
void plugin_init() {
    bool expected = false;
    if (!initialized.compare_exchange_strong(expected, true)) return;

    roctracer_status_t status = roctracer_enable_domain_callback(
        static_cast<activity_domain_t>(ACTIVITY_DOMAIN_HIP_API),
        hip_api_callback,
        nullptr
    );

    if (status == ROCTRACER_STATUS_SUCCESS) {
        fprintf(stderr, "[RCCL HIP Tracer] Loaded\n");
    } else {
        fprintf(stderr, "[RCCL HIP Tracer] Failed to register roctracer: %d\n", status);
        initialized.store(false);
    }
}

__attribute__((destructor))
void plugin_fini() {
    if (initialized.load()) {
        roctracer_disable_domain_callback(static_cast<activity_domain_t>(ACTIVITY_DOMAIN_HIP_API));
    }
}

}  // namespace
