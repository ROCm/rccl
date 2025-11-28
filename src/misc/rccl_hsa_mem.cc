#include "rccl_hsa_mem.h"

#define CHECK_HSA(msg, status) \
    if ((status) != HSA_STATUS_SUCCESS) { \
        fprintf(stderr, "%s failed (%d)\n", msg, status); \
        return hipErrorUnknown; \
    }

static const std::vector<hsa_agent_t>& get_all_gpu_agents()
{
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    static int initialized = 0;
    static std::vector<hsa_agent_t> agents;
    if (!__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) {
        pthread_mutex_lock(&mutex);
        if (!__atomic_load_n(&initialized, __ATOMIC_RELAXED)) {
            hsa_status_t status = hsa_iterate_agents([](hsa_agent_t agent, void *data) -> hsa_status_t
                               {
                            hsa_device_type_t type;
                            if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS){
                                return HSA_STATUS_ERROR;
                            }
                            if (type == HSA_DEVICE_TYPE_GPU) {
                                ((std::vector<hsa_agent_t>*)data)->push_back(agent);
                            }
                            return HSA_STATUS_SUCCESS; }, &agents);
            // Log warning on status
            __atomic_store_n(&initialized, 1, __ATOMIC_RELEASE);
        }
        pthread_mutex_unlock(&mutex);
    }
    return agents;
}

static const std::unordered_map<uint32_t,hsa_agent_t>& get_bdf_to_gpu_agent_map() {
    
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    static int initialized = 0;
    static std::unordered_map<uint32_t,hsa_agent_t> bdf_to_agent_map;
    const std::vector<hsa_agent_t>& gpu_agents = get_all_gpu_agents();
    if (!__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) {
        pthread_mutex_lock(&mutex);
        if (!__atomic_load_n(&initialized, __ATOMIC_RELAXED)) {
            for (auto &agent : gpu_agents) {
                uint32_t bdfid = 0;
                if (hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdfid) == HSA_STATUS_SUCCESS) {
                    bdf_to_agent_map[bdfid] = agent;
                }
            }
            __atomic_store_n(&initialized, 1, __ATOMIC_RELEASE);
        }
        pthread_mutex_unlock(&mutex);
    }
    return bdf_to_agent_map;
}

static uint32_t pci_bus_str_to_bdfid(const char* bus_id) {
    unsigned domain, bus, device, function;
    if (!bus_id || sscanf(bus_id, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4) {
         return 0xffffffff;
    }
    // Encode domain:bus:device.function as 32-bit BDF ID
    return (domain << 16) | (bus << 8) | (device << 3) | function;
}

static const std::unordered_map<uint32_t, hsa_agent_t>& get_hip_device_to_hsa_agent_map_by_bdfid() {
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    static int initialized = 0;
    static std::unordered_map<uint32_t, hsa_agent_t> device_to_agent;
    // Get HIP device count
    if (!__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) {
        pthread_mutex_lock(&mutex);
        if (!__atomic_load_n(&initialized, __ATOMIC_RELAXED)) {
            int hip_device_count = 0;
            hipError_t devcountStatus = hipGetDeviceCount(&hip_device_count);
            const std::unordered_map<uint32_t, hsa_agent_t> bdf_to_agent = get_bdf_to_gpu_agent_map();
            // Match HIP devices to HSA agents by PCI Bus ID
            for (uint32_t i = 0; i < hip_device_count; ++i) {
                char busid_str[64];
                hipError_t pcibusidStatus = hipDeviceGetPCIBusId(busid_str, sizeof(busid_str), i);
                uint32_t bdfid = pci_bus_str_to_bdfid(busid_str);
                auto it = bdf_to_agent.find(bdfid);
                if (it != bdf_to_agent.end()) {
                    device_to_agent[i] = it->second;
                } else {
                    // Log warning
                }
            }
            __atomic_store_n(&initialized, 1, __ATOMIC_RELEASE);
        }
        pthread_mutex_unlock(&mutex);
    }
    return device_to_agent;
}

static hsa_amd_memory_pool_t get_gpu_pool(hsa_agent_t agent) {
    hsa_amd_memory_pool_t pool{};
    hsa_amd_agent_iterate_memory_pools(agent, [](hsa_amd_memory_pool_t pool, void* data) -> hsa_status_t {
        uint32_t alloc = 0;
        hsa_status_t status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
        if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
            // log warning
        }
        if (alloc) {
            *(hsa_amd_memory_pool_t*)data = pool;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    }, &pool);
    return pool;
}

static hsa_status_t hsa_init_once() {
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    static int initialized = 0;
    static hsa_status_t returnStatus = HSA_STATUS_ERROR_NOT_INITIALIZED;
    if (!__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) {
        pthread_mutex_lock(&mutex);
        if (!__atomic_load_n(&initialized, __ATOMIC_RELAXED)) {
            //returnStatus = hsa_init();
            int hip_device_count = 0;
            hipError_t devcountStatus = hipGetDeviceCount(&hip_device_count);
            if(devcountStatus == hipSuccess){
                returnStatus = HSA_STATUS_SUCCESS;
            }
            __atomic_store_n(&initialized, 1, __ATOMIC_RELEASE);
        }
        pthread_mutex_unlock(&mutex);
    }
    return returnStatus;
}

hipError_t rcclExtMallocWithFlagsOnDevice(void** ptr,
                                         size_t size,
                                         uint32_t flags) {
    if (!ptr || size == 0)
        return hipErrorInvalidValue;
    if (flags == hipDeviceMallocUncached){
        int device_id;
        hipError_t hip_status  = hipGetDevice(&device_id);
        hsa_status_t status = hsa_init_once();
        if (status != HSA_STATUS_SUCCESS) {
            return hipErrorInitializationError;
        }

        const std::unordered_map<uint32_t, hsa_agent_t>& device_id_to_agent = get_hip_device_to_hsa_agent_map_by_bdfid();
        auto it = device_id_to_agent.find(device_id);
        if (it == device_id_to_agent.end()) {
            fprintf(stderr, "[rcclExtMalloc] Invalid device_id: %d\n", device_id);
            return hipErrorInvalidDevice;
        }
        hsa_agent_t target_agent = it->second;
        hsa_amd_memory_pool_t pool = get_gpu_pool(target_agent);
        if (pool.handle == 0) {
            return hipErrorMemoryAllocation;
        }
        void* mem = nullptr;
        status = hsa_amd_memory_pool_allocate(pool, size, flags, &mem);
        CHECK_HSA("hsa_amd_memory_pool_allocate", status);
        status = hsa_amd_agents_allow_access(1, &target_agent, NULL, mem);
        CHECK_HSA("hsa_amd_agents_allow_access", status);
        *ptr = mem;
        return hipSuccess;
    }
    // ------- NORMAL HIP PATH -------
    hipError_t hip_status  =  hipExtMallocWithFlags(ptr, size, flags);
    return hip_status;
}


hipError_t rcclExtFree_impl(void* ptr) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    hsa_status_t status = hsa_amd_memory_pool_free(ptr);
    return (status == HSA_STATUS_SUCCESS) ? hipSuccess : hipErrorInvalidDevicePointer;
}