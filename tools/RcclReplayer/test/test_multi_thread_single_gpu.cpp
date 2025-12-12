/**
 * Multi-threaded Single GPU Test for RCCL Replayer
 * 
 * Tests the replayer's ability to handle logs from multi-threaded applications
 * where multiple CPU threads issue NCCL collectives on the same GPU.
 * 
 * Pattern:
 * - Single GPU per MPI rank, multiple CPU threads per rank
 * - Each thread creates its own communicator
 * - Threads run concurrent collectives (AllReduce, etc.)
 * - Thread N on GPU0 <-> Thread N on GPU1 = Comm N
 */

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        std::cerr << "NCCL error: " << ncclGetErrorString(r) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Global synchronization
std::atomic<int> threadsReady{0};
std::atomic<bool> startCollectives{false};

// Thread function - receives pre-created communicator
void threadFunc(int threadId, int mpiRank, ncclComm_t comm, int device) {
    HIP_CHECK(hipSetDevice(device));
    
    // Each thread creates its own stream on the SAME device
    hipStream_t stream;
    hipEvent_t event;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipEventCreate(&event));
    
    std::cout << "[MPI " << mpiRank << " Thread " << threadId << "] Started, stream " << stream 
              << ", comm " << comm << std::endl;
    
    // Allocate buffers (all on same device)
    const size_t size = 128 * 1024;
    float *sendbuf, *recvbuf;
    HIP_CHECK(hipMalloc(&sendbuf, size * sizeof(float)));
    HIP_CHECK(hipMalloc(&recvbuf, size * sizeof(float)));
    HIP_CHECK(hipMemset(sendbuf, 0, size * sizeof(float)));
    
    // Signal ready and wait for all threads globally
    threadsReady++;
    while (!startCollectives.load()) {
        std::this_thread::yield();
    }
    
    // === CONCURRENT COLLECTIVES ===
    // All threads submit collectives roughly at the same time
    for (int i = 0; i < 5; i++) {
        // Submit collective (concurrently with other threads!)
        NCCL_CHECK(ncclAllReduce(sendbuf, recvbuf, size, ncclFloat, ncclSum, comm, stream));
        
        // Record event and sync
        HIP_CHECK(hipEventRecord(event, stream));
        
        if (i % 2 == 0) {
            HIP_CHECK(hipStreamSynchronize(stream));
        } else {
            HIP_CHECK(hipEventSynchronize(event));
        }
        
        std::cout << "[MPI " << mpiRank << " Thread " << threadId << "] Iteration " << i << " done" << std::endl;
    }
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Cleanup (but NOT the comm - main thread owns it)
    HIP_CHECK(hipFree(sendbuf));
    HIP_CHECK(hipFree(recvbuf));
    HIP_CHECK(hipEventDestroy(event));
    HIP_CHECK(hipStreamDestroy(stream));
    
    std::cout << "[MPI " << mpiRank << " Thread " << threadId << "] Complete" << std::endl;
}

int main(int argc, char** argv) {
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI does not support MPI_THREAD_MULTIPLE" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    
    // Number of threads (communicators) per GPU - configurable
    int numThreads = 3;
    if (argc > 1) {
        numThreads = atoi(argv[1]);
    }
    
    // Each MPI rank uses ONE GPU, but has multiple threads/comms
    int device = mpiRank;
    HIP_CHECK(hipSetDevice(device));
    
    if (mpiRank == 0) {
        std::cout << "=== Multi-Thread Single-GPU Test ===" << std::endl;
        std::cout << "MPI ranks (GPUs): " << mpiSize << std::endl;
        std::cout << "Threads per GPU: " << numThreads << std::endl;
        std::cout << "Total communicators: " << numThreads << " (each spans " << mpiSize << " GPUs)" << std::endl;
        std::cout << "=====================================" << std::endl;
    }
    
    // ===== PHASE 1: Create communicators =====
    std::vector<ncclComm_t> comms(numThreads);
    std::vector<ncclUniqueId> uniqueIds(numThreads);
    
    // Generate unique IDs on rank 0 and broadcast
    for (int t = 0; t < numThreads; t++) {
        if (mpiRank == 0) {
            NCCL_CHECK(ncclGetUniqueId(&uniqueIds[t]));
        }
        MPI_Bcast(&uniqueIds[t], sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    
    std::cout << "[MPI " << mpiRank << "] Creating " << numThreads << " communicators..." << std::endl;
    
    // Create each communicator
    for (int t = 0; t < numThreads; t++) {
        NCCL_CHECK(ncclCommInitRank(&comms[t], mpiSize, uniqueIds[t], mpiRank));
        std::cout << "[MPI " << mpiRank << "] Comm " << t << " created: " << comms[t] << std::endl;
    }
    
    std::cout << "[MPI " << mpiRank << "] All communicators created" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== PHASE 2: Spawn threads with pre-created comms =====
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back(threadFunc, t, mpiRank, comms[t], device);
    }
    
    // Wait for all threads across all MPI ranks to be ready
    while (threadsReady.load() < numThreads) {
        std::this_thread::yield();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Signal all threads to start collectives CONCURRENTLY
    startCollectives.store(true);
    
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== PHASE 3: Cleanup communicators =====
    for (int t = 0; t < numThreads; t++) {
        NCCL_CHECK(ncclCommDestroy(comms[t]));
    }
    
    if (mpiRank == 0) {
        std::cout << "=== All ranks completed successfully ===" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
