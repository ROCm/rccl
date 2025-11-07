#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#define NGPUS 8
#define NRINGS 7
#define MAX_GPUS 8
#define WARP_SIZE 64
#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Ring topology in device constant memory
__device__ __constant__ int d_rings[NRINGS][MAX_GPUS] = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {1, 0, 2, 4, 3, 5, 7, 6},
    {2, 5, 0, 3, 6, 1, 7, 4},
    {3, 7, 0, 4, 2, 1, 6, 5},
    {4, 6, 2, 7, 3, 0, 5, 1},
    {5, 4, 7, 1, 3, 2, 6, 0},
    {6, 3, 1, 4, 0, 7, 5, 2}
};

// ==================== HELPER FUNCTIONS ====================
__device__ __forceinline__
void get_segment(size_t total, int parts, int idx, size_t &start, size_t &end) {
    size_t base = total / parts;
    size_t rem = total % parts;
    if (idx < (int)rem) {
        start = idx * (base + 1);
        end = start + (base + 1);
    } else {
        start = rem * (base + 1) + (idx - rem) * base;
        end = start + base;
    }
    if (start > total) start = total;
    if (end > total) end = total;
}

// ==================== INITIALIZATION KERNEL ====================
__global__ void init_buffer_kernel(float* buffer, float init_val, size_t num_floats) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t j = idx; j < num_floats; j += stride) {
        buffer[j] = init_val;
    }
}

// ==================== THREADBLOCK MODE KERNEL ====================
__global__ void ring_allreduce_threadblock_kernel(
    float* local_buffer,
    float** peer_ptrs,
    int my_gpu_id,
    size_t num_floats,
    int step,
    int num_threadblocks)
{
    int block_id = blockIdx.x;
    int ring_id = block_id % NRINGS;
    
    // Find my position in this ring
    int my_ring_pos = 0;
    for (int i = 0; i < NGPUS; i++) {
        if (d_rings[ring_id][i] == my_gpu_id) {
            my_ring_pos = i;
            break;
        }
    }
    
    // Compute predecessor in the ring
    int pred_pos = (my_ring_pos - 1 + NGPUS) % NGPUS;
    int pred_gpu = d_rings[ring_id][pred_pos];
    
    // Use proper segmentation: divide total data among blocks first
    const int chunkIdx = block_id;
    size_t chunkStart, chunkEnd;
    get_segment(num_floats, num_threadblocks, chunkIdx, chunkStart, chunkEnd);
    const size_t chunkElems = (chunkEnd > chunkStart) ? (chunkEnd - chunkStart) : 0;
    
    if (chunkElems == 0) return;
    
    // Now divide this chunk into NGPUS ring pieces
    auto chunk_segment = [&](int ringChunkIdx, size_t &s, size_t &e) {
        size_t start_rel, end_rel;
        get_segment(chunkElems, NGPUS, ringChunkIdx, start_rel, end_rel);
        s = chunkStart + start_rel;
        e = chunkStart + end_rel;
    };
    
    // Determine which ring chunk to receive this step
    int recvChunk;
    if (step < NGPUS - 1) {
        // Reduce-Scatter phase
        recvChunk = (my_ring_pos - step - 1 + NGPUS) % NGPUS;
    } else {
        // AllGather phase
        int agStep = step - (NGPUS - 1);
        recvChunk = (my_ring_pos - agStep + NGPUS) % NGPUS;
    }
    
    size_t block_start, block_end;
    chunk_segment(recvChunk, block_start, block_end);
    
    // Each thread processes multiple elements
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    float* pred_buffer = peer_ptrs[pred_gpu];
    
    // Process with float4 packing when possible
    size_t work_size = block_end - block_start;
    size_t num_float4 = work_size / 4;
    size_t remainder = work_size % 4;
    
    // Process float4s
    for (size_t i = tid; i < num_float4; i += num_threads) {
        size_t idx = block_start + i * 4;
        float4 local_val = *reinterpret_cast<float4*>(&local_buffer[idx]);
        float4 pred_val = *reinterpret_cast<float4*>(&pred_buffer[idx]);
        
        if (step < NGPUS - 1) {
            // Reduce-Scatter: add
            local_val.x += pred_val.x;
            local_val.y += pred_val.y;
            local_val.z += pred_val.z;
            local_val.w += pred_val.w;
        } else {
            // AllGather: copy
            local_val = pred_val;
        }
        
        *reinterpret_cast<float4*>(&local_buffer[idx]) = local_val;
    }
    
    // Handle remainder
    size_t remainder_start = block_start + num_float4 * 4;
    for (size_t i = tid; i < remainder; i += num_threads) {
        size_t idx = remainder_start + i;
        if (step < NGPUS - 1) {
            local_buffer[idx] += pred_buffer[idx];
        } else {
            local_buffer[idx] = pred_buffer[idx];
        }
    }
}

// ==================== WARP MODE KERNEL ====================
__global__ void ring_allreduce_warp_kernel(
    float* local_buffer,
    float** peer_ptrs,
    int my_gpu_id,
    size_t num_floats,
    int step,
    int num_threadblocks)
{
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Each warp handles a different ring (7 warps)
    int ring_id = warp_id % NRINGS;
    
    // Skip if this warp is beyond the 7 rings
    if (ring_id >= NRINGS) return;
    
    // Find my position in this ring
    int my_ring_pos = 0;
    for (int i = 0; i < NGPUS; i++) {
        if (d_rings[ring_id][i] == my_gpu_id) {
            my_ring_pos = i;
            break;
        }
    }
    
    // Compute predecessor in the ring
    int pred_pos = (my_ring_pos - 1 + NGPUS) % NGPUS;
    int pred_gpu = d_rings[ring_id][pred_pos];
    
    // Use proper segmentation: divide total data among blocks first
    const int chunkIdx = blockIdx.x;
    size_t chunkStart, chunkEnd;
    get_segment(num_floats, num_threadblocks, chunkIdx, chunkStart, chunkEnd);
    const size_t chunkElems = (chunkEnd > chunkStart) ? (chunkEnd - chunkStart) : 0;
    
    if (chunkElems == 0) return;
    
    // Within block's chunk, subdivide among the NRINGS warps
    size_t lineStart, lineEnd;
    get_segment(chunkElems, NRINGS, ring_id, lineStart, lineEnd);
    const size_t tileStart = chunkStart + lineStart;
    const size_t tileEnd = chunkStart + lineEnd;
    const size_t tileElems = (tileEnd > tileStart) ? (tileEnd - tileStart) : 0;
    
    if (tileElems == 0) return;
    
    // Now divide tile into NGPUS ring pieces
    auto chunk_segment = [&](int ringChunkIdx, size_t &s, size_t &e) {
        size_t start_rel, end_rel;
        get_segment(tileElems, NGPUS, ringChunkIdx, start_rel, end_rel);
        s = tileStart + start_rel;
        e = tileStart + end_rel;
    };
    
    // Determine which ring chunk to receive this step
    int recvChunk;
    if (step < NGPUS - 1) {
        // Reduce-Scatter phase
        recvChunk = (my_ring_pos - step - 1 + NGPUS) % NGPUS;
    } else {
        // AllGather phase
        int agStep = step - (NGPUS - 1);
        recvChunk = (my_ring_pos - agStep + NGPUS) % NGPUS;
    }
    
    size_t warp_start, warp_end;
    chunk_segment(recvChunk, warp_start, warp_end);
    
    float* pred_buffer = peer_ptrs[pred_gpu];
    
    // Process with float4 packing when possible
    size_t work_size = warp_end - warp_start;
    size_t num_float4 = work_size / 4;
    size_t remainder = work_size % 4;
    
    // Process float4s
    for (size_t i = lane_id; i < num_float4; i += WARP_SIZE) {
        size_t idx = warp_start + i * 4;
        float4 local_val = *reinterpret_cast<float4*>(&local_buffer[idx]);
        float4 pred_val = *reinterpret_cast<float4*>(&pred_buffer[idx]);
        
        if (step < NGPUS - 1) {
            // Reduce-Scatter: add
            local_val.x += pred_val.x;
            local_val.y += pred_val.y;
            local_val.z += pred_val.z;
            local_val.w += pred_val.w;
        } else {
            // AllGather: copy
            local_val = pred_val;
        }
        
        *reinterpret_cast<float4*>(&local_buffer[idx]) = local_val;
    }
    
    // Handle remainder
    size_t remainder_start = warp_start + num_float4 * 4;
    for (size_t i = lane_id; i < remainder; i += WARP_SIZE) {
        size_t idx = remainder_start + i;
        if (step < NGPUS - 1) {
            local_buffer[idx] += pred_buffer[idx];
        } else {
            local_buffer[idx] = pred_buffer[idx];
        }
    }
}

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -s, --size SIZE         Data size in bytes (default: 1GB)\n");
    printf("  -m, --mode MODE         Mode: threadblock or warp (default: threadblock)\n");
    printf("  -b, --blocks BLOCKS     Number of threadblocks per GPU (default: 28 for threadblock, 32 for warp)\n");
    printf("  -w, --warmup WARMUP     Number of warmup iterations (default: 5)\n");
    printf("  -i, --iters ITERS       Number of benchmark iterations (default: 10)\n");
    printf("  -h, --help              Show this help message\n");
}

int main(int argc, char** argv) {
    // Default parameters
    size_t data_bytes = 1ULL * 1024 * 1024 * 1024; // 1GB
    const char* mode = "threadblock";
    int num_threadblocks = -1; // -1 means use default based on mode
    int warmup_iters = 5;
    int bench_iters = 10;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--size") == 0) {
            if (i + 1 < argc) {
                data_bytes = strtoull(argv[++i], NULL, 0);
            }
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mode") == 0) {
            if (i + 1 < argc) {
                mode = argv[++i];
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--blocks") == 0) {
            if (i + 1 < argc) {
                num_threadblocks = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 < argc) {
                warmup_iters = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iters") == 0) {
            if (i + 1 < argc) {
                bench_iters = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    bool is_warp_mode = (strcmp(mode, "warp") == 0);
    
    // Set default threadblocks if not specified
    if (num_threadblocks == -1) {
        num_threadblocks = is_warp_mode ? 32 : 28;
    }
    
    int threads_per_block = 448;
    size_t num_floats = data_bytes / sizeof(float);
    
    // Validate parameters
    if (!is_warp_mode && strcmp(mode, "threadblock") != 0) {
        fprintf(stderr, "Error: Mode must be 'threadblock' or 'warp'\n");
        return 1;
    }
    
    if (is_warp_mode) {
        int warps_per_block = threads_per_block / WARP_SIZE;
        if (warps_per_block < NRINGS) {
            fprintf(stderr, "Error: Need at least %d warps per block (7 rings), but got %d\n", 
                    NRINGS, warps_per_block);
            return 1;
        }
    }
    
    // Print configuration
    printf("==================== Configuration ====================\n");
    printf("Mode:              %s\n", is_warp_mode ? "WARP" : "THREADBLOCK");
    printf("Data size:         %.2f GB (%zu bytes)\n", data_bytes / (1024.0 * 1024.0 * 1024.0), data_bytes);
    printf("Num floats:        %zu\n", num_floats);
    printf("Threadblocks/GPU:  %d\n", num_threadblocks);
    printf("Threads/block:     %d\n", threads_per_block);
    printf("Grid dim:          %d\n", num_threadblocks);
    printf("Block dim:         %d\n", threads_per_block);
    printf("Warmup iters:      %d\n", warmup_iters);
    printf("Benchmark iters:   %d\n", bench_iters);
    
    if (is_warp_mode) {
        int warps_per_block = threads_per_block / WARP_SIZE;
        printf("Warps/block:       %d\n", warps_per_block);
        printf("Total warps/GPU:   %d\n", num_threadblocks * warps_per_block);
        size_t work_per_warp = num_floats / NGPUS / num_threadblocks / (warps_per_block / NRINGS);
        printf("Work/warp:         %zu floats\n", work_per_warp);
    } else {
        size_t work_per_block = num_floats / num_threadblocks;
        printf("Work/block:        %zu floats\n", work_per_block);
    }
    printf("=======================================================\n\n");
    
    // Check GPU count
    int device_count = 0;
    CHECK_HIP(hipGetDeviceCount(&device_count));
    if (device_count < NGPUS) {
        fprintf(stderr, "Error: Need %d GPUs but only %d found\n", NGPUS, device_count);
        return 1;
    }
    
    // Allocate buffers and enable P2P
    float* d_buffers[NGPUS];
    float** h_peer_ptrs[NGPUS];
    float** d_peer_ptrs[NGPUS];
    
    printf("Initializing GPUs and enabling P2P access...\n");
    
    // Allocate on each GPU
    for (int i = 0; i < NGPUS; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMalloc(&d_buffers[i], data_bytes));
        
        // Initialize data: GPU i has all values (i+1).0
        float init_val = (float)(i + 1);
        
        // Simple initialization kernel
        hipLaunchKernelGGL(init_buffer_kernel,
                          dim3((num_floats + 255) / 256),
                          dim3(256),
                          0, 0,
                          d_buffers[i], init_val, num_floats);
        
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipGetLastError());  // Check for kernel launch errors
    }
    
    // Verify initialization
    printf("Verifying initialization...\n");
    for (int i = 0; i < NGPUS; i++) {
        CHECK_HIP(hipSetDevice(i));
        float h_test[10];
        CHECK_HIP(hipMemcpy(h_test, d_buffers[i], 10 * sizeof(float), hipMemcpyDeviceToHost));
        float expected = (float)(i + 1);
        bool init_ok = true;
        for (int j = 0; j < 10; j++) {
            if (fabs(h_test[j] - expected) > 1e-3) {
                init_ok = false;
                break;
            }
        }
        if (!init_ok) {
            printf("GPU %d initialization FAILED: expected %.1f, got %.1f\n", i, expected, h_test[0]);
        }
    }
    printf("Initialization verification complete.\n\n");
    
    // Enable P2P access between all pairs
    for (int i = 0; i < NGPUS; i++) {
        CHECK_HIP(hipSetDevice(i));
        for (int j = 0; j < NGPUS; j++) {
            if (i != j) {
                int can_access = 0;
                CHECK_HIP(hipDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    hipError_t err = hipDeviceEnablePeerAccess(j, 0);
                    if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
                        CHECK_HIP(err);
                    }
                } else {
                    fprintf(stderr, "Warning: GPU %d cannot access GPU %d\n", i, j);
                }
            }
        }
    }
    
    // Create peer pointer tables
    for (int i = 0; i < NGPUS; i++) {
        CHECK_HIP(hipSetDevice(i));
        h_peer_ptrs[i] = (float**)malloc(NGPUS * sizeof(float*));
        for (int j = 0; j < NGPUS; j++) {
            h_peer_ptrs[i][j] = d_buffers[j];
        }
        CHECK_HIP(hipMalloc(&d_peer_ptrs[i], NGPUS * sizeof(float*)));
        CHECK_HIP(hipMemcpy(d_peer_ptrs[i], h_peer_ptrs[i], NGPUS * sizeof(float*), hipMemcpyHostToDevice));
    }
    
    printf("P2P access enabled and peer pointer tables created.\n\n");
    
    // Warmup runs
    printf("Running %d warmup iterations...\n", warmup_iters);
    for (int iter = 0; iter < warmup_iters; iter++) {
        for (int step = 0; step < 2 * (NGPUS - 1); step++) {
            for (int gpu = 0; gpu < NGPUS; gpu++) {
                CHECK_HIP(hipSetDevice(gpu));
                
                if (is_warp_mode) {
                    hipLaunchKernelGGL(ring_allreduce_warp_kernel,
                                      dim3(num_threadblocks),
                                      dim3(threads_per_block),
                                      0, 0,
                                      d_buffers[gpu],
                                      d_peer_ptrs[gpu],
                                      gpu,
                                      num_floats,
                                      step,
                                      num_threadblocks);
                } else {
                    hipLaunchKernelGGL(ring_allreduce_threadblock_kernel,
                                      dim3(num_threadblocks),
                                      dim3(threads_per_block),
                                      0, 0,
                                      d_buffers[gpu],
                                      d_peer_ptrs[gpu],
                                      gpu,
                                      num_floats,
                                      step,
                                      num_threadblocks);
                }
            }
            
            // Synchronize all GPUs after each step
            for (int gpu = 0; gpu < NGPUS; gpu++) {
                CHECK_HIP(hipSetDevice(gpu));
                CHECK_HIP(hipDeviceSynchronize());
                CHECK_HIP(hipGetLastError());  // Check for kernel errors
            }
        }
        
        // Re-initialize for next iteration
        for (int i = 0; i < NGPUS; i++) {
            CHECK_HIP(hipSetDevice(i));
            float init_val = (float)(i + 1);
            
            hipLaunchKernelGGL(init_buffer_kernel,
                              dim3((num_floats + 255) / 256),
                              dim3(256),
                              0, 0,
                              d_buffers[i], init_val, num_floats);
            
            CHECK_HIP(hipDeviceSynchronize());
        }
    }
    
    printf("Warmup complete.\n\n");
    
    // Benchmark runs
    printf("Running %d benchmark iterations...\n", bench_iters);
    
    double total_time_ms = 0.0;
    
    for (int iter = 0; iter < bench_iters; iter++) {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        // Ring allreduce: 2 * (N-1) steps
        for (int step = 0; step < 2 * (NGPUS - 1); step++) {
            for (int gpu = 0; gpu < NGPUS; gpu++) {
                CHECK_HIP(hipSetDevice(gpu));
                
                if (is_warp_mode) {
                    hipLaunchKernelGGL(ring_allreduce_warp_kernel,
                                      dim3(num_threadblocks),
                                      dim3(threads_per_block),
                                      0, 0,
                                      d_buffers[gpu],
                                      d_peer_ptrs[gpu],
                                      gpu,
                                      num_floats,
                                      step,
                                      num_threadblocks);
                } else {
                    hipLaunchKernelGGL(ring_allreduce_threadblock_kernel,
                                      dim3(num_threadblocks),
                                      dim3(threads_per_block),
                                      0, 0,
                                      d_buffers[gpu],
                                      d_peer_ptrs[gpu],
                                      gpu,
                                      num_floats,
                                      step,
                                      num_threadblocks);
                }
            }
            
            // Synchronize all GPUs after each step
            for (int gpu = 0; gpu < NGPUS; gpu++) {
                CHECK_HIP(hipSetDevice(gpu));
                CHECK_HIP(hipDeviceSynchronize());
            }
        }
        
        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_time_ms += elapsed_ms;
        
        // Re-initialize for next iteration (except last)
        if (iter < bench_iters - 1) {
            for (int i = 0; i < NGPUS; i++) {
                CHECK_HIP(hipSetDevice(i));
                float init_val = (float)(i + 1);
                
                hipLaunchKernelGGL(init_buffer_kernel,
                                  dim3((num_floats + 255) / 256),
                                  dim3(256),
                                  0, 0,
                                  d_buffers[i], init_val, num_floats);
                
                CHECK_HIP(hipDeviceSynchronize());
            }
        }
    }
    
    double avg_time_ms = total_time_ms / bench_iters;
    
    // Calculate bandwidth
    // Algorithm bandwidth: total data processed / time
    // For ring allreduce: each GPU processes N bytes in reduce-scatter and N bytes in allgather
    double algbw_gbps = (data_bytes / 1e9) / (avg_time_ms / 1000.0);
    
    // Bus bandwidth: actual data transferred over links
    // For ring allreduce with N GPUs: (N-1)/N * data is sent and received twice
    // Total bus traffic = 2 * data_bytes * (N-1) / N per GPU
    double bus_factor = 2.0 * (NGPUS - 1) / (double)NGPUS;
    double busbw_gbps = (data_bytes / 1e9) * bus_factor / (avg_time_ms / 1000.0);
    
    printf("\n==================== Results ====================\n");
    printf("Time:        %.3f ms\n", avg_time_ms);
    printf("AlgBW:       %.2f GB/s\n", algbw_gbps);
    printf("BusBW:       %.2f GB/s\n", busbw_gbps);
    printf("=================================================\n\n");
    
    // Verification
    printf("Verifying results...\n");
    float expected_value = 36.0f; // Sum of 1+2+3+4+5+6+7+8
    bool all_correct = true;
    int first_error_gpu = -1;
    size_t first_error_idx = 0;
    float first_error_val = 0.0f;
    
    for (int i = 0; i < NGPUS; i++) {
        CHECK_HIP(hipSetDevice(i));
        float* h_buffer = (float*)malloc(data_bytes);
        CHECK_HIP(hipMemcpy(h_buffer, d_buffers[i], data_bytes, hipMemcpyDeviceToHost));
        
        for (size_t j = 0; j < num_floats; j++) {
            if (fabs(h_buffer[j] - expected_value) > 1e-3) {
                if (all_correct) {
                    first_error_gpu = i;
                    first_error_idx = j;
                    first_error_val = h_buffer[j];
                }
                all_correct = false;
                break;
            }
        }
        
        free(h_buffer);
        if (!all_correct) break;
    }
    
    if (all_correct) {
        printf("PASS: All elements equal %.1f\n", expected_value);
    } else {
        printf("FAIL: First error at GPU %d, index %zu: expected %.1f, got %.1f\n",
               first_error_gpu, first_error_idx, expected_value, first_error_val);
    }
    
    // Cleanup
    for (int i = 0; i < NGPUS; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipFree(d_buffers[i]));
        CHECK_HIP(hipFree(d_peer_ptrs[i]));
        free(h_peer_ptrs[i]);
    }
    
    return all_correct ? 0 : 1;
}
