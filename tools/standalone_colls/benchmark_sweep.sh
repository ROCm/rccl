
#!/bin/bash

# Performance sweep script for ring allreduce benchmark
# Collects Time, AlgBW, BusBW across different sizes and modes

WARMUP=100
ITERS=100
SIZES=(268435456 536870912 1073741824 2147483648)  # 256MB, 512MB, 1GB, 2GB
SIZE_NAMES=("256MB" "512MB" "1GB" "2GB")
MODES=("threadblock" "warp")

# Configuration
TB_BLOCKS=28
TB_THREADS=448
WARP_BLOCKS=32
WARP_THREADS=448

echo "================================================================================"
echo "                         RING ALLREDUCE BENCHMARK SWEEP"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Warmup iterations:  $WARMUP"
echo "  Benchmark iterations: $ITERS"
echo "  Number of GPUs:     8"
echo "  Number of rings:    7"
echo ""
echo "Threadblock Mode:"
echo "  Threadblocks/GPU:   $TB_BLOCKS"
echo "  Threads/block:      $TB_THREADS"
echo "  Total threads/GPU:  $((TB_BLOCKS * TB_THREADS))"
echo ""
echo "Warp Mode:"
echo "  Threadblocks/GPU:   $WARP_BLOCKS"
echo "  Threads/block:      $WARP_THREADS"
echo "  Warps/block:        7 (one per ring)"
echo "  Total threads/GPU:  $((WARP_BLOCKS * WARP_THREADS))"
echo "  Total warps/GPU:    $((WARP_BLOCKS * WARP_THREADS / 64))"
echo ""
echo "================================================================================"
echo ""

# Arrays to store results
declare -A times
declare -A algbw
declare -A busbw

# Run benchmarks
for size_idx in "${!SIZES[@]}"; do
    size=${SIZES[$size_idx]}
    size_name=${SIZE_NAMES[$size_idx]}
    
    echo "Running $size_name..."
    
    for mode in "${MODES[@]}"; do
        echo "  Mode: $mode"
        
        # Run benchmark and capture output
        output=$(./ring_allreduce_bench -s $size -m $mode -w $WARMUP -i $ITERS 2>&1)
        
        # Extract metrics using grep and awk
        time_ms=$(echo "$output" | grep "Time:" | awk '{print $2}')
        algbw_val=$(echo "$output" | grep "AlgBW:" | awk '{print $2}')
        busbw_val=$(echo "$output" | grep "BusBW:" | awk '{print $2}')
        
        # Store results
        key="${size_name}_${mode}"
        times[$key]=$time_ms
        algbw[$key]=$algbw_val
        busbw[$key]=$busbw_val
        
        echo "    Time: $time_ms ms, AlgBW: $algbw_val GB/s, BusBW: $busbw_val GB/s"
    done
    echo ""
done

echo ""
echo "================================================================================"
echo "                            PERFORMANCE RESULTS"
echo "================================================================================"
echo ""
echo "Mode Configuration:"
echo "  Threadblock: ${TB_BLOCKS} blocks × ${TB_THREADS} threads = $((TB_BLOCKS * TB_THREADS)) total threads/GPU"
echo "  Warp:        ${WARP_BLOCKS} blocks × ${WARP_THREADS} threads = $((WARP_BLOCKS * WARP_THREADS)) total threads/GPU (224 warps)"
echo ""
echo "Improvement Formula:"
echo "  Time:   (Threadblock - Warp) / Threadblock × 100  [lower time is better]"
echo "  AlgBW:  (Warp - Threadblock) / Threadblock × 100  [higher BW is better]"
echo "  BusBW:  (Warp - Threadblock) / Threadblock × 100  [higher BW is better]"
echo ""
echo "================================================================================"
echo ""

# Pretty formatted table with improvement calculations
printf "%-10s | %12s | %12s | %12s | %s\n" "Size" "Threadblock" "Warp" "Improvement" "Formula"
printf "%-10s | %12s | %12s | %12s | %s\n" "" "Time (ms)" "Time (ms)" "(%)" ""
echo "-----------|--------------|--------------|--------------|-----------------------------"
for size_idx in "${!SIZES[@]}"; do
    size_name=${SIZE_NAMES[$size_idx]}
    tb_time=${times[${size_name}_threadblock]}
    warp_time=${times[${size_name}_warp]}
    
    # Calculate improvement: (TB - Warp) / TB * 100
    improvement=$(awk "BEGIN {printf \"%.2f\", ($tb_time - $warp_time) / $tb_time * 100}")
    
    printf "%-10s | %12.3f | %12.3f | %11.2f%% | (%.3f-%.3f)/%.3f*100\n" \
           "$size_name" "$tb_time" "$warp_time" "$improvement" "$tb_time" "$warp_time" "$tb_time"
done

echo ""
printf "%-10s | %12s | %12s | %12s | %s\n" "Size" "Threadblock" "Warp" "Improvement" "Formula"
printf "%-10s | %12s | %12s | %12s | %s\n" "" "AlgBW(GB/s)" "AlgBW(GB/s)" "(%)" ""
echo "-----------|--------------|--------------|--------------|-----------------------------"
for size_idx in "${!SIZES[@]}"; do
    size_name=${SIZE_NAMES[$size_idx]}
    tb_algbw=${algbw[${size_name}_threadblock]}
    warp_algbw=${algbw[${size_name}_warp]}
    
    # Calculate improvement: (Warp - TB) / TB * 100 (higher is better)
    improvement=$(awk "BEGIN {printf \"%.2f\", ($warp_algbw - $tb_algbw) / $tb_algbw * 100}")
    
    printf "%-10s | %12.2f | %12.2f | %11.2f%% | (%.2f-%.2f)/%.2f*100\n" \
           "$size_name" "$tb_algbw" "$warp_algbw" "$improvement" "$warp_algbw" "$tb_algbw" "$tb_algbw"
done

echo ""
printf "%-10s | %12s | %12s | %12s | %s\n" "Size" "Threadblock" "Warp" "Improvement" "Formula"
printf "%-10s | %12s | %12s | %12s | %s\n" "" "BusBW(GB/s)" "BusBW(GB/s)" "(%)" ""
echo "-----------|--------------|--------------|--------------|-----------------------------"
for size_idx in "${!SIZES[@]}"; do
    size_name=${SIZE_NAMES[$size_idx]}
    tb_busbw=${busbw[${size_name}_threadblock]}
    warp_busbw=${busbw[${size_name}_warp]}
    
    # Calculate improvement: (Warp - TB) / TB * 100 (higher is better)
    improvement=$(awk "BEGIN {printf \"%.2f\", ($warp_busbw - $tb_busbw) / $tb_busbw * 100}")
    
    printf "%-10s | %12.2f | %12.2f | %11.2f%% | (%.2f-%.2f)/%.2f*100\n" \
           "$size_name" "$tb_busbw" "$warp_busbw" "$improvement" "$warp_busbw" "$tb_busbw" "$tb_busbw"
done

echo ""
echo "================================================================================"
echo ""

# Generate CSV tables
echo "CSV Output:"
echo "==========="
echo ""

echo "Table 1: Time (ms)"
echo "Size,Threadblock,Warp,Improvement%"
for size_idx in "${!SIZES[@]}"; do
    size_name=${SIZE_NAMES[$size_idx]}
    tb_time=${times[${size_name}_threadblock]}
    warp_time=${times[${size_name}_warp]}
    improvement=$(awk "BEGIN {printf \"%.2f\", ($tb_time - $warp_time) / $tb_time * 100}")
    echo "$size_name,$tb_time,$warp_time,$improvement"
done

echo ""
echo "Table 2: Algorithm Bandwidth (GB/s)"
echo "Size,Threadblock,Warp,Improvement%"
for size_idx in "${!SIZES[@]}"; do
    size_name=${SIZE_NAMES[$size_idx]}
    tb_algbw=${algbw[${size_name}_threadblock]}
    warp_algbw=${algbw[${size_name}_warp]}
    improvement=$(awk "BEGIN {printf \"%.2f\", ($warp_algbw - $tb_algbw) / $tb_algbw * 100}")
    echo "$size_name,$tb_algbw,$warp_algbw,$improvement"
done

echo ""
echo "Table 3: Bus Bandwidth (GB/s)"
echo "Size,Threadblock,Warp,Improvement%"
for size_idx in "${!SIZES[@]}"; do
    size_name=${SIZE_NAMES[$size_idx]}
    tb_busbw=${busbw[${size_name}_threadblock]}
    warp_busbw=${busbw[${size_name}_warp]}
    improvement=$(awk "BEGIN {printf \"%.2f\", ($warp_busbw - $tb_busbw) / $tb_busbw * 100}")
    echo "$size_name,$tb_busbw,$warp_busbw,$improvement"
done

echo ""
echo "================================================================================"
echo ""
echo "CSV files generated:"
echo "  - benchmark_time.csv"
echo "  - benchmark_algbw.csv"
echo "  - benchmark_busbw.csv"
echo ""
echo "================================================================================"

# Save to CSV files with improvement column
echo "Size,Threadblock,Warp,Improvement%,Formula" > benchmark_time.csv
for size_idx in "${!SIZES[@]}"; do
    size_name=${SIZE_NAMES[$size_idx]}
    tb_time=${times[${size_name}_threadblock]}
    warp_time=${times[${size_name}_warp]}
    improvement=$(awk "BEGIN {printf \"%.2f\", ($tb_time - $warp_time) / $tb_time * 100}")
    echo "$size_name,$tb_time,$warp_time,$improvement,=(B-C)/B*100" >> benchmark_time.csv
done

echo "Size,Threadblock,Warp,Improvement%,Formula" > benchmark_algbw.csv
for size_idx in "${!SIZES[@]}"; do
    size_name=${SIZE_NAMES[$size_idx]}
    tb_algbw=${algbw[${size_name}_threadblock]}
    warp_algbw=${algbw[${size_name}_warp]}
    improvement=$(awk "BEGIN {printf \"%.2f\", ($warp_algbw - $tb_algbw) / $tb_algbw * 100}")
    echo "$size_name,$tb_algbw,$warp_algbw,$improvement,=(C-B)/B*100" >> benchmark_algbw.csv
done

echo "Size,Threadblock,Warp,Improvement%,Formula" > benchmark_busbw.csv
for size_idx in "${!SIZES[@]}"; do
    size_name=${SIZE_NAMES[$size_idx]}
    tb_busbw=${busbw[${size_name}_threadblock]}
    warp_busbw=${busbw[${size_name}_warp]}
    improvement=$(awk "BEGIN {printf \"%.2f\", ($warp_busbw - $tb_busbw) / $tb_busbw * 100}")
    echo "$size_name,$tb_busbw,$warp_busbw,$improvement,=(C-B)/B*100" >> benchmark_busbw.csv
done

echo ""
echo "Done!"
