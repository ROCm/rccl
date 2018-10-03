/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include <iostream>
#include <chrono>

constexpr size_t knum_iter = 1 << 4;

class perf_marker {
    std::chrono::high_resolution_clock::time_point start, stop;
    double elapsedSec;

  public:
    perf_marker() { start = std::chrono::high_resolution_clock::now(); }
    void done() {
        stop = std::chrono::high_resolution_clock::now();
        elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                         stop - start)
                         .count();
    }
    void bw(double bytes) {
        double algbw = (double)(bytes) / 1.0E9 / elapsedSec;
        std::cout << "BW: " << algbw << " GB/s" << std::endl;
    }
    double get_bw(double bytes) { return (double)(bytes) / 1.0E9 / elapsedSec; }
};
