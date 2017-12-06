/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include <ctime>

inline void rcclLogger(const char* file, int line, const char* str) {
    time_t now = time(0);
    char *dt = ctime(&now);
    std::cout << "[" << dt <<"]" << file << ": " << line << ": " << str<<std::endl;
}

#define LOG(str) \
        rcclLogger(__FILE__, __LINE__, str);

#define RCCLDEBUG4(one, two, three, four) \
    std::cout<<__FILE__<<" Line:"<<__LINE__<<" func: "<<__func__<<" "<<one<<" "<<two<<" "<<three<<" "<<four<<std::endl;

