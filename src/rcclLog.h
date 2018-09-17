/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include <string>
#include <unordered_map>
#include "rccl/rccl.h"

#define MAKE_STR_PAIR(val) \
    { int(val), #val }

std::unordered_map<int, std::string> umap_red_op = {
    MAKE_STR_PAIR(rcclSum), MAKE_STR_PAIR(rcclProd), MAKE_STR_PAIR(rcclMax),
    MAKE_STR_PAIR(rcclMin)};

std::unordered_map<int, std::string> umap_datatype = {
    MAKE_STR_PAIR(rcclUchar),  MAKE_STR_PAIR(rcclChar),
    MAKE_STR_PAIR(rcclUshort), MAKE_STR_PAIR(rcclShort),
    MAKE_STR_PAIR(rcclUint),   MAKE_STR_PAIR(rcclInt),
    MAKE_STR_PAIR(rcclUlong),  MAKE_STR_PAIR(rcclLong),
    MAKE_STR_PAIR(rcclFloat),  MAKE_STR_PAIR(rcclHalf),
    MAKE_STR_PAIR(rcclDouble)};
