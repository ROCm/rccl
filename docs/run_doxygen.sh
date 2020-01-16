#!/bin/bash
# # Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

if [ -d docBin ]; then
    rm -rf docBin
fi

rm nccl.h

sed -e 's/ROCFFT_EXPORT //g' ../src/nccl.h.in > nccl.h
doxygen Doxyfile
#rm nccl.h

