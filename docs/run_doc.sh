#!/bin/bash
# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

if [ -d docBin ]; then
    rm -rf docBin
fi

sed -e 's/ROCFFT_EXPORT //g' ../library/include/rocfft.h > rocfft.h
doxygen Doxyfile

cd source
make clean
make html
cd ..

rm rocfft.h

