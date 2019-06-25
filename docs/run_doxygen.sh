#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

rm nccl.h

sed -e 's/ROCFFT_EXPORT //g' ../src/nccl.h.in > nccl.h
doxygen Doxyfile
#rm nccl.h

