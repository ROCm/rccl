#!/bin/bash

for numBlocks in 1 2 4 8 16 32; do
		for blockSize in 64 128 256; do
				for numTimers in 1 4 8; do
						for useNuma in 0 1; do
								echo "numBlocks=$numBlocks blockSize=$blockSize numTimers=$numTimers useNuma=$useNuma";
								./LaunchBench $numBlocks $blockSize $numTimers $useNuma &> output.$numBlocks.$blockSize.$numTimers.$useNuma.txt
						done;
				done;
		done;
done;
