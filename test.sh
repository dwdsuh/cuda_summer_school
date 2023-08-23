#!/bin/bash
set -e

for i in 1 4 16 32 64 128 256 512 1024
do
    srun ./main -i data/bins/input${i}N.bin -v -a data/bins/answer${i}N.bin
done
