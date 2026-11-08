#!/bin/bash
# Builds build/matmul3. -fno-tree-vectorize keeps the kernels scalar, as in Assignment 2
# (the optional SIMD extension is not part of this study), so only the thread count changes.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
gcc -O2 -fno-tree-vectorize -fopenmp -std=c11 -Wall -o build/matmul3 src/matmul3.c -lm
echo "build OK"
