#!/bin/bash
# Records the execution environment in results/environment.txt
cd "$(dirname "$0")/.."
{
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== CPU =="; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|NUMA|L1d|L2|L3|Hypervisor"
echo "nproc (execution contexts available to this process): $(nproc)"
echo "== memory =="; free -m
echo "== OS =="; grep PRETTY_NAME /etc/os-release; uname -r
echo "== C =="; gcc --version | head -1; echo "flags: -O2 -fno-tree-vectorize -fopenmp -std=c11 -Wall -lm"
echo "OpenMP: libgomp, _OPENMP = $(echo _OPENMP | gcc -fopenmp -E -P -x c - | tail -1)"
echo "OMP settings: OMP_NUM_THREADS=p, OMP_DYNAMIC=false, OMP_PROC_BIND/OMP_PLACES unset (no affinity), OMP_WAIT_POLICY unset (libgomp default), schedule(static)"
echo "Numerical libraries in the benchmark process: none (no BLAS, no internal threads besides the OpenMP team)"
echo "== Python (inputs and analysis only) =="; python3 --version; python3 -c "import numpy, matplotlib; print('numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)"
} > results/environment.txt
cat results/environment.txt
