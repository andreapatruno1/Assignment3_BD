# Assignment 3 — Parallel Dense Matrix Multiplication with OpenMP (C)

Big Data, ULPGC, 2026-2027 · Andrea Patruno · Report: `Assignment03_Patruno_Report.pdf` (source `report/report.md`)

Question: how does the number of OpenMP threads p affect runtime, speedup S(p) = Tseq/Tp, efficiency E(p) = S(p)/p
and memory of a shared-memory matrix multiplication, compared with its equivalent sequential version?

Reused: i-k-j kernel and input generator of Assignment 2 (https://github.com/andreapatruno1/Assignment2_BD, commit d7ab62f),
whose generator is the one of Assignment 1.

## Contents

| Path | What it is |
|---|---|
| `src/matmul3.c` | `seq` (Assignment 2 i-k-j, sequential baseline), `par` (same row body, `#pragma omp parallel for schedule(static)` over the rows), correctness tests |
| `scripts/build.sh` | `gcc -O2 -fno-tree-vectorize -fopenmp -std=c11 -Wall -o build/matmul3 src/matmul3.c -lm` |
| `scripts/gen_inputs.py` | dense inputs (PCG64, seed 42 + n, uniform [-1,1), raw float64), unchanged from Assignment 2 |
| `scripts/run_benchmarks.py` | main sweep: 1 warm-up + 5 reps, 20 s budget per multiplication, one process per configuration |
| `scripts/check_superlinear.sh` | control experiment (interleaved rounds, busy loop on the second vCPU) |
| `scripts/analyze.py` | `results/summary.csv`, `results/summary_check.csv`, `results/fig/*.png` |
| `scripts/env_info.sh` | `results/environment.txt` (CPU, cores/threads, OS, compiler, OpenMP settings) |
| `results/raw.csv` | every measured repetition: method,n,p,team,rep,batch,time_s,cpu_s,checksum,peak_rss_kb |
| `results/raw_check.csv` | control experiment, same columns + round,condition |
| `results/limits.csv`, `results/correctness.csv` | timeouts; test results |

## Thread configuration

p = 1, 2, 3, 4, 8 (the VM has 2 logical processors: p > 2 is oversubscription). `OMP_NUM_THREADS=p`, `OMP_DYNAMIC=false`,
no affinity (`OMP_PROC_BIND`/`OMP_PLACES` unset), `OMP_WAIT_POLICY` default. The column `team` records the threads that ran.

## Exact commands (Linux, gcc with OpenMP, python3 with numpy and matplotlib)

```bash
bash scripts/build.sh
./build/matmul3 test > results/correctness.csv            # must print ALL TESTS PASSED
python3 scripts/gen_inputs.py                              # data/A_n.bin, data/B_n.bin (~270 MB; MATMUL_DATA=dir to change)
python3 scripts/run_benchmarks.py --n 128 256 512 1024 1536 2048 3072 --p 1 2 3 4 8
bash scripts/check_superlinear.sh                          # n = 1024: seq, seq+spinner, par p=1, par p=2, 3 rounds
bash scripts/check_superlinear.sh 2048 "1 2 3" "seq par1"  # n = 2048: seq vs par p=1, 3 rounds
bash scripts/env_info.sh
python3 scripts/analyze.py
python3 report/build_pdf.py Assignment03_Patruno_Report.pdf   # needs python markdown + playwright
```

Rows are appended: delete `results/raw*.csv` and `results/limits.csv` before a full re-run. In the original run the
sweep was executed in several invocations (`--skip-seq` runs only `par` for sizes whose `seq` was already measured),
because each shell call was limited to 3 minutes; the protocol per configuration is the same.
