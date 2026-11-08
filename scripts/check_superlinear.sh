#!/bin/bash
# Control experiments (interleaved rounds) for the superlinear speedups (S(2) > 2) seen at some n in the
# main sweep and for the one-thread overhead of par.
# Question: is one thread faster when the other vCPU is also busy (host-side frequency / core placement),
# rather than because of the parallel code? Same binary, same inputs, same protocol (1 warm-up + 5 reps),
# n = 1024, 3 interleaved rounds of:
#   seq              sequential baseline alone
#   seq+spinner      the same seq run while a busy loop (pure integer spinning, no memory traffic) keeps the
#                    second vCPU busy: the seq work is unchanged, only the load of the VM changes
#   par p=1, par p=2 as in the main sweep
# Usage: bash scripts/check_superlinear.sh [N] [ROUNDS] [CONDITIONS]
#   defaults: N=1024, ROUNDS="1 2 3", CONDITIONS="seq seq+spinner par1 par2"
#   run for the report: (1) defaults; (2) N=2048, CONDITIONS="seq par1", ROUNDS "1 2" then "3"
# Output: results/raw_check.csv  round,condition,<columns of matmul3 bench>
cd "$(dirname "$0")/.."
DATA=${MATMUL_DATA:-data}
N=${1:-1024}
ROUNDS=${2:-"1 2 3"}
CONDS=${3:-"seq seq+spinner par1 par2"}
OUT=results/raw_check.csv
[ -f $OUT ] || echo "round,condition,method,n,p,team,rep,batch,time_s,cpu_s,checksum,peak_rss_kb" > $OUT
export OMP_DYNAMIC=false BUDGET_S=20
for round in $ROUNDS; do
  for c in $CONDS; do
    case $c in
      seq)  OMP_NUM_THREADS=1 ./build/matmul3 bench seq $N 1 1 5 $DATA | sed "s/^/$round,seq,/" >> $OUT ;;
      seq+spinner)
            sh -c 'while :; do :; done' & SP=$!
            OMP_NUM_THREADS=1 ./build/matmul3 bench seq $N 1 1 5 $DATA | sed "s/^/$round,seq+spinner,/" >> $OUT
            kill $SP; wait $SP 2>/dev/null ;;
      par1) OMP_NUM_THREADS=1 ./build/matmul3 bench par $N 1 1 5 $DATA | sed "s/^/$round,par p=1,/" >> $OUT ;;
      par2) OMP_NUM_THREADS=2 ./build/matmul3 bench par $N 2 1 5 $DATA | sed "s/^/$round,par p=2,/" >> $OUT ;;
    esac
  done
done
python3 - "$OUT" <<'PY'
import csv, sys, numpy as np
from collections import defaultdict
g = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    g[(r["n"], r["condition"])].append(float(r["time_s"]))
for k, v in g.items():
    q1, m, q3 = np.percentile(v, [25, 50, 75]); print(f"{k} reps={len(v)} median={m:.4f} IQR={q3-q1:.4f}")
PY
