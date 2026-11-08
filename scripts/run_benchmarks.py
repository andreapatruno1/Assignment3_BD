"""Runs the thread-scaling benchmark of build/matmul3 and appends raw rows to results/.

Protocol (fixed before running, same as Assignment 2):
  warm-up      1 run (not reported); repetitions 5 measured runs per configuration
  time budget  20 s for ONE timed run (BUDGET_S, alarm() inside the program); a timeout is written
               to results/limits.csv, never as a time, and larger n are not attempted for that configuration
  memory       budget 3 GB peak resident memory; metric VmHWM of the process
  threads      OMP_NUM_THREADS = p and omp_set_num_threads(p), OMP_DYNAMIC=false; the program records the
               team size that actually ran. No affinity: OMP_PROC_BIND / OMP_PLACES are left unset.
  order        for each n: seq first, then par with every p (seq and par of one n are measured close in time);
               each configuration runs in its own process.

Usage:   python3 scripts/run_benchmarks.py --n 128 256 512 1024 1536 2048 3072 --p 1 2 3 4 8 [--skip-seq]
Outputs: results/raw.csv     method,n,p,team,rep,batch,time_s,cpu_s,checksum,peak_rss_kb
         results/limits.csv  method,n,p,event,limit
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BIN = os.path.join(ROOT, "build", "matmul3")
RES = os.path.join(ROOT, "results")
DATA = os.environ.get("MATMUL_DATA", os.path.join(ROOT, "data"))
WARMUP, REPS, BUDGET_S = 1, 5, 20
HEAD = {"raw.csv": "method,n,p,team,rep,batch,time_s,cpu_s,checksum,peak_rss_kb",
        "limits.csv": "method,n,p,event,limit"}


def append(name, lines):
    path = os.path.join(RES, name)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(HEAD[name] + "\n")
        f.writelines(line + "\n" for line in lines)


def run(method, n, p):
    env = dict(os.environ, BUDGET_S=str(BUDGET_S), OMP_NUM_THREADS=str(p), OMP_DYNAMIC="false")
    env.pop("OMP_PROC_BIND", None)
    env.pop("OMP_PLACES", None)
    r = subprocess.run([BIN, "bench", method, str(n), str(p), str(WARMUP), str(REPS), DATA],
                       capture_output=True, text=True, env=env)
    rows = [x for x in r.stdout.splitlines() if x.strip()]
    if r.returncode == 0:
        append("raw.csv", rows)
    elif r.returncode == 3:
        append("limits.csv", [f"{method},{n},{p},timeout (one multiplication not finished),{BUDGET_S} s"])
    else:
        append("limits.csv", [f"{method},{n},{p},failure exit code {r.returncode},-"])
        print(r.stderr, file=sys.stderr)
    print(method, n, p, "rc", r.returncode, rows[-1] if rows else "", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", nargs="+", type=int, required=True)
    ap.add_argument("--p", nargs="+", type=int, required=True)
    ap.add_argument("--skip-seq", action="store_true", help="do not run seq (already measured for these n)")
    a = ap.parse_args()
    stopped = set()                      # configurations that already timed out
    for n in a.n:
        for method, p in ([] if a.skip_seq else [("seq", 1)]) + [("par", p) for p in a.p]:
            if (method, p) in stopped:
                continue
            if run(method, n, p) != 0:
                stopped.add((method, p))


if __name__ == "__main__":
    main()
